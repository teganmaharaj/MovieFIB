import os, sys, time, copy
from collections import OrderedDict
import json
import numpy
import RAB_tools
from RAB_tools import set_config
from jobman import DD, expand
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool
from layers import init_tparams_fc, fprop_fc, LSTM, ones, zeros, init_tparams_matrix, bn_ff
import common
from common import adam
import data_engine
from popstats import get_stat_estimator

import layers

rng_numpy, rng_theano = RAB_tools.get_two_rngs()
ftensor5 = T.TensorType('float32', (False,)*5)
ftensor4 = T.TensorType('float32', (False,)*4)
use_noise = theano.shared(numpy.float32(0.), name='use_noise')
dropout_flag = theano.shared(0.)
zoneout_flag = theano.shared(0.)

class Model(object):
    def __init__(self, options, test_values):
        self.options = options
        self.test_values = test_values

    def get_fprop(self,
                  features, features_mask,
                  ctx_before, ctx_before_mask,
                  ctx_after, ctx_after_mask,
                  label, popstats=None):
        if self.options.use_bn and self.options.use_popstats:
            popstats_enc = popstats
        else:
            popstats_enc = None

        ''' Encoder '''
        m, t, c = features.shape

        features = features.dimshuffle(1, 0, 2)
        features_mask =  features_mask.dimshuffle(1, 0)
        ctx_before = ctx_before.dimshuffle(1, 0)
        ctx_before_mask = ctx_before_mask.dimshuffle(1, 0)
        ctx_after = ctx_after.dimshuffle(1, 0)
        ctx_after_mask = ctx_after_mask.dimshuffle(1, 0)

        ## Reverse context_after
        ctx_after = ctx_after[::-1, :]
        ctx_after_mask = ctx_after_mask[::-1, :]

        ### Word embedding
        ctx_b_emb = self.Wemb[ctx_before.flatten()].reshape(
            [ctx_before.shape[0], ctx_before.shape[1], self.options.dim_word])
        ctx_a_emb = self.Wemb[ctx_after.flatten()].reshape(
            [ctx_after.shape[0], ctx_after.shape[1], self.options.dim_word])

        ### LSTM encoding
        vid_rvals = self.lstm_vid.fprop(features, features_mask, popstats=popstats_enc)
        ctx_b_rvals = self.lstm_ctxb.fprop(ctx_b_emb, ctx_before_mask, popstats=popstats_enc)
        ctx_a_rvals = self.lstm_ctxa.fprop(ctx_a_emb, ctx_after_mask, popstats=popstats_enc)

        vid_ctx = vid_rvals[0]
        ctx_b = ctx_b_rvals[0]
        ctx_a = ctx_a_rvals[0]

        ### Final classification
        if self.options.text_only:
            ctx = T.concatenate([ctx_b[-1, :, :], ctx_a[-1, :, :]], axis=1)
        else:
            ctx = T.concatenate([vid_ctx[-1, :, :], ctx_b[-1, :, :], ctx_a[-1, :, :]], axis=1)
        if self.options.use_dropout:
            dropout_mask = T.switch(
                dropout_flag,
                rng_theano.binomial(ctx.shape, p=0.5, n=1, dtype='float32'),
                T.ones_like(ctx) * 0.5)
            ctx = ctx * dropout_mask

        output = T.dot(ctx, self.W) + self.b
        y_hat = T.nnet.softmax(output)

        '''cost'''
        ce = T.nnet.categorical_crossentropy(y_hat, label)
        err = T.neq(label, T.argmax(y_hat, axis=1))
        cost = ce.mean()

        return [cost, ce, err, y_hat]

    def build(self):
        if self.options.debug:
            theano.config.compute_test_value = "warn"

        ''' Create Theano Variables '''
        features = T.ftensor3('features') # (m,f,c)'
        features.tag.test_value = self.test_values[0]
        features_mask = T.fmatrix('features_mask') # (m,f)
        features_mask.tag.test_value = self.test_values[1]

        context_b = T.lmatrix('context b') # (m,f,c)'
        context_b.tag.test_value = self.test_values[2]
        context_b_mask = T.fmatrix('context_b_mask') # (m,f)
        context_b_mask.tag.test_value = self.test_values[3]

        context_a = T.lmatrix('context a') # (m,f,c)'
        context_a.tag.test_value = self.test_values[4]
        context_a_mask = T.fmatrix('context_a_mask') # (m,f)
        context_a_mask.tag.test_value = self.test_values[5]

        label = T.lvector('label') # (m,)
        label.tag.test_value = self.test_values[6]


        ''' Initialize model param '''
        self.params = []
        self.Wemb = theano.shared(
            common.norm_weight(self.options.n_words,
                               self.options.dim_word), name='Wemb')
        if self.options.train_emb:
            self.params.append(self.Wemb)

        self.lstm_vid = layers.LSTM(options=self.options,
                                    num_inputs=self.options.input_dim,
                                    num_hiddens=self.options.hdims,
                                    prefix="lstm_vid")
        self.lstm_vid.init_tparams()
        if not self.options.text_only:
            self.params += self.lstm_vid.params

        self.lstm_ctxb = layers.LSTM(options=self.options,
                                     num_inputs=self.options.dim_word,
                                     num_hiddens=self.options.hdims,
                                     prefix="ctxb_lstm")
        self.lstm_ctxb.init_tparams()
        self.params += self.lstm_ctxb.params

        self.lstm_ctxa = layers.LSTM(options=self.options,
                                     num_inputs=self.options.dim_word,
                                     num_hiddens=self.options.hdims,
                                     prefix="ctxa_lstm")
        self.lstm_ctxa.init_tparams()
        self.params += self.lstm_ctxa.params

        ''' Get population statistics '''
        popstats = OrderedDict()
        popstats.update(self.lstm_vid.get_popstat())
        popstats.update(self.lstm_ctxb.get_popstat())
        popstats.update(self.lstm_ctxa.get_popstat())
        popstats['ff_x_mean'] = T.fvector('x_mean_ff')
        popstats['ff_x_var'] = T.fvector('x_var_ff')


        '''Classification '''
        n_words = self.options.n_words_out
        if self.options.text_only:
            self.W, self.b = init_tparams_fc(nin=2*self.options.hdims,
                                             nout=n_words, prefix='logit')
        else:
            self.W, self.b = init_tparams_fc(nin=3*self.options.hdims,
                                             nout=n_words, prefix='logit')
        self.params += [self.W, self.b]

        ''' Construct theano grah '''
        [cost, ce, err, y_hat] = self.get_fprop(features, features_mask,
                                                context_b, context_b_mask,
                                                context_a, context_a_mask,
                                                label)

        '''Compute gradient'''
        # [grad_dummy_h_vid, grad_dummy_c_vid, grad_dummy_xW_hU_b_vid] = T.grad(
        #     cost, [dummy_h_vid, dummy_c_vid, dummy_xW_hU_b_vid])
        grads = T.grad(cost, wrt=self.params)
        if self.options.clip_c > 0.:
            g2 = 0.
            for g in grads:
                g2 += (g**2).sum()
            new_grads = []
            for g in grads:
                new_grads.append(
                    T.switch(g2 > (self.options.clip_c**2),
                             g / T.sqrt(g2) * self.options.clip_c, g))
            grads = new_grads
        print 'start compiling theano fns'
        t0 = time.time()

        inputs = [features, features_mask,
                  context_b, context_b_mask,
                  context_a, context_a_mask,
                  label]
        print 'compile train fns'
        self.f_grad_shared, self.f_update = eval(self.options.optimizer)(
            T.scalar(name='lr'), self.params, grads,
            inputs, cost,
            extra=[ce, err])
        self.f_train = theano.function(inputs, [cost, ce, err, y_hat], name='f_train', on_unused_input='warn')
        ### Write y from y_hat and dictionary

        ''' Batch Norm population graph '''
        print 'get estimate for inference'
        symbatchstats, estimators = get_stat_estimator([ce])
        sample_stat_inputs = []
        self.f_stat_estimator = None
        if len(estimators) > 0:
            self.f_stat_estimator = theano.function(
                [features, features_mask,
                 context_a, context_a_mask,
                 context_b, context_b_mask], estimators,
                on_unused_input='warn')
            self.options.use_popstats = True
            for v in symbatchstats:
                print v.tag.bn_label
                sample_stat_inputs.append(popstats[v.tag.bn_label])
            # Get inference graph
            [cost, ce, err, y_hat] = self.get_fprop(features, features_mask,
                                                    context_b, context_b_mask,
                                                    context_a, context_a_mask,
                                                    label,
                                                    popstats=popstats)
        self.options.use_popstats = False
        print 'compile inference fns'
        inputs = inputs + sample_stat_inputs
        self.f_inference = theano.function(inputs, [cost, ce, err, y_hat],
                                           on_unused_input='warn',
                                           name='f_inference')
        print 'compiling theano fns used %.2f sec'%(time.time()-t0)



class MainLoop(object):
    def __init__(self, options):
        self.options = options
        self.engine = data_engine.LSMDC(options)
        self.options.n_words = self.engine.len_vocab
        self.options.n_words_out = self.engine.len_vocab
        import pdb; pdb.set_trace()
        if self.options.use_out_vocab:
            self.options.n_words_out = len(self.engine.out_vocab.keys())
        self.test_values = self.engine.get_test_value()
        self.model = Model(self.options, self.test_values)
        RAB_tools.dump_pkl(self.options, self.options.save_model_dir+'options.pkl')

    def save_model(self):
        print 'saving best model'
        t0 = time.time()
        D = {}
        for tparam in self.model.params:
            D[tparam.name] = tparam.get_value()
        numpy.savez(self.options.save_model_dir+'model_params_best.npz', **D)
        print 'saving model took %.2f'%(time.time()-t0)


    def load_model(self):
        import pdb; pdb.set_trace()
        print 'loading model from:', self.options.load_model_from + 'model_params_best.npz'
        params = numpy.load(self.options.load_model_from + 'model_params_best.npz')
        t0 = time.time()
        for p in params.keys():
            print p, params[p].shape
        for tparam in self.model.params:
            if tparam.name in params.keys() and tparam.get_value().shape == params[tparam.name].shape:
                print tparam.name, tparam.get_value().shape, params[tparam.name].shape
                assert tparam.get_value().shape == params[tparam.name].shape
                tparam.set_value(params[tparam.name])
            else:
                print "Warning, no value for %s"  % tparam.name
        print 'loading model took %.2f'%(time.time()-t0)

    def get_population_statistics(self, max_batch=None):
        def copy_padding(array, maxlen):
            padding = []
            padding.append((0, max(0, maxlen - array.shape[0])))
            for i in xrange(1, array.ndim):
                padding.append((0, 0))
                array = numpy.pad(array, padding, mode="edge")
            return array
        additional_stat_inputs = []

        if self.model.f_stat_estimator is not None:
            t0 = time.time()
            nb = 0.0
            self.engine.send_request('train')
            while True:
                print nb, '/', max_batch
                [data, epoch_flag] = self.engine.fetch_mb('train')
                if epoch_flag:
                    break;
                if max_batch and nb > max_batch:
                    break
                video_, video_mask_, q_b, q_b_mask, q_a, q_a_mask, label_ = data
                nb += 1.0
                tmp = self.model.f_stat_estimator(video_, video_mask_,
                                                  q_b, q_b_mask,
                                                  q_a, q_a_mask)
                if len(additional_stat_inputs) == 0:
                    additional_stat_inputs = tmp
                else:
                    for i in xrange(len(additional_stat_inputs)):
                        ## make sure they have the same temporal dim
                        max_t = max(additional_stat_inputs[i].shape[0],
                                    tmp[i].shape[0])
                        tmp[i] = copy_padding(tmp[i], max_t)
                        additional_stat_inputs[i] = copy_padding(additional_stat_inputs[i], max_t)
                        additional_stat_inputs[i] += tmp[i]
            for i in xrange(len(additional_stat_inputs)):
                additional_stat_inputs[i] /= nb
            print 'Compute popstat in %.2f'%(time.time()-t0),
        return additional_stat_inputs



    def validation(self, counter, unconditional=False):
        self.engine.send_request('valid')
        nlls = []
        errs = []
        counter = 0
        total = len(self.engine.kf_valid)
        while True:
            [data, epoch_flag] = self.engine.fetch_mb('valid')
            if epoch_flag:
                break
            if self.options.estimate_population_statistics and self.model.f_stat_estimator is not None:
                additional_stat_inputs = []
                additional_stat_inputs = self.get_population_statistics(max_batch=300)
                cost, nll, err, y_hat = self.model.f_inference(data + additional_stat_inputs)
            else:
                print [d.shape for d in data]
                cost, nll, err, y_hat   = self.model.f_train(data[0], data[1],
                                                             data[2], data[3],
                                                             data[4], data[5],
                                                             data[6])
            nlls += nll.tolist()
            errs += err.tolist()
            counter += 1
            print 'validation %d / %d'%(counter, total)
        return numpy.mean(nlls), numpy.mean(errs)


    def train(self):

        self.model.build()

        if self.options.load_model_from is not None:
            self.load_model()

        if self.options.do_eval:

            additional_stat_inputs = []
            if self.model.f_stat_estimator is not None:
                additional_stat_inputs = self.get_population_statistics(max_batch=100)

            self.engine.send_request('test')
            nlls = []
            errs = []
            counter = 0
            total = len(self.engine.kf_test)
            predictions  = []


            def to_text(pre_caption, pre_mask,
                        post_caption, post_mask,
                        yhat, target):
                pre  = []
                post = []

                for i in xrange(pre_mask.sum()):
                    pre.append(self.engine.vocab[pre_caption[i]])
                for i in xrange(post_mask.sum()):
                    post.append(self.engine.vocab[post_caption[i]])
                if self.options.use_out_vocab:
                    return [pre, post, self.engine.out_vocab[target+1], self.engine.out_vocab[yhat.argmax() + 1]]
                else:
                    return [pre, post, self.engine.vocab[target], self.engine.vocab[yhat.argmax()]]


            while True:
                [data, epoch_flag] = self.engine.fetch_mb('test')
                if epoch_flag:
                    break

                pre_capt = data[2]
                pre_mask = data[3]
                post_capt = data[4]
                post_mask = data[5]
                target = data[6]

                if self.model.f_stat_estimator is not None:
                    inputs = data+additional_stat_inputs
                    cost, nll, err, y_hat = self.model.f_inference(*inputs)
                else:
                    print [d.shape for d in data]
                    cost, nll, err, y_hat   = self.model.f_train(data[0], data[1],
                                                                 data[2], data[3],
                                                                 data[4], data[5],
                                                                 data[6])
                nlls += nll.tolist()
                errs += err.tolist()
                counter += 1


                for i in xrange(target.shape[0]):
                    cur = to_text(pre_capt[i, :], pre_mask[i, :],
                                  post_capt[i, :], post_mask[i, :],
                                  y_hat[i, :], target[i])
                    predictions.append(cur)

                print 'Done %d / %d'%(counter, total)
            RAB_tools.dump_pkl(predictions, self.options.save_model_dir+'predictions.pkl')
            print numpy.mean(nlls), numpy.mean(errs)
            sys.exit(1)


        self.engine.send_request('update')
        epoch_counter = -1
        update_counter = 0
        minibatch_counter = 0
        patience_counter = 0
        moving_cost = 0.
        n_minibatch_total = len(self.engine.kf_train)
        nll_train = []
        err_train = []
        record = []
        record_moving_cost = []
        record_moving_err = []
        t00 = time.time()
        while True:
            if epoch_counter > self.options.max_epoch:
                break
            t0 = time.time()
            [minibatch_data, epoch_flag] = self.engine.fetch_mb('update')
            t1 = time.time()
            if epoch_counter == -1 or epoch_flag:
                dropout_flag.set_value(0.)
                zoneout_flag.set_value(0.)
                use_noise.set_value(numpy.float32(0.))
                epoch_counter += 1
                minibatch_counter = 0
                train_cost = numpy.mean(nll_train)
                train_err = numpy.mean(err_train)
                nll_train = []
                err_train = []
                valid_cost, valid_err = self.validation(epoch_counter)
                if len(record) >= 1:
                    # save best model
                    history_best = numpy.min([r[-2] for r in record])
                    print 'Valid:', valid_err, '/', history_best
                    if valid_err < history_best:
                        patience_counter = 0
                        self.save_model()
                    else:
                        patience_counter += 1
                print "Validation %d -- train_cost: %.2f, train_err: %.2f, valid_cost: %.2f, valid_err: %.2f" % (epoch_counter,
                                                                                                                 train_cost, train_err,
                                                                                                                 valid_cost, valid_err)
                record.append([epoch_counter, update_counter,
                               train_cost, valid_cost,
                               train_err, valid_err, time.time()-t00])
                numpy.savetxt(self.options.save_model_dir+'record.txt', record, fmt='%.4f')
                self.engine.send_request('update')
                continue

            t2 = time.time()
            [features, features_mask,
             before, before_mask, after, after_mask,
             target] = minibatch_data
            dropout_flag.set_value(1.)
            zoneout_flag.set_value(1.)
            use_noise.set_value(numpy.float32(1.))
            [cost, nll, err ] = self.model.f_grad_shared(features, features_mask,
                                                         before, before_mask,
                                                         after, after_mask,
                                                         target)
            nll_train += nll.tolist()
            err_train += err.tolist()
            dropout_flag.set_value(0.)
            zoneout_flag.set_value(0.)
            if numpy.isnan(cost):
                print 'NaN cost'
                import pdb; pdb.set_trace()
            if epoch_counter == 0 and update_counter == 0:
                moving_cost = cost
                moving_err = numpy.mean(err)
            else:
                moving_cost = moving_cost * 0.95 + cost * 0.05
                moving_err = moving_err  * 0.95 + numpy.mean(err) * 0.05
            record_moving_cost.append(moving_cost)
            record_moving_err.append(moving_err)
            t3 = time.time()
            # if update_counter % self.options.visFreq == 0:
            #     if self.options.encoder:
            #         encoder = self.layers
            #         visualize(self.options.save_model_dir,
            #                   update_counter, record_moving_cost,
            #                   video, state_vid, memory_vid, delta_c_vid, delta_c_2_vid,
            #                   g_i_vid, g_f_vid, g_o_vid, g_g_vid, xW_b_vid, xW_hU_b_vid,
            #                   grad_dummy_h, grad_dummy_c, grad_dummy_xW_hU_b,
            #                   encoder.W_i,
            #                   encoder.W_o,
            #                   encoder.W_f,
            #                   encoder.W_g,
            #                   encoder.U_i,
            #                   encoder.U_o,
            #                   encoder.U_f,
            #                   encoder.U_g
            #                   )
            #     LSTM_Conditional.visualize(
            #         state_text, memory_text, g_i_text, g_f_text, g_o_text, g_g_text,
            #         alpha_text, hM_text, cN_text, xW_b_text, hU_text, cC_text,
            #         emb_text, video_ctx,
            #         update_counter, self.options.save_model_dir, caption_mask, self.decoder)
            t4 = time.time()
            self.model.f_update(self.options.lr)
            update_counter += 1
            minibatch_counter += 1
            t5 = time.time()

            if update_counter % self.options.dispFreq == 0:
                print '(fetch %.2f, update %.2f, run time %.2f) epoch %d, %d / %d minibatches, moving cost %.3f, moving err %.3f, save_dir %s'%(
                    t1-t0, t3-t2 + t5-t4, t5-t00, epoch_counter, minibatch_counter,
                    n_minibatch_total, moving_cost, moving_err,
                    self.options.save_model_dir)
        print 'train over, exp dir ', self.options.save_model_dir


def experiment(state, channel=None):
    from config import options
    set_config(options, state)
    save_model_dir = options.save_model_dir
    if options.load_options_from is not None:
        print 'loading options from %s'%options.load_options_from
        options = RAB_tools.load_pkl(options.load_options_from)
        options.load_options_from = False
        options.save_model_dir = save_model_dir
    if options.save_model_dir == 'current':
        options.save_model_dir = './'
    RAB_tools.create_dir_if_not_exist(options.save_model_dir)
    #if options.erase:
    #    os.system('rm %s*.*'%options.save_model_dir)
    loop = MainLoop(options)
    loop.train()

if __name__ == '__main__':
    args = {}
    try:
        for arg in sys.argv[1:]:
            k, v = arg.split('=')
            args[k] = v
    except:
        print 'args must be like a=X b.c=X'
        exit(1)
    state = expand(args)
    experiment(state)
