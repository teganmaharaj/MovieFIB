import os, sys, time, copy
from collections import OrderedDict
import json
import numpy
from capgen_vid.iccv15_challenge import RAB_tools
from capgen_vid.iccv15_challenge.RAB_tools import set_config
from jobman import DD, expand
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool
from conv_lstm import init_tparams_fc, fprop_fc, ConvLSTM, \
     visualize, ones, zeros, init_tparams_matrix
from capgen_vid.video_qa import common
from capgen_vid.video_qa.common import adam
import data_engine
from popstats import get_stat_estimator
from nunits import NTanhP, NSigmoidP, NTanh

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

    def get_fprop(self, video, video_mask, label, popstats=None):
        if self.options.rec_bn and self.options.use_popstats:
            popstats_enc = popstats
        else:
            popstats_enc = None

        ''' Encoder '''
        m, t, x, y = video.shape
        state_vid = video.dimshuffle(1,0,2,3,4)
        state_mask = video_mask.dimshuffle(1,0)
        for l in xrange(0, len(self.encoders)):
            ### Conv RNN
            rvals = self.layers[0].fprop(state_vid, state_mask, popstats=popstats_enc)
            state_vid = rvals[0]
            ### Spatial Pool
            if self.options.layer_pool[0] != (0, 0):
                state_vid = dnn_pool(state_vid.reshape((t*m,x, y)),
                                    size=self.options.pool[0], stride=self.options.pool_stride[0]))
                state_vid = state_vid.reshape(t, m, x, y)


        ''' Classifier '''
        video_ctx = state_vid[0].flatten(2) # (m,d)
        output = T.dot(logit, self.W) + self.b
        y_hat = T.nnet.softmax(output)

        '''cost'''
        ce = tensor.nnet.categorical_crossentropy(y_hat, label)
        cost = ce.mean()

        ### Returns Err as well
        return [cost, ce]

    def build_model(self):
        #theano.config.compute_test_value = "warn"

        ''' Create Theano Variables '''
        video = ftensor5('video') # (m,f,c,x,y)'
        video.tag.test_value = self.test_values[0]
        video_mask = T.fmatrix('video_mask') # (m,f)
        video_mask.tag.test_value = self.test_values[1]
        label = T.lvector('label') # (m,)
        label.tag.test_value = self.test_values[2]
        popstats = OrderedDict()
        for l in xrange(options.nlayers):
            popstats['encoder_%d_x_mean' % l] = ftensor4('x_mean')
            popstats['encoder_%d_x_var'  % l] = ftensor4('x_var')
            popstats['encoder_%d_h_mean' % l] = ftensor4('h_mean')
            popstats['encoder_%d_h_var'  % l] = ftensor4('h_var')
            popstats['encoder_%d_c_mean' % l] = ftensor4('c_mean')
            popstats['encoder_%d_c_var'  % l] = ftensor4('c_var')


        ''' Initialize model param '''
        self.params = []
        self.layers = []
        for l in xranges(len(self.options.num_filters)):
            ### FIXME use convgru instead
           conv_gru = layers.ConvLSTM(options=self.options,
                                      num_channel=self.options.num_filters[i]
                                      num_filters=self.options.num_filters[i+1],
                                      filter_size=self.options.filter_size[i], padding=(0,0),
                                      mem_size=self.options.mem_size[i],
                                      prefix="encoder_%d" % l)
           self.params += conv_gru.params
           self.layers.append(conv_gru)
           out_dim = self.options.num_filters[i+1]  * np.prod(self.options.mem_size[i])
        '''Classification '''
        self.W, self.b = init_tparams_fc(nin=out_dim, nout=self.options.nclasses, prefix='logit')
        self.params += [self.W, self.b]

        ''' Construct theano grah '''
        [cost, ce] = self.get_fprop(video, video_mask, label)

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
        print 'compile train fns'
        self.f_grad_shared, self.f_update = eval(self.options.optimizer)(
            T.scalar(name='lr'), self.params, grads,
            [video, video_mask, label], cost,
            extra=[ce])
        self.f_train = theano.function([video, video_mask, label], ce,
                                       on_unused_input='warn')

        ''' Batch Norm population graph '''
        print 'get estimate for inference'
        symbatchstats, estimators = get_stat_estimator([ce])
        sample_stat_inputs = []
        self.f_stat_estimator = None
        if len(estimators) > 0:
            self.f_stat_estimator = theano.function(
                [video, video_mask, label], estimators,
                on_unused_input='warn')
            self.options.use_popstats = True
            for v in symbatchstats:
                sample_stat_inputs.append(popstats[v.tag.bn_label])
        # Get inference graph
        [cost, ce] = self.get_fprop(video, video_mask, label,
                                    popstats=popstats)
        self.options.use_popstats = False
        print 'compile inference fns'
        self.f_inference = theano.function([video, video_mask, label], [cost, ce],
                                           on_unused_input='warn',
                                           name='f_inference')
        print 'compiling theano fns used %.2f sec'%(time.time()-t0)
