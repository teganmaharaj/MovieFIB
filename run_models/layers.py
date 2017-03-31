import os, sys, time, socket
from collections import OrderedDict
import numpy
import tables
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool
import RAB_tools

from jobman import DD, expand
import common
import data_engine

rng_numpy, rng_theano = RAB_tools.get_two_rngs()
ftensor5 = T.TensorType('float32', (False,)*5)

def zeros(shape):
    return numpy.zeros(shape, dtype=theano.config.floatX)

def ones(shape):
    return numpy.ones(shape, dtype=theano.config.floatX)


# batch normalization for fully connected layer
def bn_ff(x, mean, var, gamma=1., beta=0., prefix="",
          axis=0, use_popstats=False):
    assert x.ndim == 2
    if not use_popstats:
        mean = x.mean(axis=axis)
        var = x.var(axis=axis)
        mean.tag.bn_statistic = True
        mean.tag.bn_label = prefix + "_mean"
        var.tag.bn_statistic = True
        var.tag.bn_label = prefix + "_var"

    var_corrected = var + 1e-7
    y = theano.tensor.nnet.bn.batch_normalization(
        inputs=x,
        gamma=gamma, beta=beta,
        mean=T.shape_padleft(mean),
        std=T.shape_padleft(T.sqrt(var_corrected)))
    assert y.ndim == 2
    return y, mean, var

# batch normalization for convolutional layer
def bn(x, gammas, betas, mean, var, args, axis=0, prefix="",
       normalize=True, use_popstats=False):
    assert x.ndim == 2
    if not normalize or not use_popstats:
        ### FIXME do we want to share statistics across space as well?
        mean = x.mean(axis=axis)
        var = x.var(axis=axis)
        mean.tag.bn_statistic = True
        mean.tag.bn_label = prefix + "_mean"
        var.tag.bn_statistic = True
        var.tag.bn_label = prefix + "_var"

    if not args.use_bn:
        y = x + betas
    else:
        var_corrected = var + 1e-7
        y = theano.tensor.nnet.bn.batch_normalization(
            inputs=x, gamma=gammas, beta=betas,
            mean=T.shape_padleft(mean), std=T.shape_padleft(T.sqrt(var_corrected)),
            mode="low_mem")
    return y, mean, var

def init_tparams_fc(nin, nout, prefix, scale=0.01):
    W = theano.shared(common.norm_weight(nin, nout, scale), name='%s_W'%prefix)
    b = theano.shared(numpy.zeros((nout,), dtype='float32'),
                           name='%s_b'%prefix)
    return W, b

def init_tparams_matrix(nin, nout, prefix, scale=0.01):
    W = theano.shared(common.norm_weight(nin, nout, scale), name='%s_W'%prefix)
    return W

def fprop_fc(W, b, x, activation='rectifier', use_dropout=False, dropout_flag=None):
    # x (m, f, dim_frame)
    y = T.dot(x, W) + b
    if activation == 'rectifier':
        y = common.rectifier(y)
    elif activation == 'tanh':
        y = T.tanh(y)
    elif activation == 'linear':
        pass
    else:
        raise NotImplementedError()
    if use_dropout:
        print 'lstm uses dropout'
        assert dropout_flag is not None
        dropout_mask = T.switch(
            dropout_flag,
            rng_theano.binomial(y.shape, p=0.5, n=1, dtype='float32'),
            T.ones_like(y) * 0.5)
        y = dropout_mask * y
    return y



class LSTM(object):
    def __init__(self, options, num_hiddens, num_inputs,
                 prefix, scale=0.001):
        print 'init _lstm: ' + prefix

        self.options = options
        self.prefix = prefix
        self.num_hiddens = num_hiddens
        self.num_inputs = num_inputs
        self.scale = scale

        self.use_bn = options['use_bn']
        self.use_popstat = options['use_popstats']

    def init_tparams(self):
        # for embedding
        W_i = common.norm_weight_tensor((self.num_inputs, self.num_hiddens), self.scale)
        self.W_i = theano.shared(W_i, name='%s_lstm_W_i'%self.prefix)
        W_o = common.norm_weight_tensor((self.num_inputs, self.num_hiddens), self.scale)
        self.W_o = theano.shared(W_o, name='%s_lstm_W_o'%self.prefix)
        W_f = common.norm_weight_tensor((self.num_inputs, self.num_hiddens), self.scale)
        self.W_f = theano.shared(W_f, name='%s_lstm_W_f'%self.prefix)
        W_g = common.norm_weight_tensor((self.num_inputs, self.num_hiddens), self.scale)
        self.W_g = theano.shared(W_g, name='%s_lstm_W_g'%self.prefix)
        self.W = T.concatenate([self.W_i, self.W_f, self.W_o, self.W_g], axis=1)

        # for h
        U_i = common.norm_weight_tensor((self.num_hiddens, self.num_hiddens), self.scale)
        self.U_i = theano.shared(U_i, name='%s_lstm_U_i'%self.prefix)
        U_o = common.norm_weight_tensor((self.num_hiddens, self.num_hiddens), self.scale)
        self.U_o = theano.shared(U_o, name='%s_lstm_U_o'%self.prefix)
        U_f = common.norm_weight_tensor((self.num_hiddens, self.num_hiddens), self.scale)
        self.U_f = theano.shared(U_f, name='%s_lstm_U_f'%self.prefix)
        U_g = common.norm_weight_tensor((self.num_hiddens, self.num_hiddens), self.scale)
        self.U_g = theano.shared(U_g, name='%s_lstm_U_g'%self.prefix)
        self.U = T.concatenate([self.U_i, self.U_f, self.U_o, self.U_g], axis=1)

        ### BN/bias parameter
        self.x_gammas = theano.shared(
            self.options.initial_gamma * ones((4 * self.num_hiddens,)), name="%s_x_gammas"%self.prefix)
        self.h_gammas = theano.shared(
            self.options.initial_gamma * ones((4 * self.num_hiddens,)), name="%s_h_gammas)"%self.prefix)
        self.xh_betas = theano.shared(
            self.options.initial_beta  * ones((4 * self.num_hiddens,)), name="%s_xh_betas"%self.prefix)
        self.c_gammas = theano.shared(
            self.options.initial_gamma * ones((self.num_hiddens,)), name="%s_c_gammas"%self.prefix)
        self.c_betas  = theano.shared(
            self.options.initial_beta  * ones((self.num_hiddens,)), name="%s_c_betas"%self.prefix)

        ### Forget biais init
        forget_biais = self.xh_betas.get_value()
        forget_biais[self.num_hiddens:2*self.num_hiddens] = 1.
        self.xh_betas.set_value(forget_biais)

        ### Init h/c value
        self.h0 = theano.shared(
            zeros((self.num_hiddens)), name="%s_h0"%self.prefix)
        self.c0 = theano.shared(
            zeros((self.num_hiddens)), name="%s_c0"%self.prefix)

        if not self.use_bn:
            self.params = [self.W_i, self.W_f, self.W_o, self. W_g,
                           self.U_i, self.U_f, self.U_o, self. U_g,
                           self.xh_betas,
                           self.c_betas,
                           self.h0, self.c0]
        else:
            self.params = [self.W_i, self.W_f, self.W_o, self. W_g,
                           self.U_i, self.U_f, self.U_o, self. U_g,
                           self.x_gammas, self.h_gammas,
                           self.xh_betas,
                           self.c_gammas, self.c_betas,
                           self.h0, self.c0]

    def get_popstat(self):
        popstats = OrderedDict()

        if self.options.use_bn:
            popstats['%s_x_mean' % self.prefix] = T.fmatrix('%s_x_mean' % self.prefix)
            popstats['%s_x_var'  % self.prefix] = T.fmatrix('%s_x_var' % self.prefix)
            popstats['%s_h_mean' % self.prefix] = T.fmatrix('_%s_h_mean' % self.prefix)
            popstats['%s_h_var'  % self.prefix] = T.fmatrix('%s_h_var' % self.prefix)
            popstats['%s_c_mean' % self.prefix] = T.fmatrix('%s_c_mean' % self.prefix)
            popstats['%s_c_var'  % self.prefix] = T.fmatrix('%s_c_var' % self.prefix)

        return popstats

    def fprop(self, x, mask, dropout_flag=None, popstats=None):
        # x (t, m, c)
        n_step, n_sample, n_channel = x.shape

        # used to check the gradient
        dummy_h = T.alloc(0., n_step, n_sample, self.num_hiddens)
        dummy_c = T.alloc(0., n_step, n_sample, self.num_hiddens)
        init_state = self.h0
        init_memory = self.c0

        # (t*m, 4*f, w, h)
        W = self.W
        U = self.U
        # for one by one convolution, direct matrix multiply is faster
        # (t*m, 4f)
        state_below = T.dot(x, self.W)

        def _slice(_x, n, dim):
            if _x.ndim == 4:
                return _x[:, n*dim:(n+1)*dim, :, :]
            elif _x.ndim == 3:
                return _x[n*dim:(n+1)*dim, :, :]
            elif _x.ndim == 2:
                return _x[:, n*dim:(n+1)*dim]

        def step(x_, m_, dummy_h_, dummy_c_,
                 pop_means_x, pop_means_h, pop_means_c,
                 pop_vars_x, pop_vars_h, pop_vars_c,
                 h_, c_,
                 U,
                 x_gammas, h_gammas, xh_betas,
                 c_gammas, c_betas):

            # use matrix multiply to replace 1 by 1 conv, more efficient
            preact = T.dot((h_ + dummy_h_), U)

            ### BN
            if not self.options.use_bn:
                x_normal, x_mean, x_var = bn(
                    x_, 1.0, xh_betas, pop_means_x, pop_vars_x, self.options, prefix=self.prefix+"_x")
                h_normal, h_mean, h_var = bn(
                    preact, 1.0, 0, pop_means_h, pop_vars_h, self.options, prefix=self.prefix+"_h")
            else:
                x_normal, x_mean, x_var = bn(
                    x_, x_gammas, xh_betas, pop_means_x, pop_vars_x,
                    self.options, axis=0, prefix=self.prefix + "_x")
                h_normal, h_mean, h_var = bn(
                    preact, h_gammas, 0, pop_means_h, pop_vars_h,
                    self.options, axis=0, prefix=self.prefix + "_h")
            preact = x_normal + h_normal

            i = T.nnet.sigmoid(_slice(preact, 0, self.num_hiddens))
            f = T.nnet.sigmoid(_slice(preact, 1, self.num_hiddens))
            o = T.nnet.sigmoid(_slice(preact, 2, self.num_hiddens))
            g = T.tanh(_slice(preact, 3, self.num_hiddens))

            delta_c_ = i * g
            c = f * c_ + delta_c_ + dummy_c_

            ## BN on C
            if not self.options.use_bn:
                c_normal, c_mean, c_var = bn(
                    c, 1.0, c_betas, pop_means_c, pop_vars_c, self.options, prefix=self.prefix+"_c")
            else:
                c_normal, c_mean, c_var = bn(
                    c, c_gammas, c_betas, pop_means_c, pop_vars_c,
                    self.options, axis=0, prefix=self.prefix + "_c")

            h = o * T.tanh(c_normal)
            h = m_ * h + (1. - m_)* h_
            c = m_ * c + (1. - m_) * c_
            return h, c, i, f, o, g, preact

        if popstats is None:
            popstats = OrderedDict()
            for key, size in zip(
                    "xhc", [4*self.num_hiddens, 4*self.num_hiddens, self.num_hiddens]):
                for stat, init in zip("mean var".split(), [0, 1]):
                    name = "_%s_%s" % (key, stat)
                    popstats[self.prefix+name] = T.alloc(0., n_step, size)
        popstats_seq = [popstats[self.prefix + '_x_mean'], popstats[self.prefix + '_h_mean'], popstats[self.prefix + '_c_mean'],
                        popstats[self.prefix + '_x_var'], popstats[self.prefix + '_h_var'], popstats[self.prefix + '_c_var']]

        rval, updates = theano.scan(
            step,
            sequences=[state_below, mask.dimshuffle(0, 1, 'x'), dummy_h, dummy_c] + popstats_seq,
            non_sequences=[U,
                           self.x_gammas, self.h_gammas, self.xh_betas,
                           self.c_gammas, self.c_betas],
            outputs_info=[T.repeat(self.h0[None, :], n_sample, axis=0),
                          T.repeat(self.c0[None, :], n_sample, axis=0),
                          None, None, None, None, None],
            strict=True)

        return rval + [dummy_h, dummy_c, state_below]
