'''
should get something like:
0 / 782 minibatches, acu TOP1 64.0625, TOP5 84.3750
1 / 782 minibatches, acu TOP1 67.9688, TOP5 85.9375
2 / 782 minibatches, acu TOP1 65.6250, TOP5 88.0208
3 / 782 minibatches, acu TOP1 64.8438, TOP5 87.8906
4 / 782 minibatches, acu TOP1 62.8125, TOP5 86.5625
5 / 782 minibatches, acu TOP1 63.2812, TOP5 87.2396
6 / 782 minibatches, acu TOP1 64.0625, TOP5 88.1696
7 / 782 minibatches, acu TOP1 65.6250, TOP5 89.0625
8 / 782 minibatches, acu TOP1 65.4514, TOP5 89.2361
9 / 782 minibatches, acu TOP1 65.0000, TOP5 89.5312
10 / 782 minibatches, acu TOP1 64.4886, TOP5 88.6364
...
'''
import glob, os, time, re, socket
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear
import lasagne
import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle
from PIL import Image
from data_engine import VGGImageFuncs

hostname = socket.gethostname()
if 'gpu' in hostname or 'helios' in hostname:
    HOST = 'helios'
    googlenet_file = '/scratch/jvb-000-aa/yaoli001/models/blvc_googlenet.pkl'
else:
    HOST = 'lisa'
    googlenet_file = '/data/lisatmp4/yaoli/models/blvc_googlenet.pkl'
    
def build_inception_module(name, input_layer, nfilters):
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}
    net['pool'] = PoolLayerDNN(input_layer, pool_size=3, stride=1, pad=1)
    net['pool_proj'] = ConvLayer(
        net['pool'], nfilters[0], 1, flip_filters=False)

    net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False)

    net['3x3_reduce'] = ConvLayer(
        input_layer, nfilters[2], 1, flip_filters=False)
    net['3x3'] = ConvLayer(
        net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)

    net['5x5_reduce'] = ConvLayer(
        input_layer, nfilters[4], 1, flip_filters=False)
    net['5x5'] = ConvLayer(
        net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)

    net['output'] = ConcatLayer([
        net['1x1'],
        net['3x3'],
        net['5x5'],
        net['pool_proj']])

    return {'{}/{}'.format(name, k): v for k, v in net.items()}


def build_model(x):
    print 'build googlenet'
    net = {}
    net['input'] = InputLayer((None, 3, None, None), x)
    net['conv1/7x7_s2'] = ConvLayer(
        net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
    net['pool1/3x3_s2'] = PoolLayer(
        net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
    net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
    net['conv2/3x3_reduce'] = ConvLayer(
        net['pool1/norm1'], 64, 1, flip_filters=False)
    net['conv2/3x3'] = ConvLayer(
        net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
    net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
    net['pool2/3x3_s2'] = PoolLayer(net['conv2/norm2'], pool_size=3, stride=2)

    net.update(build_inception_module('inception_3a',
                                      net['pool2/3x3_s2'],
                                      [32, 64, 96, 128, 16, 32]))
    net.update(build_inception_module('inception_3b',
                                      net['inception_3a/output'],
                                      [64, 128, 128, 192, 32, 96]))
    net['pool3/3x3_s2'] = PoolLayer(net['inception_3b/output'],
                                    pool_size=3, stride=2)

    net.update(build_inception_module('inception_4a',
                                      net['pool3/3x3_s2'],
                                      [64, 192, 96, 208, 16, 48]))
    net.update(build_inception_module('inception_4b',
                                      net['inception_4a/output'],
                                      [64, 160, 112, 224, 24, 64]))
    net.update(build_inception_module('inception_4c',
                                      net['inception_4b/output'],
                                      [64, 128, 128, 256, 24, 64]))
    net.update(build_inception_module('inception_4d',
                                      net['inception_4c/output'],
                                      [64, 112, 144, 288, 32, 64]))
    net.update(build_inception_module('inception_4e',
                                      net['inception_4d/output'],
                                      [128, 256, 160, 320, 32, 128]))
    net['pool4/3x3_s2'] = PoolLayer(net['inception_4e/output'],
                                    pool_size=3, stride=2)

    net.update(build_inception_module('inception_5a',
                                      net['pool4/3x3_s2'],
                                      [128, 256, 160, 320, 32, 128]))
    net.update(build_inception_module('inception_5b',
                                      net['inception_5a/output'],
                                      [128, 384, 192, 384, 48, 128]))

    net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
    net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'],
                                         num_units=1000,
                                         nonlinearity=linear)
    net['prob'] = NonlinearityLayer(net['loss3/classifier'],
                                    nonlinearity=softmax)

    '''
    download from 
    https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl
    '''
    file = open(googlenet_file, 'r')
    vals = pickle.load(file)

    values = pickle.load(
        open(googlenet_file))['param values']
    lasagne.layers.set_all_param_values(
        net['prob'], [v.astype(np.float32) for v in values])
    return net, vals['synset words']


def apply_google_net(x):
    MEAN_VALUES = np.array([104, 117, 123]).reshape((3, 1, 1))
    # Convert RGB to BGR
    xx = x[:, ::-1, :, :] * 255.0
    xx = xx - MEAN_VALUES[np.newaxis, :, :, :].astype('float32')
    net, class_names = build_model(xx)
    output = lasagne.layers.get_output(net['prob'], deterministic=True)
    params = lasagne.layers.get_all_params(net['prob'], trainable=True)
    return output, params, class_names
