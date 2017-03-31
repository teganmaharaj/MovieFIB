#!/usr/bin/env python
import matplotlib
# make a plot without needing an X-server at all
# this is for saving .png
matplotlib.use('Agg')
# this is for interactive
#matplotlib.use('TkAgg')

import cPickle
import sys, logging,time
import tarfile
import os, re, subprocess,inspect, re, struct, pdb
from os.path import basename, exists, dirname, join, isdir
from os import remove
from time import sleep
from itertools import imap, izip
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import numpy
#import scipy
#import scipy.stats


from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import TextArea, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.decomposition import PCA
#import locale
#locale.setlocale(locale.LC_NUMERIC, "")
#import ipdb
from PIL import Image
import contextlib
#from data_tools import image_tiler
#from hinton_demo import hinton
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from scipy import ndimage

import scipy
from scipy.stats import itemfreq
import scipy.sparse
from jobman import DD, expand

theano.config.floatX = 'float32'
floatX = theano.config.floatX

relu = lambda x: T.maximum(0.0, x)
    
def nan_mode():
    from pylearn2.devtools.nan_guard import NanGuardMode
    mode = NanGuardMode(nan_is_error=True, if_is_error=True, big_is_error=True)
    return mode


class Online_Mean_Std(object):
    def __init__(self):
        self.n = 0
        self.mean = 0.
        self.M2 = 0.
        
    def update(self, point):
        self.n += 1
        delta = point - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (point - self.mean)
        return self.mean, numpy.sqrt(self.M2 / self.n)
    
def online_mean_std(data):
    n = 0
    mean = 0.0
    M2 = 0.0

    for x in data:
        n = n + 1
        delta = x - mean
        mean = mean + delta/n
        M2 = M2 + delta*(x - mean)
        
    if n < 2:
        return float('nan'), float('nan')
    else:
        return mean, numpy.sqrt(M2 / (n))
    
class History(object):
    # a data structure that always keep the latest N record, for NaN debugging
    def __init__(self, size=5):
        self.size = size
        self.record = [None] * self.size
    def push(self, entry):
        if len(self.record) > self.size:
            self.record.remove(self.record[0])
        self.record.append(entry)
    
monitor = []
def check_tv_nan(var):
    rval = False
    if hasattr(var.tag,'test_value'):
        tv = var.tag.test_value
        rval = numpy.isnan(tv).any()
    return rval

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

        """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = file('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
            
def count_files_in_dir(dir):
    return len([item for item in os.listdir(dir) if os.path.isfile(os.path.join(dir, item))])
            
def get_normal(rng_numpy, shape, scale=0.01):
    val = rng_numpy.normal(size=shape) * scale
    val = val.astype('float32')
    return val

def extract_rtn(self, img, i, j):
    """
    Retinal transform, kindly provided by Yin and Hugo.
    
    i,j is the ith,jth grid, start from 0 to self.grid-1
    zero padding
    """
    def rebin(a, k):
        shape = (a.shape[0]/(2**k), a.shape[1]/(2**k))
        sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
        sh = T.cast(sh, 'int32')
        return a.reshape(sh, ndim=4).mean(-1).mean(1)

    x_c = self.starting_point[0] + j*self.fixation_step + self.pad_width
    y_c = self.starting_point[1] + i*self.fixation_step + self.pad_width

    sub_img_list = []
    k=0
    for rtn_w in self.rtn_width:
        if rtn_w ==0:
            continue
        else:
            sub_img = img[y_c-rtn_w/2:y_c+(rtn_w-rtn_w/2), x_c-rtn_w/2:x_c+(rtn_w-rtn_w/2)] 
            subsampled_img = rebin(sub_img, k).flatten()
            sub_img_list.append(subsampled_img)
        k+=1 
    rtn_feature = T.concatenate(sub_img_list, axis=0)
    return rtn_feature

def sharedX(value):
    return theano.shared(value)
    
def query2words(query):
    
    tokens = nltk.pos_tag(nltk.word_tokenize(text))
    good_words = [w for w, wtype in tokens
                  if wtype not in stop]
    import ipdb;ipdb.set_trace()
    
def parse_kv_file(i, file_name):
    data = []
    keys = []
    fid = open(file_name,'rb')
    # read first keylen
    keylen = struct.unpack('i', fid.read(4))[0]

    counter = 0
    while keylen > 0:
        #try: 
        counter +=1
        # read image name, its the contsign of the image
        key = fid.read(keylen)
        sys.stdout.write('\rProcessing %d-th image in file %d'%(counter, i))
        sys.stdout.flush()
        #print i,counter, key
        # read total buffer length of feature

        vallen = struct.unpack('i', fid.read(4))[0]
        # read number of non-zero features
        num_pair = struct.unpack('i', fid.read(4))[0]
        # read pairs
        val = struct.unpack('if'*num_pair, 
                            fid.read(8*num_pair))
        # this is index
        index = numpy.array(val[0::2], dtype='int32')
        # this is value
        value = numpy.array(val[1::2], dtype='float32')
        # read next data
        content = fid.read(4)
        if content == '':
            break
        keylen = struct.unpack('i', content)[0]
        features = numpy.zeros((1,9192),dtype='float32')
        features[0,index-1] = value
        data.append(features)
        keys.append(key)
        #except Exception,e:
        #    raise e
        #    import ipdb;ipdb.set_trace()
    fid.close()
    print
    return keys, data
    
def get_updates_grads_momentum(gparams, params, updates, lr, momentum, floatX):
    print 'building updates with momentum'
    # build momentum
    gparams_mom = []
    for param in params:
        gparam_mom = theano.shared(
            numpy.zeros(param.get_value(borrow=True).shape,
            dtype=floatX))
        gparams_mom.append(gparam_mom)

    for gparam, gparam_mom, param in zip(gparams, gparams_mom, params):
        inc = momentum * gparam_mom - (constantX(1) - momentum) * lr * gparam
        updates[gparam_mom] = inc
        updates[param] = param + inc
    return updates

def get_updates_adadelta(grads, params, 
                         updates,
                         learning_rate, adadelta_epsilon,
                         lr_decrease, floatX, decay=0.95):
    decay = constantX(decay)
    print 'build updates with adadelta'
    for param, grad in zip(params, grads):
        # mean_squared_grad := E[g^2]_{t-1}
        mean_square_grad = sharedX(numpy.zeros(param.get_value().shape, dtype=floatX))
        # mean_square_dx := E[(\Delta x)^2]_{t-1}
        mean_square_dx = sharedX(numpy.zeros(param.get_value().shape, dtype=floatX))
        if param.name is not None:
            mean_square_grad.name = 'mean_square_grad_' + param.name
            mean_square_dx.name = 'mean_square_dx_' + param.name

        # Accumulate gradient
        new_mean_squared_grad = \
                decay * mean_square_grad +\
                (1 - decay) * T.sqr(grad)
        # Compute update
        #epsilon = constantX(lr_decrease) * learning_rate
        #epsilon = constantX(0.00001) * learning_rate
        #epsilon = constantX(1e-7)
        epsilon = constantX(adadelta_epsilon)
        rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
        rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
        delta_x_t = - rms_dx_tm1 / rms_grad_t * grad

        # Accumulate updates
        
        new_mean_square_dx = \
                decay * mean_square_dx + \
                (1 - decay) * T.sqr(delta_x_t)

        # Apply update
        updates[mean_square_grad] = new_mean_squared_grad
        updates[mean_square_dx] = new_mean_square_dx
        updates[param] = param + delta_x_t
    return updates

def get_updates_rmsprop(grads, params, updates, learning_rate, floatX, decay=0.95):
    for param,grad in zip(params,grads):
        mean_square_grad = sharedX(numpy.zeros(param.get_value().shape, dtype=floatX))
        new_mean_squared_grad = (decay * mean_square_grad +
                                 (1 - decay) * T.sqr(grad))
        rms_grad_t = T.sqrt(new_mean_squared_grad)
        delta_x_t = constantX(-1) * learning_rate * grad / rms_grad_t
        updates[mean_square_grad] = new_mean_squared_grad
        updates[param] = param + delta_x_t
    return updates
    
def get_updates_Nesterov():
    pass

def get_updates_adagrad(cost, params, consider_constant, updates, lr, floatX):
    # based on
    # http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
    print 'building updates with AdaGrad'
    grads = T.grad(cost, params, consider_constant)
    fudge_factor = constantX(1e-6)
    grads_history = []
    for param in params:
        grad_h = theano.shared(numpy.zeros(param.get_value().shape, dtype=floatX))
        grads_history.append(grad_h)
    
    for grad, grad_h, param in zip(grads, grads_history, params):
        updates[grad_h] = grad_h + grad **2
        grad_adjusted = grad /(fudge_factor + T.sqrt(grad_h))
        updates[param] = param - lr * grad_adjusted
    return updates

def build_updates_with_rules(
        cost, params, consider_constant,
        updates,
        lr, epsilon,lr_decrease, momentum,
        floatX, which_type, grads=None):
    # lr: scalar theano shared var
    # mom: scalar theano constant
    # lr_decrease: scalar python constant
    if grads is None:
        # given grads, no need to grad again
        grads = T.grad(cost, params, consider_constant)
    if which_type == 0:
        # use easy SGD with momentum
        updates = get_updates_grads_momentum(
            grads, params, updates,
            lr, momentum, floatX)
        
    elif which_type == 1:
        # use adadelta
        updates = get_updates_adadelta(grads, params, 
                updates, lr, epsilon,lr_decrease, floatX)
        
    elif which_type == 2:
        # use adagrad
        updates = get_updates_adagrad(grads, params,
                updates, lr, floatX)

    elif which_type == 3:
        # use rmsprop
        updates = get_updates_rmsprop(grads, params, 
                updates, lr, floatX)

    else:
        raise NotImplementedError()
    
    return updates

# activations
def sigmoid_py_(x):
    return 0.5*numpy.tanh(0.5*x) + 0.5

def softmax_py(x):
    return numpy.exp(x) / numpy.exp(x).sum()

def sigmoid_py(x):
    return 1./(1 + numpy.exp(-x))

def tanh_py(x):
    return (1-numpy.exp(-2*x))/(1+numpy.exp(-2*x))

def apply_act_py(x, act):
    if act == 'sigmoid':
        return sigmoid_py(x)
    elif act == 'tanh':
        return tanh_py(x)
    elif act == 'linear':
        return x
    else:
        raise NotImplementedError('%s not supported!'%act)
        
def noisy_tanh_py(x):
    noise1 = numpy.random.normal(loc=0, scale=2, size=x.shape)
    noise2 = numpy.random.normal(loc=0, scale=2, size=x.shape)
    return tanh_py(x+noise1) + noise2


#------------------------------------------------------------------------------------
# data preprocessing
#------------------------------------------------------------------------------------
def resample(x,y):
    idx = []
    for i in range(6):
        idx.append(y[:,1]==i)
    x0 = x[idx[0]];y0=y[idx[0]]
    x1 = x[idx[1]];y1=y[idx[1]]
    x2 = x[idx[2]];y2=y[idx[2]]
    x3 = x[idx[3]];y3=y[idx[3]]
    x4 = x[idx[4]];y4=y[idx[4]]
    x5 = x[idx[5]];y5=y[idx[5]]

    k = x0.shape[0] / (x.shape[0]-x0.shape[0])
    x1,y1 = duplicate_dataset(x1,y1,k)
    x2,y2 = duplicate_dataset(x2,y2,k)
    x3,y3 = duplicate_dataset(x3,y3,k*3)
    x4,y4 = duplicate_dataset(x4,y4,k*5)
    x5,y5 = duplicate_dataset(x5,y5,k*2)

    new_x = [x0,x1,x2,x3,x4,x5]
    new_y = [y0,y1,y2,y3,y4,y5]
    new_x = numpy.concatenate(new_x,axis=0)
    new_y = numpy.concatenate(new_y,axis=0)
    new_x, idx = shuffle_dataset(new_x)
    new_y = new_y[idx]
    return new_x, new_y

def duplicate_dataset(x,y, k):
    assert k!=0
    # duplicate data k times
    print 'duplicating dataset'
    assert x.ndim == 2
    assert x.shape[0] == y.shape[0]
    t = []
    m = []
    for i in range(k):
        t.append(numpy.copy(x))
        m.append(numpy.copy(y))
    t = numpy.asarray(t)
    m = numpy.asarray(m)

    a,b,c = t.shape
    t = t.reshape((a*b, c))
    a,b,c = m.shape
    m = m.reshape((a*b, c))
    assert t.shape[0] == m.shape[0]
    return t, m

def rebalance_dataset(datasets):
    # datasets is a list of datasets, e.g., [p0,p1,p2,...]
    # The rebalance duplicate small set by n_dup.

    # figure out duplicate numbers
    counts = [d.shape[0] for d in datasets]
    idx = numpy.argmax(counts)
    max_count = counts[idx]
    mapping = dict()
    for i, count in enumerate(counts):
        n_dup = max_count / count
        mapping[count] = n_dup
        
    # now duplicates datasets
    rebalanced = []
    for i, d in enumerate(datasets):
        count = d.shape[0]
        n_dup = mapping[count]
        print 'dataset %d: original size %d, duplicate %d times '%(i, count, n_dup)
        d = numpy.repeat(d,n_dup,axis=0)
        rebalanced.append(d)
    return rebalanced

def shuffle_dataset(data):
    idx = shuffle_idx(data.shape[0])
    return data[idx], idx

def shuffle_idx(n, shuffle_seed=1234):
    print 'shuffling dataset'
    idx=range(n)
    numpy.random.seed(shuffle_seed)
    numpy.random.shuffle(idx)
    return idx

def shuffle_list(l):
    # l is a list of stuff
    idx = shuffle_idx(len(l))
    return numpy.asarray(l)[idx].tolist()

def shuffle_idx_matrix(n,how_many_orderings,shuffle_seed=1234):
    print 'shuffling dataset'
    idx=range(n)
    idx = []
    numpy.random.seed(shuffle_seed)
    for i in range(how_many_orderings):
        t = range(n)
        numpy.random.shuffle(t)
        idx.append(t)
    return idx

def divide_to_3_folds(size, mode=[.70, .15, .15]):
    """
    this function shuffle the dataset and return indices to 3 folds
    of train, valid, test

    minibatch_size is not None then, we move tails around to accommadate for this.
    mostly for convnet.
    """
    numpy.random.seed(1234)
    indices = range(size)
    numpy.random.shuffle(indices)
    s1 = int(numpy.floor(size * mode[0]))
    s2 = int(numpy.floor(size * (mode[0] + mode[1])))
    s3 = size
    idx_1 = indices[:s1]
    idx_2 = indices[s1:s2]
    idx_3 = indices[s2:]

    return idx_1, idx_2, idx_3

def shuffle_and_divide_to_2_fold_with_minibatches(
        size, how_many_minibatches, mode=[.70,.30],
        invalid_idx=None, keep_order=False):
    # shuffle and get rid of idx that are in invalid_idx
    # invalid_idx should be a list
    # keep_order: indexing a big matrix randomly may take a lot
    # more time than I think, better keep the order
    numpy.random.seed(1234)
    if invalid_idx is not None:
        indices = list(set(range(size))-set(invalid_idx))
    else:
        indices = range(size)
    if not keep_order:
        numpy.random.shuffle(indices)
    s1 = int(numpy.floor(len(indices)*mode[0]))
    idx_1 = indices[:s1]
    idx_2 = indices[s1:]
    minibatches_1 = numpy.array_split(
        idx_1,how_many_minibatches)
    minibatches_2 = numpy.array_split(
        idx_2,how_many_minibatches)
    return minibatches_1, minibatches_2

def divide_by_minibatch_size(size,minibatch_size):
    # return [start_idx,end_idx]
    idx = range(size)
    idxs = []
    
    if minibatch_size >= size:
        idxs = [[0,size]]
    else:
        n_minibatches, leftover = (size/minibatch_size,
                                   size%minibatch_size)
        for i in range(n_minibatches):
            start = minibatch_size * i
            end = minibatch_size * (i+1)
            idxs.append([start,end-1])
        if leftover != 0:
            idxs.append([end,size-1])
    return idxs
    
def scaler_to_one_hot(labels, dim):
    enc = OneHotEncoder(n_values=dim)
    t = labels.reshape((labels.shape[0],1))
    enc.fit(t)
    rval = enc.transform(t).todense()
    return numpy.asarray(rval)

def zero_mean_unit_variance(data):
    # zero mean unit variance
    print 'standardizing dataset'
    return preprocessing.scale(data)

def min_max_scale(data):
    # [0,1]
    print 'scale to [0,1]'
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)

def uniformization(data):
    # map data into its normalized rank
    pass

def logarithmization(data):
    rval = numpy.log(data)
    assert not numpy.isnan(rval.sum())
    return rval

def uniformization(inparray,zer = True):
    print 'uniformization of dataset'
    "Exact uniformization of the inparray (matrix) data"
    # Create ordered list of elements
    listelem = list(numpy.sort(list(set(inparray.flatten()))))
    dictP = {}
    totct = 0
    outarray = numpy.ones_like(inparray)
    #initialize element count
    for i in listelem:
        dictP.update({i:0})
    #count
    for i in range(inparray.shape[0]):
        if len(inparray.shape) == 2:
            for j in range(inparray.shape[1]):
                dictP[inparray[i,j]]+=1
                totct +=1
        else:
            dictP[inparray[i]]+=1
            totct +=1
    #cumulative
    prev = 0
    for i in listelem:
        dictP[i]+= prev
        prev = dictP[i]
    #conversion
    for i in range(inparray.shape[0]):
        if len(inparray.shape) == 2:
            for j in range(inparray.shape[1]):
                outarray[i,j] = dictP[inparray[i,j]]/float(totct)
        else:
            outarray[i] = dictP[inparray[i]]/float(totct)
    if zer:
        outarray = outarray - dictP[listelem[0]]/float(totct)
        outarray /= outarray.max()
    return outarray

#-----------------------------------------------------------------------------------
def check_monotonic_inc(x):
    # check if elements in x increases monotonically
    dx = numpy.diff(x)
    return numpy.all(dx >= 0)

        
def git_record_most_recent_commit(save_path):
    print 'saving git repo info'
    record = subprocess.check_output(["git", "show", "--summary"])
    save_file = save_path+'repo_info.txt'
    create_dir_if_not_exist(save_path)
    file = open(save_file, 'w')
    file.write(record)
    
def print_a_line():
    print '--------------------------------------------------'

def get_two_rngs(seed=None):
    if seed is None:
        seed = 1234
    else:
        seed = seed
    rng_numpy = numpy.random.RandomState(seed)
    rng_theano = MRG_RandomStreams(seed)

    return rng_numpy, rng_theano

def get_zero():
    # in case stuff gets pulled out of GPU
    return numpy.zeros((1,)).astype('float32')

def get_parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir))

def find_all_numbers_in_string(string):
    # return ['10', '100'] when used on samples_10k_e100.png'
    return re.findall(r"([0-9.]*[0-9]+)", string)

def generate_geometric_sequence(start, end, times):
    # generate a geometric sequence
    # e.g. [1, 2, 4, 6, 8, 16, ...., end]
    print 'generating a geometric sequence'
    assert start < end
    rval = [start]
    e = start
    while True:
        e = e * times
        if e <= end:
            rval.append(e)
        else:
            break

    if rval[-1] < end:
        rval.append(end)

    return rval

@contextlib.contextmanager
def printoptions(*args, **kwargs):
        original = numpy.get_printoptions()
        numpy.set_printoptions(*args, **kwargs)
        yield
        numpy.set_printoptions(**original)
        
def print_numpy(x, precision=3):
    with printoptions(precision=precision, suppress=True):
        print(x)

def show_theano_graph_size(fn):
    print ('graph size:', len(fn.maker.fgraph.toposort()))


def get_file_name_from_full_path(path):
    # only return file name without type, given path
    # e.g. 'xx/xx/xx/a.png' return a
    return basename(path).split('.')[0]
    
def sort_by_numbers_in_file_name(list_of_file_names):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('(\-?[0-9]+)', s) ]

    def sort_nicely(l):
        """ Sort the given list in the way that humans expect.
        """
        l.sort(key=alphanum_key)
        return l

    return sort_nicely(list_of_file_names)

def get_shape_convnet(image_shape, filter_shape, conv_type='standard'):
    if conv_type == 'standard':
        x = image_shape[2] - filter_shape[2] + 1
        y = image_shape[3] - filter_shape[3] + 1
        m = image_shape[0]
        c = filter_shape[0]
        outputs_shape = (m,c,x,y)
    else:
        raise NotImplementedError()
    return outputs_shape

def extract_epoch_number(list_of_file_names):
    # a list of model_params_e-1.pkl, return just the epoch number
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('(\-?[0-9]+)', s) ]

    epoch_number = [alphanum_key(l)[1] for l in list_of_file_names]
    return epoch_number

def plot_manifold_samples(sample, data, save_path):
    assert sample.shape[1] == data.shape[1]
    print 'using first %d samples to generate the plot'%data.shape[0]
    n_dim = data.shape[1]
    sample = sample[:data.shape[0]]
    fig = plt.figure()
    for i in range(n_dim-1):
        #fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(2,9,i+1)
        ax.plot(sample[:,i], sample[:, i+1], '*r')
        #ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 6))
        #ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        #ax.set_title('samples')
        ax.set_axis_off()

        ax = fig.add_subplot(2,9,9+i+1)
        ax.plot(data[:,i], data[:, i+1], '*b')
        #ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 6))
        #ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        #ax.set_title('data')
        ax.set_axis_off()
    plt.savefig(save_path)

def plot_manifold_denoising(x, tilde, recons, save_path):
    assert x.shape == tilde.shape
    n_dim = x.shape[1]
    fig = plt.figure()
    for i in range(n_dim-1):
        # plot x
        #fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(3,9,i+1)
        ax.plot(x[:,i], x[:, i+1], '*r')
        #ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 6))
        #ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        #ax.set_title('samples')
        ax.set_axis_off()

        # plot tilde
        ax = fig.add_subplot(3,9,9+i+1)
        ax.plot(tilde[:,i], tilde[:, i+1], '*b')
        #ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 6))
        #ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        #ax.set_title('data')
        ax.set_axis_off()

        # plot reconstructed x
        ax = fig.add_subplot(3,9,18+i+1)
        ax.plot(recons[:,i], recons[:, i+1], '*k')
        #ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 6))
        #ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        #ax.set_title('data')
        ax.set_axis_off()
    plt.savefig(save_path)

def plot_3Dcorkscrew_denoising(x,tdx,recon,save_path):
    n_dim = x.shape[1]
    fig = plt.figure()
    D = [x,tdx,recon]
    n = len(D)
    for i in range(n):
        # plot x
        #fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(1,n,i+1,projection='3d')
        d = D[i]
        ax.scatter(d[:,0],d[:,1],d[:,2],'.')

    plt.savefig(save_path)

def plot_3Dcorkscrew_samples(sample,original,save_path):
    n_dim = sample.shape[1]
    fig = plt.figure()
    D = [sample,original]
    n = len(D)
    for i in range(n):
        # plot x
        #fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(1,n,i+1,projection='3d')
        d = D[i]
        ax.scatter(d[:,0],d[:,1],d[:,2],'.')

    plt.savefig(save_path)

def plot_3Dcorkscrew_denoising(x,tdx,recon,save_path):
    n_dim = x.shape[1]
    fig = plt.figure()
    D = [x,tdx,recon]
    n = len(D)
    for i in range(n):
        # plot x
        #fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(1,n,i+1,projection='3d')
        d = D[i]
        ax.scatter(d[:,0],d[:,1],d[:,2],'.')

    plt.savefig(save_path)

def plot_2D_denoising(x,tdx,recon,save_path):
    assert x.shape[1] == 2
    n_dim = x.shape[1]
    fig = plt.figure()
    D = [x,tdx,recon]
    n = len(D)

    distances = [0, numpy.sqrt(((D[1] - D[0])**2).sum(axis=1)).mean(),
                 numpy.sqrt(((D[2] - D[0])**2).sum(axis=1)).mean()]
    # pick up some points to tag
    idx = shuffle_idx(x.shape[0])[:5]
    points = []
    for data in D:
        points.append(data[idx,:])

    axs = []
    title = ['trainset','corrupted','reconstruct']
    for i in range(n):
        # plot x
        #fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(1,n,i+1)
        d = D[i]
        ax.scatter(d[:,0],d[:,1])
        ax.set_title('%s,%.3f '%(title[i],distances[i]))
        axs.append(ax)

        # now mark some points
        for i, point in enumerate(points[i]):
            offsetbox = TextArea("%d"%i, minimumdescent=False)
            ab = AnnotationBbox(offsetbox, point,
                                xybox=(-20, 40),
                                xycoords='data',
                                boxcoords="offset points",
                                arrowprops=dict(arrowstyle="->",
                                                connectionstyle='arc3,rad=0.5',
                                                color='r'))
            ax.add_artist(ab)

    x_lim = axs[1].get_xlim()
    y_lim = axs[1].get_ylim()
    for ax in axs:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    plt.savefig(save_path)

def make_historgram():
    # mainly for the multimodal gsn paper
    font = {'family' : 'normal',
                    'weight' : 'normal',
                    'size'   : 15}

    matplotlib.rc('font', **font)
    
    nade = [-143, -124, -118, -114]
    gsn = [-148, -131, -125, -121]
    ind = numpy.arange(4)
    width = 0.35
    fig, ax = plt.subplots()
    nade_bar = ax.bar(ind, nade, width, color='g')
    gsn_bar = ax.bar(ind+width, gsn, width, color='b')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('10K', '50K', '100K', '150K'))
    ax.legend( (nade_bar, gsn_bar), ('multimodal GSN', 'unimodal GSN'))
    plt.gca().invert_yaxis()
    plt.xlabel('number of H samples')
    plt.ylabel('Log-probability')
    plt.ylim((-85,-160))
    plt.show()
    
def plot_2D_samples(sample,original,save_path):
    assert sample.shape[1] == 2
    n_dim = sample.shape[1]
    fig = plt.figure()
    D = [sample,original]
    n = len(D)
    axs = []
    for i in range(n):
        # plot x
        #fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(1,n,i+1)
        d = D[i]
        ax.scatter(d[:,0],d[:,1])
        if i == 0:
            ax.set_title('samples')
        if i == 1:
            ax.set_title('trainset')
        axs.append(ax)

    x_lim = axs[0].get_xlim()
    y_lim = axs[0].get_ylim()
    for ax in axs:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    plt.savefig(save_path)


def plot_3D_samples(sample, data, save_path):
    assert sample.shape[1] == 10
    assert sample.shape[1] == data.shape[1]
    print 'using first %d samples to generate the plot'%data.shape[0]
    n_dim = data.shape[1]
    sample = sample[:data.shape[0]]
    fig = plt.figure()
    for i in range(n_dim-1):
        #fig.subplots_adjust(wspace=0.4)
        ax = fig.add_subplot(2,9,i+1)
        ax.plot(sample[:,i], sample[:, i+1], '*r')
        #ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 6))
        #ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        #ax.set_title('samples')
        ax.set_axis_off()

        ax = fig.add_subplot(2,9,9+i+1)
        ax.plot(data[:,i], data[:, i+1], '*b')
        #ax.set_xlabel(r'$x_{%d}$'%i, fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 6))
        #ax.set_ylabel(r'$x_{%d}$'%(i+1), fontsize=20)
        #ax.set_title('data')
        ax.set_axis_off()
    plt.savefig(save_path)


def apply_act(x, act=None):
    # apply act(x)
    # linear:0, sigmoid:1, tanh:2, relu:3, softmax:4, ultra_fast_sigmoid:5
    if act == 'sigmoid' or act == 1:
        rval = T.nnet.sigmoid(x)
    elif act == 'tanh' or act == 2:
        rval = T.tanh(x)
    elif act == 'relu' or act == 3:
        rval = relu(x)
    elif act == 'linear' or act == 0:
        rval = x
    elif act == 'softmax' or act == 4:
        rval = T.nnet.softmax(x)
    elif act == 'ultra_fast_sigmoid' or act == 5:
        # does not seem to work with the current Theano, gradient not defined!
        rval = T.nnet.ultra_fast_sigmoid(x)
    else:
        raise NotImplementedError()
    return rval

def build_weights(n_row=None, n_col=None, style=None,
                  name=None,
                  rng_numpy=None, value=None,
                  size=None,on_cpu=False, scale=0.01):
    # build shared theano var for weights
    if size is None:
        size = (n_row, n_col)
    if rng_numpy is None:
        rng_numpy, _ = get_two_rngs()
    if value is not None:
        print '\tuse existing value to init weights'
        if len(size) == 3:
            assert value.shape == (n_row, how_many, n_col)
        else:
            assert value.shape == (n_row, n_col)
        rval = theano.shared(value=value, name=name)
    else:
        if style == -1:
            print '\tbuild_weights: init %s with Gaussian (0, %f)'%(name, scale)
            value = numpy.asarray(rng_numpy.normal(loc=0, scale=scale,
                                size=size), dtype=floatX)
        elif style == 0:
            # do this only when sigmoid act
            print '\tbuid_weights: init %s with FORMULA'%name
            value = numpy.asarray(rng_numpy.uniform(
                          low=-4 * numpy.sqrt(6. / (n_row + n_col)),
                          high=4 * numpy.sqrt(6. / (n_row + n_col)),
                          size=size), dtype=floatX)
        elif style == 99:
            print '\tbuild_weights: init %s with Gaussian (0, %f)'%(name, scale)
            value = numpy.asarray(rng_numpy.normal(loc=0, scale=scale,
                                size=size), dtype=floatX)
            u, s, v = numpy.linalg.svd(value, full_matrices=0)
            value = u
        elif style == 1:
            print '\tbuild_weights: init %s with Gaussian (0, %f)'%(name, scale)
            value = numpy.asarray(rng_numpy.normal(loc=0, scale=scale,
                                size=size), dtype=floatX)
        elif style == 2:
            print '\tbuild_weights: init with another FORMULA'
            value = numpy.asarray(rng_numpy.uniform(
                    low=-numpy.sqrt(6. / (n_row + n_col)),
                    high=numpy.sqrt(6. / (n_row + n_col)),
                    size=size), dtype=floatX)
        elif style == 3:
            print '\tbuild_weights: int weights to be all ones, only for test'
            value = numpy.ones(size, dtype=floatX)
        elif style == 4:
            print '\tbuild_weights: usual uniform initialization of weights -1/sqrt(n_in)'
            value = numpy.asarray(rng_numpy.uniform(
                low=-1/numpy.sqrt(n_row),
                high=1/numpy.sqrt(n_row), size=size), dtype=floatX)
        elif style == 5:
            print '\tbuild_weights: init %s with Gaussian (0, %f)'%(name, 0.2)
            value = numpy.asarray(rng_numpy.normal(loc=0, scale=0.2,
                                size=size), dtype=floatX)
        elif style == 6:
            print '\tbuild_weights: init %s with Gaussian (0, %f)'%(name, 0.1)
            value = numpy.asarray(rng_numpy.normal(loc=0, scale=0.1,
                                size=size), dtype=floatX)
        elif style == 7:
            print '\tbuild_weights: init %s with Gaussian (0, %f)'%(name, 0.05)
            value = numpy.asarray(rng_numpy.normal(loc=0, scale=0.05,
                                size=size), dtype=floatX)
        else:
            raise NotImplementedError()
        if on_cpu:
            print '\tforce weights allocated on cpu'
            rval = theano.tensor._shared(value=value,name=name)
        else:
            rval = theano.shared(value=value, name=name)
    return rval

def build_bias(size=None, name=None, value=None):
    # build theano shared var for bias
    if value is not None:
        assert value.shape == (size,)
        print '\tbuild_bias: use existing value to init bias'
        rval = theano.shared(value=value, name=name)
    else:
        print '\tbuild_bias: bias of size (%d,) init to 0'%size
        rval = theano.shared(value=numpy.zeros(size, dtype=floatX), name=name)
    return rval

def generate_masks_deep_orderless_nade(shape, rng_numpy):
    # to generate masks for deep orderless nade training
    """
    Returns a random binary maks with ones_per_columns[i] ones
    on the i-th column

    shape: (minibatch_size * n_dim)
    Example: random_binary_maks((3,5),[1,2,3,1,2])
    Out:
    array([[ 0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  0.],
    [ 0.,  0.,  1.,  0.,  1.]])
    """
    ones_per_column = rng_numpy.randint(shape[1], size=shape[0])
    assert(shape[0] == len(ones_per_column))
    shape_ = shape[::-1]
    indexes = numpy.asarray(range(shape_[0]))
    mask = numpy.zeros(shape_, dtype="float32")
    for i,d in enumerate(ones_per_column):
        rng_numpy.shuffle(indexes)
        mask[indexes[:d],i] = 1.0
    return mask.T

def corrupt_with_masking(x, size, corruption_level, rng_theano):
    rval = rng_theano.binomial(size=size, n=1,
                               p=1 - self.corruption_level,
                               dtype=theano.config.floatX) * x
    return rval

def corrupt_with_salt_and_pepper(x, size, corruption_level, rng_theano):
    a = rng_theano.binomial(size=size, n=1,
                            p=1-corruption_level,
                            dtype=theano.config.floatX)
    b = rng_theano.binomial(size=size, n=1,
                            p=0.5,
                            dtype=theano.config.floatX)
    c = T.eq(a,0) * b

    rval = x * a + c
    return rval

def corrupt_with_gaussian(x, size, corruption_level, rng_theano):
    noise = rng_theano.normal(size=size, avg=0.0,
                              std=corruption_level)
    rval = x + noise
    return rval

def corrupt_with_gaussian_py(x, size, corruption_level, rng_numpy):
    noise = rng_numpy.normal(size=size, loc=0.0,
                              scale=corruption_level)
    rval = x + noise
    return rval.astype('float32')

def cross_entropy_cost(outputs, targets):
    L = - T.mean(targets * T.log(outputs) +
                (1 - targets) * T.log(1 - outputs), axis=1)
    cost = T.mean(L)
    return cost

def diagonal_gaussian_LL(means_estimated, stds_estimated, targets):
    # estimated mean and std
    # This function has been fully tested in test_diagonal_gaussian_pdf()
    A = -((targets - means_estimated)**2) / (2*(stds_estimated**2))
    B = -T.log(stds_estimated * T.sqrt(2*numpy.pi))
    raw = A + B 
    if A.ndim == 2:
        # minibatch
        LL = (A + B).sum(axis=1).mean()
    else:
        # just one example
        LL = (A+B).mean()
    return raw, LL

def diagonal_gaussian_LL_py(means_estimated, stds_estimated, targets):
    # this does not work since scipy version is too low (0.9)
    import scipy
    from scipy.stats import multivariate_normal
    var = multivariate_normal(mean=means_estimated, cov=numpy.diag(stds_estimated))
    return numpy.log(var.pdf(targets))

def mse_cost(outputs, targets, mean_over_second=True):
    if mean_over_second:
        cost = T.mean(T.sqr(targets - outputs))
    else:
        cost = T.mean(T.sqr(targets - outputs).sum(axis=1))
    return cost

def compute_norm(data):
    # data is a matrix (samples, features)
    # compute the avg norm of data
    
    return (data**2).sum(axis=1).mean()

def gaussian_filter(data,sigma):
    return ndimage.gaussian_filter(data,sigma)

def mcmc_autocorrelation(samples):
    assert samples.ndim == 2
    # compute the autocorrelation of samples from MCMC, reference Heng's paper.
    N = samples.shape[0]
    taos = numpy.arange(N/2)
    vals = []
    for tao in taos:
        idx_a = numpy.arange(N/2)
        a = samples[idx_a,:]
        b = samples[idx_a+tao,:]
        assert a.shape == b.shape
        numer = (a*b).sum()
        denom_1 = numpy.sqrt((a*a).sum())
        denom_2 = numpy.sqrt((b*b).sum())
        autocorr = (numer+0.) / (denom_1*denom_2)
        vals.append(autocorr)
    return vals, taos

    
def mcmc_effective_sample_size(values_mcmc):
    # ref:http://www.stats.ox.ac.uk/~burke/Autocorrelation/MCMC%20Output.pdf
    # both arguments are scalars
    print 'computing effective sample size'
    # BUG, number always small
    N = values_mcmc.shape[0]
    taos = numpy.arange(N/2)
    down = numpy.var(values_mcmc)
    ups = []
    for tao in taos:
        idx_a = numpy.arange(N/2)
        a = values_mcmc[idx_a]
        b = values_mcmc[idx_a+tao]
        assert a.shape == b.shape
        ups.append(numpy.cov(a,b)[0,1])
    sigmas = numpy.asarray(ups)/down
    effective_sample_size = values_mcmc.shape[0] / (1+2*(sigmas.sum()))
    return sigmas, taos, effective_sample_size


def color_to_gray(D):
    img = Image.fromarray(D)
    img_gray = img.convert('L')
    plt.imshow(image, cmap = plt.get_cmap('gray'))
    plt.show()

def get_history(records, k):
    # records: list
    # k: int
    # this function returns the latest k elements of records
    # if records does not have k elements, return what is there
    def rounding(element):
        try:
            if type(element) is list:
                rval = [round(e,4) for e in element]
            else:
                rval = round(element,4)
        except TypeError, e:
            print e
            import ipdb; ipdb.set_trace()
        return rval

    if len(records) < k:
        rval = records[::-1]
    else:
        rval = records[-k:][::-1]

    rval = numpy.asarray(rval).tolist()
    rval = map(rounding, rval)
    return rval

def bow_to_sparse(bow, voc_size,i, split=True,
                   make_sparse_matrix=True):
    # turn bow '123-45-46' into a sparse representation
    # i is the row index

    # if split is False: bow [123,45,46]
    if split:
        bow_str = bow.split('-') 
        words_int = [int(word) for word in bow_str]
    else:
        words_int = bow
    counts = itemfreq(words_int)
    new_rows = [i] * (counts.shape[0])
    new_columns = counts[:,0].tolist()
    new_data = counts[:,1].tolist()
    if make_sparse_matrix:
        try:
            data_sparse = scipy.sparse.csr_matrix(
            (new_data,(new_rows,new_columns)),
            shape=(1,voc_size))
        except Exception,e:
            import ipdb;ipdb.set_trace()
        assert len(new_rows) == len(new_columns)
        assert len(new_rows) == len(new_data)
        return new_rows, new_columns, new_data, data_sparse
    else:
        return new_rows, new_columns, new_data
 
def image2array(im):
    if im.mode not in ("L", "F"):
        raise ValueError, "can only convert single-layer images"
    if im.mode == "L":
        a = Numeric.fromstring(im.tostring(), Numeric.UnsignedInt8)
    else:
        a = Numeric.fromstring(im.tostring(), Numeric.Float32)
    a.shape = im.size[1], im.size[0]
    return a

def array2image(a):
    if a.typecode() == Numeric.UnsignedInt8:
        mode = "L"
    elif a.typecode() == Numeric.Float32:
        mode = "F"
    else:
        raise ValueError, "unsupported image mode"
    return Image.fromstring(mode, (a.shape[1], a.shape[0]), a.tostring())

def resize_img(imgs, old_shape, new_shape=[28,28]):
    # imgs is a matrix (N,dim)
    a,b = imgs.shape
    imgs = imgs.reshape([a,old_shape[0],old_shape[1]])
    # imgs is e.g. (60k, 28, 28)
    new_imgs = []
    for k, img in enumerate(imgs):
        sys.stdout.write('\rResizing %d/%d examples'%(
                             k, imgs.shape[0]))
        sys.stdout.flush()
        new_img = Image.fromarray(img)
        new_img = new_img.resize(new_shape, Image.ANTIALIAS)
        #new_img.save('/tmp/yaoli/test/mnist_rescaled.png')
        new_imgs.append(numpy.asarray(list(new_img.getdata())))

    imgs_scaled = numpy.asarray(new_imgs).astype('float32')
    #image_tiler.visualize_mnist(data=mnist_scaled, how_many=100, image_shape=new_shape)
    return imgs_scaled

def get_rab_exp_path():
    #LISA: /data/lisa/exp/yaoli/
    return os.environ.get('RAB_EXP_PATH')

def get_rab_dataset_base_path():
    #LISA: /data/lisatmp2/yaoli/datasets/
    return os.environ.get('RAB_DATA_PATH')

def get_rab_model_base_path():
    #LISA: /data/lisatmp2/yaoli/models/
    return os.environ.get('RAB_MODEL_PATH')

def get_parent_dir(path):
    # from '/a/b/c' to '/a/b'
    return '/'.join(path.split('/')[:-1])

def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        print 'creating directory %s'%directory
        os.makedirs(directory)
    else:
        print "%s already exists!"%directory


def get_dummy_theano_var_with_test_value(shape, vtype='fmatrix', name='fm'):
    if vtype == 'fmatrix':
        var = T.fmatrix(name)
        var.tag.test_value = numpy.random.binomial(1, 0.5, shape).astype(floatX)

    return var

def from_bow(bows, dict_size):
    # convert bow to a bow representation suitable for training
    N = bow.shape[0]
    queries = []
    for bow in bows:
        this_query = []
        for w in bow:
            base = numpy.zeros((1,dict_size))
            base[0,w] = 1
            this_query.append(base)
        this_query = numpy.concatenate(this_query,axis=0).sum(axis=1)
    queries.append(this_query)
    return numpy.asarray(queries)

def compose_minibatches(data, minibatch_size):
    # take a (N,D), turn to a list of [(B,D), (B,D),...]
    minibatch_idx = generate_minibatch_idx(data.shape[0], minibatch_size)
    rvals = [data[idx] for idx in minibatch_idx]
    return rvals
    
def generate_minibatch_idx(dataset_size, minibatch_size):
    # generate idx for minibatches SGD
    # output [m1, m2, m3, ..., mk] where mk is a list of indices
    assert dataset_size >= minibatch_size
    n_minibatches = dataset_size / minibatch_size
    leftover = dataset_size % minibatch_size
    idx = range(dataset_size)
    if leftover == 0:
        minibatch_idx = numpy.split(numpy.asarray(idx), n_minibatches)
    else:
        print 'uneven minibath chunking, overall %d, last one %d'%(minibatch_size, leftover)
        minibatch_idx = numpy.split(numpy.asarray(idx)[:-leftover], n_minibatches)
        minibatch_idx = minibatch_idx + [numpy.asarray(idx[-leftover:])]
    minibatch_idx = [idx_.tolist() for idx_ in minibatch_idx]
    return minibatch_idx
    
def constantX(value, float_dtype='float32'):
    """
    Returns a constant of value `value` with floatX dtype
    """
    return theano.tensor.constant(numpy.asarray(value, dtype=float_dtype))

def constantX_int(value):
    """
    Returns a constant of value `value` with floatX dtype
    """
    return theano.tensor.constant(numpy.asarray(value, dtype='int32'))

def get_theano_constant(constant, dtype, bc_pattern):
    # usage: dtype = 'float32', bc_pattern=()
    # see http://deeplearning.net/software/theano/library/tensor/basic.html for details.
    try:
        rval = theano.tensor.TensorConstant(theano.tensor.TensorType(dtype,
                            broadcastable=bc_pattern), numpy.asarray(constant, 'float32'))
    except TypeError, e:
        print e
        import ipdb; ipdb.set_trace()
    return rval

def run_with_try(func):
    """
    Call `func()` with fallback in pdb on error.
    """
    try:
        return func()
    except Exception, e:
        print '%s: %s' % (e.__class__, e)
        ipdb.post_mortem(sys.exc_info()[2])
        raise

def list_of_array_to_array(l):
    base = l[0]
    sizes = [member.shape[0] for member in l]
    n_samples = sum(sizes)
    #print n_samples
    if l[0].ndim == 2:
        n_attrs = l[0].shape[1]
        X = numpy.zeros((n_samples, n_attrs))
    elif l[0].ndim == 1:
        X = numpy.zeros((n_samples,))
    else:
        NotImplementedError('ndim of the element of the list must be <=2')
    idx_start = 0
    for i, member in enumerate(l):
        #sys.stdout.write('\r%d/%d'%(i, len(l)))
        #sys.stdout.flush()
        if X.ndim == 2:
            X[idx_start:(idx_start+member.shape[0]),:] = member
        else:
            X[idx_start:(idx_start+member.shape[0])] = member
        idx_start += member.shape[0]
    return X
#--------------------------------------------------------------------------------
# the following are universial tools for check MLP training
# It assumes that train_stats.npy is found, only application when models
# are trained by K fold cv

def model_selection_one_exp(model_path):

    #model_path = '/data/lisa/exp/yaoli/test/train_stats.npy'
    stats = numpy.load(model_path)

    train_cost = numpy.empty([stats.shape[0], stats.shape[1]-1],dtype='float64')
    train_error = numpy.empty([stats.shape[0], stats.shape[1]-1],dtype='float64')
    valid_cost = numpy.empty([stats.shape[0], stats.shape[1]-1],dtype='float64')
    valid_error = numpy.empty([stats.shape[0], stats.shape[1]-1],dtype='float64')
    test_cost = numpy.empty([stats.shape[0], stats.shape[1]-1],dtype='float64')
    test_error = numpy.empty([stats.shape[0], stats.shape[1]-1],dtype='float64')

    for i in range(stats.shape[0]):
        for j in range(stats.shape[1]-1):
            value = stats[i,j]
            assert value is not None

            train_cost[i,j] = value['train_cost']
            train_error[i,j] = value['train_error']
            valid_cost[i,j] = value['valid_cost']
            valid_error[i,j] = value['valid_error']
            test_cost[i,j] = value['test_cost']
            test_error[i,j] = value['test_error']

    # epoch selection
    avg_valid_cost = valid_cost.mean(axis=1)
    best_epoch = numpy.argmin(avg_valid_cost)

    retrain_train_cost = stats[best_epoch,-1]['train_cost']
    retrain_train_error = stats[best_epoch,-1]['train_error']
    retrain_test_error = stats[best_epoch,-1]['test_error']
    retrain_test_cost = stats[best_epoch,-1]['test_cost']

    print 'best epoch %d/%d'%(best_epoch,stats.shape[0])
    print 'retrain train error ', retrain_train_error
    print 'retrain test error ', retrain_test_error

def unique_rows(data):
    uniq = numpy.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])

def print_histogram(data, bins=20):
    counts, intervals = numpy.histogram(data.flatten(),bins=20)
    print_numpy(counts)
    print_numpy(intervals)

def interval_mean(data, n_split):
    # data is numpy array
    interval = numpy.split(numpy.sort(data.flatten())[::-1], n_split)
    means = []
    for i in range(n_split):
        means.append(numpy.mean(interval[:(i+1)]))
    return means
        
#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
def show_random_sampling_graph(data=None, cost=None, name=None):
    # data: matrix, each row is one exp, each col is a hyper-param
    # cost: vector, corresponds to each row(exp) in data
    if 0:
        data = numpy.random.normal(loc=0, scale=1,size=(100,10))
        cost = numpy.random.normal(loc=0, scale=1, size=(100,))
        name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    assert data.shape[1] == len(name)
    assert data.shape[0] == cost.shape[0]
    shape = int(numpy.ceil(numpy.sqrt(data.shape[1])))
    fig = plt.figure(figsize=[20, 10])
    for i in range(data.shape[1]):
        x = data[:,i]
        y = cost
        ax = plt.subplot(shape, shape, i+1)
        ax.scatter(x, y)
        ax.set_xscale('log')
        ax.set_xlabel(name[i])

    plt.show()

def plot_nonlinear_transformation():
    samples = numpy.linspace(-5,5,100)
    output1 = sigmoid_py(samples)
    output2 = sigmoid_py(samples)
    
    plt.plot(output1+2.5+output2,'*')
    plt.show()
    
def plot_scatter_tsne(D,labels):
    fig = plt.figure()
    color = ['r','b','g','k','m','y']
    axs = []
    names = ['p0','p1','p2','p3','p4','p5']
    for i in numpy.unique(labels):
        to_scatter = D[labels == i]
        axs.append(plt.scatter(to_scatter[:,0],to_scatter[:,1],s=30, c=color[i]))
    plt.legend(axs,['0','1'])
    plt.show()

def plot_one_line(x, xlabel, ylabel,title):
    plt.plot(x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_two_lines(x, y, title_x, title_y):
    plt.plot(x, label=title_x)
    plt.plot(y, label=title_y)
    plt.legend()
    plt.show()

def plot_three_lines(x, y, z, xlabel,ylabel,zlabel, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, label=xlabel)
    ax.plot(y, label=ylabel)
    ax.plot(z, label=zlabel)
    ax.legend()
    plt.savefig(save_path)
    plt.close()
    
def plot_four_lines(x, y, z, t, legend_x, legend_y, legend_z, legend_t,
                    title,xlabel,ylabel):
    plt.plot(x, label=legend_x)
    plt.plot(y, label=legend_y)
    plt.plot(z, label=legend_z)
    plt.plot(t, label=legend_t)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_scatter(x,y,xlabel,ylabel):
    plt.scatter(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_in_3d(x, labels=None, save_path=None):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if labels is None:
        ax.scatter(x[:,0], x[:,1], x[:,2])
    else:
        color = ['r','b']
        for i in numpy.unique(labels):
            idx = labels==i
            ax.scatter(x[idx,0],x[idx,1],
                       x[idx,2],'.',color=color[i])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    return ax

def plot_acts_3D(acts):
    assert len(acts) == 3
    fig = plt.figure()
    act = acts[0]
    ax = fig.add_subplot(221,projection='3d')
    ax.scatter(act[:,0], act[:,1], act[:,2])

    act = acts[1]
    ax = fig.add_subplot(222,projection='3d')
    ax.scatter(act[:,0], act[:,1], act[:,2])

    act = acts[2]
    ax = fig.add_subplot(223,projection='3d')
    ax.scatter(act[:,0], act[:,1], act[:,2])
    plt.show()

def histogram_acts(acts, save_path=None, label=None):
    assert len(acts) == len(label)
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(acts, bins=20, label=label)
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def errorbar_acts(acts, titles, save_path=None, title=None):
    sys.setrecursionlimit(1000000)
    shape = int(numpy.ceil(numpy.sqrt(len(acts))))
    fig = plt.figure(figsize=(26,10))
    
    for k, act, title in zip(range(len(acts)), acts, titles):
        # act in one layer
        ax1 = plt.subplot2grid((len(acts), 3), (k,0))
        if act.shape[1] == 1:
            # do a historgram instead
            ax1.hist(act)
        else:
            # do error bar
            x = range(act.shape[1])
            y = numpy.mean(act, axis=0)
            error = numpy.std(act, axis=0)
            ax1.errorbar(x, y, error)
        ax1.set_ylabel(title)
        ax2 = plt.subplot2grid((len(acts), 3), (k,1))
        t = act.flatten()
        try:
            ax2.hist(t,bins=30,label='mean %.6f\nstd %.6f\nmin %.6f\nmax %.6f'%(
                numpy.mean(t),numpy.std(t),numpy.min(t),numpy.max(t)))
        except Exception,e:
            print e
            import ipdb; ipdb.set_trace()
        ax2.legend(prop={'size':9})
        
        ax3 = plt.subplot2grid((len(acts), 3), (k,2))
        plt.imshow(act,cmap=plt.cm.gray,interpolation='none',aspect='auto')
        plt.colorbar()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def diagram_acts(acts, preacts=None, label=None, save_path=None):
    # generate error bars of activation and the histogram
    # acts is a list of acts
    fig = plt.figure(figsize=(25,15))

    for k, act in enumerate(acts):
        # act in one layer
        ax = plt.subplot2grid((len(acts)+3,4), (k,0))
        x = range(act.shape[1])
        y = numpy.mean(act, axis=0)
        error = numpy.std(act, axis=0)
        ax.errorbar(x, y, error, color='b', ecolor='r')

        ax = plt.subplot2grid((len(acts)+3,4), (k,1))
        ax.hist(act, bins=10, label=label[k])
        ax.legend()

        cov = numpy.corrcoef(act)
        ax = plt.subplot2grid((len(acts)+3,4), (k,2))
        ax.imshow(cov, cmap=plt.cm.gray)


        ax = plt.subplot2grid((len(acts)+3,4), (k,3))
        x = range(act.shape[1])
        y = numpy.mean(preacts[k], axis=0)
        error = numpy.std(preacts[k],axis=0)
        ax.errorbar(x, y, error, color='b', ecolor='r')

        #ax = plt.subplot2grid((len(acts)+3,5), (k,4))
        #x = range(act.shape[1])
        #y = numpy.mean(preacts[k] - act, axis=0)
        #error = numpy.std(preacts[k] - act,axis=0)
        #ax.errorbar(x, y, error, color='b', ecolor='r')

    ax = plt.subplot2grid((len(acts)+3,4), (len(acts),0), rowspan=3, colspan=4)
    ax.hist(acts, bins=20, label=label)
    ax.legend()

    #fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def diagram_act_single_layer(act,save_path=None):
    act = act[:1000]
    # the following line complains about $DISPLAY
    fig = plt.figure(figsize=(16,6))
    ax = plt.subplot2grid((1,4), (0,0))
    #x = range(act.shape[1])
    #y = numpy.mean(act, axis=0)
    #error = numpy.std(act, axis=0)
    #ax.errorbar(x, y, error, color='b', ecolor='r')
    ax.boxplot(act)
    ax.set_xticks([-1]+range(act.shape[1]) + [act.shape[1]])

    ax = plt.subplot2grid((1,4), (0,1))
    ax.hist(act.flatten(), bins=10)
    ax.locator_params(tight=True,nbins=5)

    cov = numpy.corrcoef(act)
    ax = plt.subplot2grid((1,4), (0,2))
    ax.matshow(cov, cmap=plt.cm.gray)
    ax.locator_params(tight=True,nbins=5)
    #plt.colorbar()

    ax = plt.subplot2grid((1,4), (0,3))
    ax.matshow(act[:50], cmap=plt.cm.gray)
    #plt.colorbar()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def histogram_params(params, save_path=None):
    #data is a list of vector or matrices
    shape = int(numpy.ceil(numpy.sqrt(len(params))))
    fig = plt.figure()
    for i, param in enumerate(params):
        value = param.get_value()
        ax = fig.add_subplot(1,3,i+1)
        ax.hist(value.flatten(), bins=20)

        ax.set_title('param shape '+str(value.shape))
        ax.locator_params(tight=True,nbins=5)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def hintondiagram_params(params, save_path=None, shape=None):
    if shape is None:
        n_row = int(numpy.ceil(numpy.sqrt(len(params))))
        n_col = n_row
    else:
        n_row = shape[0]
        n_col = shape[1]

    fig = plt.figure(figsize=(16, 12))
    for i, param in enumerate(params):
        value = param.get_value()
        if value.ndim == 1:
            value = value.reshape((value.shape[0],1))
        ax = fig.add_subplot(n_row,n_col,i+1)
        hinton(value, ax=ax)
        ax.set_title('param shape '+str(value.shape))

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def heatmap_params(params, save_path=None, shape=None):
    if shape is None:
        n_row = int(numpy.ceil(numpy.sqrt(len(params))))
        n_col = n_row
    else:
        n_row = shape[0]
        n_col = shape[1]

    fig = plt.figure(figsize=(20, 15))
    for i, param in enumerate(params):
        value = param.get_value()
        if value.ndim == 1:
            value = value.reshape((value.shape[0],1))
        ax = fig.add_subplot(n_row,n_col,i+1)
        ax.imshow(value, cmap=plt.cm.gray)
        ax.set_title('param shape '+str(value.shape))

        # Move left and bottom spines outward by 10 points
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    #fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def diagram_shift(ax):
    # neat trick to move origin right-up
    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return ax

def plot_learning_curves(plt, set_1, set_2,
                         label_1='train NLL', label_2='test NLL',
                         style_1='k-', style_2='k--'):
    plt.plot(set_1, style_1, label=label_1)
    plt.plot(set_2, style_2, label=label_2)
    #plt.legend()
    #plt.show()
    return plt

def plot_learning_curves_from_npz():
    # the file should have 'train_cost', 'valid_cost'
    file_path = sys.argv[1]
    t = numpy.load(file_path)
    plt.plot(t['train_cost'], label='train cost')
    plt.plot(t['valid_cost'], label='valid cost')
    plt.plot(t['test_cost'], label='test cost')
    plt.plot(t['test_error'], label='test error')
    plt.legend()
    plt.show()

def plot_two_vector(x,y):
    plt.plot(x,'r-')
    plt.plot(y,'b-')
    plt.show()

def plot_histogram(data, bins, save_path):
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(data.flatten(), bins=bins)
    plt.savefig(save_path)
    

def plot_two_vector():
    a = [10, 20, 30, 40, 50, 60, 90, 120, 150, 180]
    b = [122, 137, 162, 156, 204, 183, 194, 154, 173, 186]
    c = [128, 164, 133, 99, 129, 124, 131, 144, 109, 132]
    plt.plot(a,b,'k-*')
    plt.plot(a,c,'r-*')
    plt.xlabel('epoch')
    plt.ylabel('log-likelihood')
    plt.legend(('9 steps of walkback','no walkback'))
    plt.show()


def plot_noisy_tanh():
    x = numpy.linspace(-10,10,100)
    x_tanh = tanh_py(x)

    x_noisy_tanhs = []
    for i in range(1000):
        x_noisy_tanh = noisy_tanh_py(x)
        x_noisy_tanhs.append(x_noisy_tanh)

    x_noisy_tanhs = numpy.asarray(x_noisy_tanhs)
    plt.plot(range(len(x)), x_tanh, 'r-')
    plt.plot(range(len(x)), x_noisy_tanhs.sum(axis=0), 'b-')
    plt.show()

def plot_cost_from_npz():
    path_1 = '/data/lisa/exp/yaoli/gsn-2-w5-n04-e100-noise2_h/stats.npz'
    path_2 = '/data/lisa/exp/yaoli/gsn-1-w1-n04-e100/stats.npz'

    t1 = numpy.load(path_1)
    t2 = numpy.load(path_2)
    to_plot_1 = t1['train_cost']
    to_plot_2 = t2['train_cost']
    x = range(len(to_plot_1))
    plt.plot(x,to_plot_1,'k-')
    plt.plot(x,to_plot_2[:len(x)],'r-')
    plt.legend(('2 walkback','1 walkback'))
    plt.show()

def show_grayscale_img(img):
    #img is a matrix with values 0-255
    plt.imshow(image, cmap = plt.get_cmap('gray'))
    plt.show()
#---------------------------------------------------------------------------------------
def dropout(acts, rng_theano, ratio):
    drop_mask = rng_theano.binomial(
        p=ratio,size=acts.shape,dtype='float32')
    return acts * drop_mask

def log_sum_exp_theano(x, axis):
    max_x = T.max(x, axis)
    return max_x + T.log(T.sum(T.exp(x - T.shape_padright(max_x, 1)), axis))

def log_sum_exp_python(x, dim=-1):
    """Compute log(sum(exp(x))) in a numerically stable way.
    
       Use second argument to specify along which dimension to sum. 
       If -1 (default), logsumexp is computed along the last dimension. 
    """
    if len(x.shape) < 2:  #got only one dimension?
        xmax = x.max()
        return xmax + numpy.log(numpy.sum(numpy.exp(x-xmax)))
    else:
        if dim != -1:
            x = x.transpose(range(dim) + range(dim+1, len(x.shape)) + [dim])
        lastdim = len(x.shape)-1
        xmax = x.max(lastdim)
        return xmax + numpy.log(numpy.sum(numpy.exp(x-xmax[...,numpy.newaxis]), lastdim))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*numpy.array(data)
    n = len(a)
    m, se = numpy.mean(a), numpy.std(a)
    h = 1.96 * se / numpy.sqrt(n)
    return m, m-h, m+h


def flatten_list_of_list(l):
    # l is a list of list
    return [item for sublist in l for item in sublist]

def load_txt_file(path):
    f = open(path,'r')
    lines = f.readlines()
    f.close()
    return lines

def write_txt_file(path, data):
    f = open(path,'w')
    lines = f.write(data.encode('utf-8'))
    f.close()

def sort_ndarray_by_column(array, d):
    assert array.ndim == 2
    idx = numpy.argsort(array[:,d])
    return array[idx,:]

def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval


def load_tar_bz2(path):
    """
    Load a file saved with `dump_tar_bz2`.
    """
    assert path.endswith('.tar.bz2')
    name = os.path.split(path)[-1].replace(".tar.bz2", ".pkl")
    f = tarfile.open(path).extractfile(name)
    try:
        data = f.read()
    finally:
        f.close()
    return cPickle.loads(data)

def dump_tar_bz2(obj, path):
    """
    Save object to a .tar.bz2 file.

    The file stored within the .tar.bz2 has the same basename as 'path', but
    ends with '.pkl' instead of '.tar.bz2'.

    :param obj: Object to be saved.

    :param path: Path to the file (must end with '.tar.bz2').
    """
    assert path.endswith('.tar.bz2')
    pkl_name = os.path.basename(path)[0:-8] + '.pkl'
    # We use StringIO to avoid having to write to disk a temporary
    # pickle file.
    obj_io = None
    f_out = tarfile.open(path, mode='w:bz2')
    try:
        obj_str = cPickle.dumps(obj)
        obj_io = StringIO.StringIO(obj_str)
        tarinfo = tarfile.TarInfo(name=pkl_name)
        tarinfo.size = len(obj_str)
        f_out.addfile(tarinfo=tarinfo, fileobj=obj_io)
    finally:
        f_out.close()
        if obj_io is not None:
            obj_io.close()

def pkl_to_hdf5(train_x, valid_x, test_x, hdf5_path):
    print 'creating %s'%hdf5_path
    import h5py
    data = [train_x, valid_x, test_x]
    name = ['train', 'valid', 'test']
    f = h5py.File(hdf5_path, 'w')
    for x, name in zip(data, name):
        group = f.create_group(name)
        dset = group.create_dataset('data',x.shape,'f')
        dset[...] = x
    f.close()

def save_sparse_csr(array,filename, separate=False):
    assert scipy.sparse.issparse(array)
    if separate:
        base_path = os.path.splitext(filename)[0]
        numpy.save(base_path+'_array_data.npy',array.data)
        numpy.save(base_path+'_array_indices.npy',array.indices)
        numpy.save(base_path+'_array_indptr.npy',array.indptr)
        numpy.save(base_path+'_array_shape.npy',array.shape)
    else:
        numpy.savez(filename,
                data = array.data,
                indices=array.indices,
                indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    assert os.path.exists(filename)
    loader = numpy.load(filename)
    return scipy.sparse.csr_matrix((  loader['data'],
                         loader['indices'],
                         loader['indptr']),
                        shape = loader['shape'])
    
def dump_pkl(obj, path):
    """
    Save a Python object into a pickle file.
    """
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()


def flatten(list_of_lists):
    """
    Flatten a list of lists.

    :param list_of_lists: A list of lists l1, l2, l3, ... to be flattened.

    :return: A list equal to l1 + l2 + l3 + ...
    """
    rval = []
    for li in list_of_lists:
        rval += li
    return rval

def center_pixels(X):
    return (X-127.5)/127.5
    
def normalize(X):
    print 'normalizing the dataset...'

    stds = numpy.std(X,axis=0)
    eliminate_columns = numpy.where(stds==0)[0]
    if eliminate_columns.size != 0:
        print "remove constant columns ", eliminate_columns
        valid_columns = stds != 0
        Y = X[:,valid_columns]
    else:
        Y = X
    stds = numpy.std(Y, axis=0)
    means = numpy.mean(Y,axis=0)

    return (Y-means)/stds

def pca(X, k=2):
    print 'performing PCA with k=%d '%k
    pca = PCA(n_components=k)
    pca.fit(X)
    #import ipdb; ipdb.set_trace()
    #print(pca.explained_variance_ratio_)
    return pca

def show_linear_correlation_coefficients(x,y):
    values = []
    if x.ndim == 1:
        print numpy.corrcoef(x,y)[0,1]
    elif x.ndim == 2:
        for idx in range(x.shape[1]):
            values.append(numpy.corrcoef(x[:,idx], y)[0,1])
    else:
        NotImplementedError
    return values

# the following is the code to print out a pretty table
def format_num(num):
    """Format a number according to given places.
    Adds commas, etc.

    Will truncate floats into ints!"""

    try:
        return '%.2e'%(num)

    except (ValueError, TypeError):
        return str(num)

def get_max_width(table, index):
    """Get the maximum width of the given column index
    """

    return max([len(format_num(row[index])) for row in table])

def pprint_table(table):
    """Prints out a table of data, padded for alignment

    @param out: Output stream ("file-like object")
    @param table: The table to print. A list of lists. Each row must have the same
    number of columns.

    """
    out = sys.stdout

    col_paddings = []

    for i in range(len(table[0])):
        col_paddings.append(get_max_width(table, i))

    for row in table:
        # left col
        print >> out, row[0].ljust(col_paddings[0] + 1),
        # rest of the cols
        for i in range(1, len(row)):
            col = format_num(row[i]).rjust(col_paddings[i] + 2)
            print >> out, col,
        print >> out

def all_binary_permutations(n_bit):
    """
    used in brute forcing the partition function
    """
    rval = numpy.zeros((2**n_bit, n_bit))

    for i, val in enumerate(xrange(2**n_bit)):
            t = bin(val)[2:].zfill(n_bit)
            t = [int(s) for s in t]
            t = numpy.asarray(t)
            rval[i] = t

    return rval

def exp_linear_nonlinear_transformation():
    pass

def set_config(conf, args, add_new_key=False):
    for key in args:
        if key != 'jobman':
            v = args[key]
            if isinstance(v, DD):
                set_config(conf[key], v)
            else:
                if conf.has_key(key):
                    conf[key] = convert_from_string(v)
                elif add_new_key:
                    # create a new key in conf
                    conf[key] = convert_from_string(v)
                else:
                    raise KeyError(key)

def convert_from_string(x):
    try:
        return eval(x, {}, {})
    except Exception:
        return x
    
def analyze_weights(path):
    # adjust k accordingly
    # usage: in the exp folder, run: RAB_tools.py model_params*.pkl
    params_path = sort_by_numbers_in_file_name(path)
    epoch_numbers = extract_epoch_number(params_path)
    # number of params
    k = 5
    boxes = [[],[],[],[],[]]
    infos = [[],[],[],[],[]]

    for epoch, path in enumerate(params_path):
        params = load_pkl(path)
        assert len(params)==len(boxes)
        for i,param in enumerate(params):
            boxes[i].append(abs(param).mean())
            infos[i] = 'shape:'+str(param.shape)
    fig = plt.figure(figsize=(15,13))
    for i, box in enumerate(boxes):
        ax = plt.subplot2grid((k,1), (i,0))
        ax.plot(epoch_numbers, box, label=infos[i])
        ax.legend(loc=7, prop={'size':15})
    plt.suptitle('change of mean(abs(param))', fontsize=20)
    plt.xlabel('training epoch', fontsize=15)
    plt.savefig('params_mag_change_all_epochs.png')
    plt.show()


def test_resample():
    x=numpy.asarray([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],
                     [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],
                     [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],
                     [1,1],[1,1],
                     [2,2],[2,2],
                     [3,3],[3,3],
                     [4,4],[4,4],
                     [5,5],[5,5]])
    y=numpy.asarray([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],
                     [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],
                     [0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],
                     [1,1],[1,1],
                     [2,2],[2,2],
                     [3,3],[3,3],
                     [4,4],[4,4],
                     [5,5],[5,5]])


    new_x, new_y = resample(x,y)
def test_plot_tsne():
    numpy.load()
    
def test_diagonal_guassian_pdf():
    means = T.fmatrix('means')
    stds = T.fmatrix('stds')
    targets = T.fmatrix('targets')
    LL,_ = diagonal_gaussian_LL(means, stds, targets)
    f = theano.function(inputs=[means, stds, targets], outputs=LL.sum(axis=1))

    m = numpy.asarray([[0,0],[0,0]]).astype('float32')
    s = numpy.asarray([[1,1],[1,1]]).astype('float32')
    t = numpy.asarray([[0,0],[0,1]]).astype('float32')

    print f(m,s,t)
    print diagonal_gaussian_LL_py(m[0],s[0],t[0])
    print diagonal_gaussian_LL_py(m[1],s[1],t[1])
    
def test_retinal_transform():
    from data_tools import data_provider
    x, _, _, _, _, _ = data_provider.load_cluttered_mnist_from_disk()
    x_pos = numpy.random.randint(low=size_original/2,high=resolution-size_original/2,size=N)
    y_pos = numpy.random.randint(low=size_original/2,high=resolution-size_original/2,size=N)
    for x,y in zip(x_pos, y_pos):
        extract_rtn()
    pass

if __name__ == "__main__":
    #table = [["", "taste", "land speed", "life"],
    #         ["spam", 1.99, 4, 1003],
    #         ["eggs", 105, 13, 42],
    #         ["lumberjacks", 13, 105, 10]]

    #pprint_table(table)
    #plot_cost_from_npz()
    #divide_to_3_folds(150)
    #all_binary_permutations(n_bit=15)
    #plot_two_vector()
    #import ipdb; ipdb.set_trace()
    #t1 = ['jobman000/samples_1.png']
    #t2 = ['jobman000/samples_2.png']
    #t3 = ['jobman000/samples_3.png']
    #t4 = ['jobman000/samples_4.png']
    #t5 = ['jobman000/samples_5.png']
    #t = t1 + t2 + t4 + t3 + t5
    #sort_by_numbers_in_file_name(t)
    #import ipdb; ipdb.set_trace()
    #plot_noisy_tanh()

    #path = sys.argv[1:]
    #analyze_weights(path)

    #show_random_sampling_graph()
    #plot_learning_curves_from_npz()
    #test_resample()
    #generate_geometric_sequence(1,500,2)
    #plot_nonlinear_transformation()
    #make_historgram()
    #generate_minibatch_idx(51,10)
    #t0=time.time()
    #mask=generate_masks_deep_orderless_nade((1000,784),numpy.random)
    #t1=time.time()
    #shuffle_and_divide_to_2_fold_with_minibatches(10,2,invalid_idx=[1,3,5])
    #import ipdb; ipdb.set_trace()
    #print 'use time', t1-t0
    #divide_by_minibatch_size(17,2)
    test_diagonal_guassian_pdf()
