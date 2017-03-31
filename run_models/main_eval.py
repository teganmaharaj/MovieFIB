import os, sys, time, copy
from collections import OrderedDict
import json
import numpy
from capgen_vid.iccv15_challenge import RAB_tools
from jobman import DD, expand
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool
from layers import init_tparams_fc, fprop_fc, ConvLSTM, ones, zeros, init_tparams_matrix, bn_ff
from capgen_vid.video_qa import common
from capgen_vid.video_qa.common import adam
import data_engine
from popstats import get_stat_estimator

import layers

rng_numpy, rng_theano = RAB_tools.get_two_rngs()
dropout_flag = theano.shared(1.)
zoneout_flag = theano.shared(1.)


def experiment(args):

    def copy_padding(array, maxlen):
            padding = []
            padding.append((0, max(0, maxlen - array.shape[0])))
            for i in xrange(1, array.ndim):
                padding.append((0, 0))
                array = numpy.pad(array, padding, mode="edge")
            return array

    from config import options
    #set_config(options, state)

    from main import MainLoop
    loop = MainLoop(options)


    print "Build Model"
    loop.model.build()
    loop.options.load_model =  args.model


    if args.estimate_population:
        print "Get population statistics"
        additional_stat_inputs = []
        additional_stat_inputs = loop.get_population_statistics(max_batch=30)

    loop.engine.send_request('full_test')
    nlls = []
    errs = []
    print "Start Evaluation"

    while True:
        [[video, video_mask, label], epoch_flag] = loop.engine.fetch_mb('full_test')
        if epoch_flag:
            break
        mlen = 30
        if args.estimate_population:
            cur_mlen = video.shape[1]
            if cur_mlen > mlen:
                mlen = cur_mlen
                for i in xrange(len(additional_stat_inputs)):
                    additional_stat_inputs[i] = copy_padding(additional_stat_inputs[i], mlen + 2)
            inputs = [video, video_mask, label] + additional_stat_inputs
            cost, nll, err, pred = loop.model.f_inference(*inputs)
        else:
            cost, nll, err, pred  = loop.model.f_train(video, video_mask, label)
        s_pred =  numpy.argmax(pred.mean(axis=0))
        errs += [s_pred != label[0]]
        print label[0], s_pred, numpy.mean(errs)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--estimate-population", type=int, default=0)
    args = parser.parse_args()

    experiment(args)
