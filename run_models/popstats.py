import sys, cPickle as pkl
import theano, itertools, pprint, copy, numpy as np, theano.tensor as T
from collections import OrderedDict
from theano.gof.op import ops_with_inner_function
from theano.scan_module.scan_op import Scan
from theano.scan_module import scan_utils
#from blocks.serialization import load

def equizip(*sequences):
    sequences = list(map(list, sequences))
    assert all(len(sequence) == len(sequences[0]) for sequence in sequences[1:])
    return zip(*sequences)

# get outer versions of the given inner variables of a scan node
def export(node, extra_inner_outputs):
    assert isinstance(node.op, Scan)

    # this is ugly but we can't use scan_utils.scan_args because that
    # clones the inner graph and then extra_inner_outputs aren't in
    # there anymore
    old_inner_inputs = node.op.inputs
    old_inner_outputs = node.op.outputs
    old_outer_inputs = node.inputs

    new_inner_inputs = list(old_inner_inputs)
    new_inner_outputs = list(old_inner_outputs)
    new_outer_inputs = list(old_outer_inputs)
    new_info = copy.deepcopy(node.op.info)

    # put the new inner outputs in the right place in the output list and
    # update info
    new_info["n_nit_sot"] += len(extra_inner_outputs)
    yuck = len(old_inner_outputs) - new_info["n_shared_outs"]
    new_inner_outputs[yuck:yuck] = extra_inner_outputs

    # in step 8, theano.scan() adds an outer input (being the actual
    # number of steps) for each nitsot. we need to do the same thing.
    # note these don't come with corresponding inner inputs.
    offset = (1 + node.op.n_seqs + node.op.n_mit_mot + node.op.n_mit_sot +
              node.op.n_sit_sot + node.op.n_shared_outs)
    # the outer input is just the actual number of steps, which is
    # always available as the first outer input.
    new_outer_inputs[offset:offset] = [new_outer_inputs[0]] * len(extra_inner_outputs)

    new_op = Scan(new_inner_inputs, new_inner_outputs, new_info)
    outer_outputs = new_op(*new_outer_inputs)

    # grab the outputs we actually care about
    extra_outer_outputs = outer_outputs[yuck:yuck + len(extra_inner_outputs)]
    return extra_outer_outputs

def gather_symbatchstats_and_estimators(outputs):
    symbatchstats = []
    estimators = []
    visited_scan_ops = set()

    for var in theano.gof.graph.ancestors(outputs):
        if hasattr(var.tag, "bn_statistic"):
            var.tag.original_id = id(var)
            symbatchstats.append(var)
            estimators.append(var)

        # descend into Scan
        try:
            op = var.owner.op
        except:
            continue
        ### FIXME: Why is one scan drop when this is activated?
        if isinstance(op, Scan): # and op not in visited_scan_ops:
            visited_scan_ops.add(op)
            print "descending into", var

            inner_estimators, inner_symbatchstats = gather_symbatchstats_and_estimators(op.outputs)
            outer_estimators = export(var.owner, inner_estimators)

            symbatchstats.extend(inner_symbatchstats)
            estimators.extend(outer_estimators)


    return symbatchstats, estimators

def get_stat_estimator(outputs):
    symbatchstats, estimators = gather_symbatchstats_and_estimators(outputs)
    return symbatchstats, estimators

