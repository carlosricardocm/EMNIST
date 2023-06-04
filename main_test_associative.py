# Copyright [2021] Rafael Morales Gamboa, Noé S. Hernández Sánchez,
# Carlos Ricardo Cruz Mendoza, Victor D. Cruz González, and
# Luis Alberto Pineda Cortés.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import gc
import argparse
import gettext

import numpy as np
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import json
from itertools import islice

import constants
import convnet
from associative import AssociativeMemory

import SMAC_run_sameconfigforall as smac
import process_iam as iam

# Translation
gettext.install('ame', localedir=None, codeset=None, names=None)

def print_error(*s):
    print('Error:', *s, file = sys.stderr)

def plot_behs_graph(no_response, no_correct, no_chosen, correct, training_stage):

    for i in range(len(no_response)):
        total = (no_response[i] + no_correct[i] + no_chosen[i] + correct[i])/100.0
        no_response[i] /= total
        no_correct[i] /= total
        no_chosen[i] /= total
        correct[i] /= total

    plt.clf()

    full_length = 100.0
    step = 0.1
    main_step = full_length/len(constants.memory_sizes)
    x = np.arange(0.0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = full_length
    width = 5       # the width of the bars: can also be len(x) sequence

    plt.bar(x, correct, width, label=_('Correct response chosen'))
    cumm = np.array(correct)
    plt.bar(x, no_chosen,  width, bottom=cumm, label=_('Correct response not chosen'))
    cumm += np.array(no_chosen)
    plt.bar(x, no_correct, width, bottom=cumm, label=_('No correct response'))
    cumm += np.array(no_correct)
    plt.bar(x, no_response, width, bottom=cumm, label=_('No responses'))

    plt.xlim(-width, xmax + width)
    plt.ylim(0.0, ymax)
    plt.xticks(x, constants.memory_sizes)

    plt.xlabel(_('Range Quantization Levels'))
    plt.ylabel(_('Labels'))

    plt.legend(loc=0)
    plt.grid(axis='y')

    graph_filename = constants.picture_filename('graph_behaviours_MEAN' + _('-english'), training_stage)
    plt.savefig(graph_filename, dpi=600)


def plot_pre_graph (pre_mean, rec_mean, acc_mean, ent_mean, \
    pre_std, rec_std, acc_std, ent_std, training_stage, tag = '', \
        xlabels = constants.memory_sizes,  xtitle = None, \
        ytitle = None):

    plt.clf()
    plt.figure(figsize=(6.4,4.8))

    full_length = 100.0
    step = 0.1
    main_step = full_length/len(xlabels)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step

    # Gives space to fully show markers in the top.
    ymax = full_length + 2

    # Replace undefined precision with 1.0.
    pre_mean = np.nan_to_num(pre_mean, copy=False, nan=100.0)

    plt.errorbar(x, pre_mean, fmt='r-o', yerr=pre_std, label=_('Precision'))
    plt.errorbar(x, rec_mean, fmt='b--s', yerr=rec_std, label=_('Recall'))
    if not ((acc_mean is None) or (acc_std is None)):
        plt.errorbar(x, acc_mean, fmt='y:d', yerr=acc_std, label=_('Accuracy'))

    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, xlabels)

    if xtitle is None:
        xtitle = _('Range Quantization Levels')
    if ytitle is None: 
        ytitle = _('Percentage')

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(loc=4)
    plt.grid(True)

    entropy_labels = [str(e) for e in np.around(ent_mean, decimals=1)]

    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['cyan','purple'])
    Z = [[0,0],[0,0]]
    levels = np.arange(0.0, xmax, step)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    cbar = plt.colorbar(CS3, orientation='horizontal')
    cbar.set_ticks(x)
    cbar.ax.set_xticklabels(entropy_labels)
    cbar.set_label(_('Entropy'))

    s = tag + 'graph_prse_MEAN' + _('-english')
    graph_filename = constants.picture_filename(s, training_stage)
    plt.savefig(graph_filename, dpi=600)


def plot_size_graph (response_size, size_stdev, training_stage):
    plt.clf()

    full_length = 100.0
    step = 0.1
    main_step = full_length/len(response_size)
    x = np.arange(0, full_length, main_step)

    # One main step less because levels go on sticks, not
    # on intervals.
    xmax = full_length - main_step + step
    ymax = constants.n_labels

    plt.errorbar(x, response_size, fmt='g-D', yerr=size_stdev, label=_('Average number of responses'))
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xticks(x, constants.memory_sizes)
    plt.yticks(np.arange(0,ymax+1, 1), range(constants.n_labels+1))

    plt.xlabel(_('Range Quantization Levels'))
    plt.ylabel(_('Size'))
    plt.legend(loc=1)
    plt.grid(True)

    graph_filename = constants.picture_filename('graph_size_MEAN' + _('-english'), training_stage)
    plt.savefig(graph_filename, dpi=600)


# def plot_pre_graph (pre_mean, rec_mean, ent_mean, pre_std, rec_std, ent_std, \
#     tag = '', xlabels = constants.memory_sizes, xtitle = None, \
#         ytitle = None, action=None, occlusion = None, bars_type = None, tolerance = 0):

#     plt.clf()
#     plt.figure(figsize=(6.4,4.8))

#     full_length = 100.0
#     step = 0.1
#     main_step = full_length/len(xlabels)
#     x = np.arange(0, full_length, main_step)

#     # One main step less because levels go on sticks, not
#     # on intervals.
#     xmax = full_length - main_step + step

#     # Gives space to fully show markers in the top.
#     ymax = full_length + 2

#     plt.errorbar(x, pre_mean, fmt='r-o', yerr=pre_std, label=_('Precision'))
#     plt.errorbar(x, rec_mean, fmt='b--s', yerr=rec_std, label=_('Recall'))

#     plt.xlim(0, xmax)
#     plt.ylim(0, ymax)
#     plt.xticks(x, xlabels)

#     if xtitle is None:
#         xtitle = _('Range Quantization Levels')
#     if ytitle is None: 
#         ytitle = _('Percentage')

#     plt.xlabel(xtitle)
#     plt.ylabel(ytitle)
#     plt.legend(loc=4)
#     plt.grid(True)

#     entropy_labels = [str(e) for e in np.around(ent_mean, decimals=1)]

#     cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['cyan','purple'])
#     Z = [[0,0],[0,0]]
#     levels = np.arange(0.0, xmax, step)
#     CS3 = plt.contourf(Z, levels, cmap=cmap)

#     cbar = plt.colorbar(CS3, orientation='horizontal')
#     cbar.set_ticks(x)
#     cbar.ax.set_xticklabels(entropy_labels)
#     cbar.set_label(_('Entropy'))

#     s = tag + 'graph_prse_MEAN' + _('-english')
#     graph_filename = constants.picture_filename(s, action, occlusion, bars_type, tolerance)
#     plt.savefig(graph_filename, dpi=600)


# def plot_size_graph (response_size, size_stdev, action=None, tolerance=0):
#     plt.clf()

#     full_length = 100.0
#     step = 0.1
#     main_step = full_length/len(response_size)
#     x = np.arange(0, full_length, main_step)

#     # One main step less because levels go on sticks, not
#     # on intervals.
#     xmax = full_length - main_step + step
#     ymax = constants.n_labels

#     plt.errorbar(x, response_size, fmt='g-D', yerr=size_stdev, label=_('Average number of responses'))
#     plt.xlim(0, xmax)
#     plt.ylim(0, ymax)
#     plt.xticks(x, constants.memory_sizes)
#     plt.yticks(np.arange(0,ymax+1, 1), range(constants.n_labels+1))

#     plt.xlabel(_('Range Quantization Levels'))
#     plt.ylabel(_('Size'))
#     plt.legend(loc=1)
#     plt.grid(True)

#     graph_filename = constants.picture_filename('graph_size_MEAN' + _('-english'), action, tolerance=tolerance)
#     plt.savefig(graph_filename, dpi=600)


# def plot_behs_graph(no_response, no_correct, no_chosen, correct, action=None, tolerance=0):

#     for i in range(len(no_response)):
#         total = (no_response[i] + no_correct[i] + no_chosen[i] + correct[i])/100.0
#         no_response[i] /= total
#         no_correct[i] /= total
#         no_chosen[i] /= total
#         correct[i] /= total

#     plt.clf()

#     full_length = 100.0
#     step = 0.1
#     main_step = full_length/len(constants.memory_sizes)
#     x = np.arange(0.0, full_length, main_step)

#     # One main step less because levels go on sticks, not
#     # on intervals.
#     xmax = full_length - main_step + step
#     ymax = full_length
#     width = 5       # the width of the bars: can also be len(x) sequence

#     plt.bar(x, correct, width, label=_('Correct response chosen'))
#     cumm = np.array(correct)
#     plt.bar(x, no_chosen,  width, bottom=cumm, label=_('Correct response not chosen'))
#     cumm += np.array(no_chosen)
#     plt.bar(x, no_correct, width, bottom=cumm, label=_('No correct response'))
#     cumm += np.array(no_correct)
#     plt.bar(x, no_response, width, bottom=cumm, label=_('No responses'))

#     plt.xlim(-width, full_length + width)
#     plt.ylim(0.0, full_length)
#     plt.xticks(x, constants.memory_sizes)

#     plt.xlabel(_('Range Quantization Levels'))
#     plt.ylabel(_('Labels'))

#     plt.legend(loc=0)
#     plt.grid(axis='y')

#     graph_filename = constants.picture_filename('graph_behaviours_MEAN' + _('-english'), action, tolerance=tolerance)
#     plt.savefig(graph_filename, dpi=600)


def get_formats(n):
    colors = ['r','b','g','y','m','c','k']
    lines = ['-','--','-.',':']
    markers = ['p','*','s','x','d','o']

    formats = []
    for _ in range(n):
        color = random.choice(colors)
        line = random.choice(lines)
        marker = random.choice(markers)
        formats.append(color+line+marker)
    return formats


def plot_features_graph(domain, means, stdevs, experiment, occlusion = None, bars_type = None):
    """ Draws the characterist shape of features per label.

    The graph is a dots and lines graph with error bars denoting standard deviations.
    """
    ymin = np.PINF
    ymax = np.NINF
    for i in constants.all_labels:
        yn = (means[i] - stdevs[i]).min()
        yx = (means[i] + stdevs[i]).max()
        ymin = ymin if ymin < yn else yn
        ymax = ymax if ymax > yx else yx

    main_step = 100.0 / domain
    xrange = np.arange(0, 100, main_step)
    fmts = get_formats(constants.n_labels)

    for i in constants.all_labels:
        plt.clf()
        plt.figure(figsize=(12,5))

        plt.errorbar(xrange, means[i], fmt=fmts[i], yerr=stdevs[i], label=str(i))
        plt.xlim(0, 100)
        plt.ylim(ymin, ymax)
        plt.xticks(xrange, labels='')

        plt.xlabel(_('Features'))
        plt.ylabel(_('Values'))
        plt.legend(loc='right')
        plt.grid(True)

        filename = constants.features_name(experiment, occlusion, bars_type) + '-' + str(i) + _('-english')
        plt.savefig(constants.picture_filename(filename), dpi=500)


# def get_label(memories, entropies = None):

#     # Random selection
#     if entropies is None:
#         i = random.atddrange(len(memories))
#         return memories[i]
#     else:
#         i = memories[0] 
#         entropy = entropies[i]

#         for j in memories[1:]:
#             if entropy > entropies[j]:
#                 i = j
#                 entropy = entropies[i]
    
#     return i

def get_label(memories, weights = None, entropies = None):
    if len(memories) == 1:
        return memories[0]
    random.shuffle(memories)
    if (entropies is None) or (weights is None):
        return memories[0]
    else:
        i = memories[0] 
        entropy = entropies[i]
        weight = weights[i]
        penalty = entropy/weight if weight > 0 else float('inf')
        for j in memories[1:]:
            entropy = entropies[j]
            weight = weights[j]
            new_penalty = entropy/weight if weight > 0 else float('inf')
            if new_penalty < penalty:
                i = j
                penalty = new_penalty
        return i

def msize_features(features, msize, min_value, max_value):
    return np.round((msize-1)*(features-min_value) / (max_value-min_value)).astype(np.int16)
    

# def get_ams_results(midx, msize, domain, lpm, trf, tef, trl, tel, tolerance=0):

#     # Round the values
#     max_value = trf.max()
#     other_value = tef.max()
#     max_value = max_value if max_value > other_value else other_value

#     min_value = trf.min()
#     other_value = tef.min()
#     min_value = min_value if min_value < other_value else other_value

#     trf_rounded = msize_features(trf, msize, min_value, max_value)
#     tef_rounded = msize_features(tef, msize, min_value, max_value)

#     n_labels = constants.n_labels
#     nmems = int(n_labels/lpm)

#     measures = np.zeros((constants.n_measures, nmems), dtype=np.float64)
#     entropy = np.zeros(nmems, dtype=np.float64)
#     behaviour = np.zeros(constants.n_behaviours, dtype=np.float64)

#     # Confusion matrix for calculating precision and recall per memory.
#     cms = np.zeros((nmems, 2, 2))
#     TP = (0,0)
#     FP = (0,1)
#     FN = (1,0)
#     TN = (1,1)

#     # Create the required associative memories.
#     ams = dict.fromkeys(range(nmems))
#     for m in ams:
#         ams[m] = AssociativeMemory(domain, msize, tolerance)

#     # Registration
#     for features, label in zip(trf_rounded, trl):
#         m = int(label/lpm)
#         ams[m].register(features)

#     # Calculate entropies
#     for m in ams:
#         entropy[m] = ams[m].entropy

#     # Recognition
#     response_size = 0

#     for features, label in zip(tef_rounded, tel):
#         correct = int(label/lpm)

#         memories = []
#         for k in ams:
#             recognized = ams[k].recognize(features)
#             if recognized:
#                 memories.append(k)

#             # For calculation of per memory precision and recall
#             if (k == correct) and recognized:
#                 cms[k][TP] += 1
#             elif k == correct:
#                 cms[k][FN] += 1
#             elif recognized:
#                 cms[k][FP] += 1
#             else:
#                 cms[k][TN] += 1
 
#         response_size += len(memories)
#         if len(memories) == 0:
#             # Register empty case
#             behaviour[constants.no_response_idx] += 1
#         elif not (correct in memories):
#             behaviour[constants.no_correct_response_idx] += 1
#         else:
#             l = get_label(memories, entropy)
#             if l != correct:
#                 behaviour[constants.no_correct_chosen_idx] += 1
#             else:
#                 behaviour[constants.correct_response_idx] += 1

#     behaviour[constants.mean_responses_idx] = response_size /float(len(tef_rounded))
#     all_responses = len(tef_rounded) - behaviour[constants.no_response_idx]
#     all_precision = (behaviour[constants.correct_response_idx])/float(all_responses)
#     all_recall = (behaviour[constants.correct_response_idx])/float(len(tef_rounded))

#     behaviour[constants.precision_idx] = all_precision
#     behaviour[constants.recall_idx] = all_recall

#     for m in range(nmems):
#         total_positives = cms[m][TP] + cms[m][FP]
#         if total_positives == 0:
#             print(f'Memory {m} in run {midx}, memory size {msize}, did not respond.')
#             measures[constants.precision_idx,m] = 1
#         else:
#             measures[constants.precision_idx,m] = cms[m][TP] / total_positives
#         measures[constants.recall_idx,m] = cms[m][TP] /(cms[m][TP] + cms[m][FN])
   
#     return (midx, measures, entropy, behaviour)


def get_ams_results(midx, msize, domain, lpm, trf, tef, trl, tel, fold, tolerance=0):
#def get_ams_results(midx, msize, domain, trf, tef, trl, tel, fold, tolerance=0):
    # Round the values
    max_value = trf.max()
    other_value = tef.max()
    max_value = max_value if max_value > other_value else other_value

    min_value = trf.min()
    other_value = tef.min()
    min_value = min_value if min_value < other_value else other_value

    trf_rounded = msize_features(trf, msize, min_value, max_value)
    tef_rounded = msize_features(tef, msize, min_value, max_value)

    n_labels = constants.n_labels
    n_mems = n_labels

    measures = np.zeros(constants.n_measures, dtype=np.float64)
    entropy = np.zeros(n_mems, dtype=np.float64)
    behaviour = np.zeros(
        (constants.n_labels, constants.n_behaviours), dtype=np.float64)
    

    # Confusion matrix for calculating precision and recall per memory.
    cms = np.zeros((n_mems, 2, 2), dtype='int')

    # Create the required associative memories.
    ams = dict.fromkeys(range(n_mems))
    for m in ams:
        ams[m] = AssociativeMemory(domain, msize, tolerance)
    # Registration in parallel, per label.
    Parallel(n_jobs=constants.n_jobs, require='sharedmem', verbose=50)(
        delayed(register_in_memory)(ams[label], features_list) \
            for label, features_list in split_by_label(zip(trf_rounded, trl)))
    print(f'Filling of memories done for fold {fold}')

    # Calculate entropies
    means = []
    for m in ams:
        entropy[m] = ams[m].entropy
        means.append(ams[m].mean)

    # Recognition
    response_size = np.zeros(n_mems, dtype=int)
    split_size = 500
    for rsize, scms, sbehavs in \
         Parallel(n_jobs=constants.n_jobs, verbose=50)(
            delayed(recognize_by_memory)(fl_pairs, ams, entropy) \
            for fl_pairs in split_every(split_size, zip(tef_rounded, tel))):
        response_size = response_size + rsize
        cms  = cms + scms
        behaviour = behaviour + sbehavs
    counters = [np.count_nonzero(tel == i) for i in range(n_labels)]
    counters = np.array(counters)
    behaviour[:,constants.response_size_idx] = response_size/counters
    all_responses = len(tef_rounded) - np.sum(behaviour[:,constants.no_response_idx], axis=0)
    all_precision = np.sum(behaviour[:, constants.correct_response_idx], axis=0)/float(all_responses)
    all_recall = np.sum(behaviour[:, constants.correct_response_idx], axis=0)/float(len(tef_rounded))

    behaviour[:,constants.precision_idx] = all_precision
    behaviour[:,constants.recall_idx] = all_recall

    positives = conf_sum(cms, TP) + conf_sum(cms, FP)
    details = True
    if positives == 0:
        print('No memory responded')
        measures[constants.precision_idx] = 1.0
        details = False
    else:
        measures[constants.precision_idx] = memories_precision(cms)
    measures[constants.recall_idx] = memories_recall(cms)
    measures[constants.accuracy_idx] = memories_accuracy(cms)
    measures[constants.entropy_idx] = np.mean(entropy)
 
    if details:
        for i in range(n_mems):
            positives = cms[i][TP] + cms[i][FP]
            if positives == 0:
                print(f'Memory {i} of size {msize} in fold {fold} did not respond.')
    return (midx, measures, behaviour, cms)

def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))

def conf_sum(cms, t):
    return np.sum([cms[i][t] for i in range(len(cms))])

def register_in_memory(memory, features_iterator):
    for features in features_iterator:
        memory.register(features)

TP = (0,0)
FP = (0,1)
FN = (1,0)
TN = (1,1)

def memories_precision(cms):
    total = conf_sum(cms, TP) + conf_sum(cms, FN)
    if total == 0:
        return 0.0
    precision = 0.0
    for m in range(len(cms)):
        denominator = (cms[m][TP] + cms[m][FP])
        if denominator == 0:
            m_precision = 1.0
        else:
            m_precision = cms[m][TP] / denominator
        weight = (cms[m][TP] + cms[m][FN]) / total
        precision += weight*m_precision
    return precision

def memories_recall(cms):
    total = conf_sum(cms, TP) + conf_sum(cms, FN)
    if total == 0:
        return 0.0
    recall = 0.0
    for m in range(len(cms)):
        m_recall = cms[m][TP] / (cms[m][TP] + cms[m][FN])
        weight = (cms[m][TP] + cms[m][FN]) / total
        recall += weight*m_recall
    return recall
 
def memories_accuracy(cms):
    total = conf_sum(cms, TP) + conf_sum(cms, FN)
    if total == 0:
        return 0.0
    accuracy = 0.0
    for m in range(len(cms)):
        m_accuracy = (cms[m][TP] + cms[m][TN]) / total
        weight = (cms[m][TP] + cms[m][FN]) / total
        accuracy += weight*m_accuracy
    return accuracy

def recognize_by_memory(fl_pairs, ams, entropy):
    n_mems = constants.n_labels
    response_size = np.zeros(n_mems, dtype=int)
    cms = np.zeros((n_mems, 2, 2), dtype='int')
    behaviour = np.zeros(
        (n_mems, constants.n_behaviours), dtype=np.float64)
    for features, label in fl_pairs:
        correct = label
        memories = []
        weights = {}
        for k in ams:
            recognized, weight = ams[k].recognize(features)
            if recognized:
                memories.append(k)
                weights[k] = weight
                response_size[correct] += 1
            # For calculation of per memory precision and recall
            cms[k][TP] += (k == correct) and recognized
            cms[k][FP] += (k != correct) and recognized
            cms[k][TN] += not ((k == correct) or recognized)
            cms[k][FN] += (k == correct) and not recognized
        if len(memories) == 0:
            # Register empty case
            behaviour[correct, constants.no_response_idx] += 1
        elif not (correct in memories):
            behaviour[correct, constants.no_correct_response_idx] += 1
        else:
            l = get_label(memories, weights, entropy)
            if l != correct:
                behaviour[correct, constants.no_correct_chosen_idx] += 1
            else:
                behaviour[correct, constants.correct_response_idx] += 1
    return response_size, cms, behaviour

def split_by_label(fl_pairs):
    label_dict = {}
    for label in range(constants.n_labels):
        label_dict[label] = []
    for features, label in fl_pairs:
        label_dict[label].append(features)
    return label_dict.items()

def run_optimization(domain, experiment, training_stage):
    constants.training_stage = training_stage
    best_configuration = smac.optimize()

def increase_data(domain,experiment):  
    iam.increase_data()
    return 

def test_memories(domain, experiment, training_stage):

    average_entropy = []
    stdev_entropy = []


    average_precision = []
    stdev_precision = [] 
    average_recall = []
    stdev_recall = []

    all_precision = []
    all_recall = []

    entropy = []
    accuracy = []
    precision = []
    recall = []
    all_cms = []
    response_size = []

    no_response = []
    no_correct_response = []
    no_correct_chosen = []
    correct_chosen = []
    total_responses = []

    labels_x_memory = constants.labels_per_memory[experiment]
    n_memories = int(constants.n_labels/labels_x_memory)

    for i in range(constants.training_stages):
        gc.collect()

        suffix = constants.filling_suffix
        training_features_filename = constants.features_name + suffix#constants.features_name(experiment) + suffix        
        training_features_filename = constants.data_filename(training_features_filename, training_stage, i)
        training_labels_filename = constants.labels_name + suffix        
        training_labels_filename = constants.data_filename(training_labels_filename, training_stage, i)

        suffix = constants.testing_suffix
        testing_features_filename = constants.features_name + suffix
        testing_features_filename = constants.data_filename(testing_features_filename, training_stage, i)
        testing_labels_filename = constants.labels_name + suffix        
        testing_labels_filename = constants.data_filename(testing_labels_filename, training_stage, i)

        training_features = np.load(training_features_filename)
        training_labels = np.load(training_labels_filename)
        testing_features = np.load(testing_features_filename)
        testing_labels = np.load(testing_labels_filename)

       
       
        # An entropy value per memory size and memory.
        measures_per_size = np.zeros(
            (len(constants.memory_sizes), constants.n_measures),
            dtype=np.float64)
        behaviours = np.zeros(
            (constants.n_labels,
            len(constants.memory_sizes),
            constants.n_behaviours))

        print('Train the different co-domain memories -- NxM: ',experiment,' run: ',i)
        # Processes running in parallel.
        list_measures = []
        list_cms = []
        for midx, msize in enumerate(constants.memory_sizes):
           results = get_ams_results(midx, msize, domain, labels_x_memory, \
                 training_features, testing_features, training_labels, testing_labels, i)
           list_measures.append( results )
        # list_measures_entropies = Parallel(n_jobs=constants.n_jobs, verbose=50)(
        #     delayed(get_ams_results)(midx, msize, domain, labels_x_memory, \
        #         training_features, testing_features, training_labels, testing_labels) \
        #             for midx, msize in enumerate(constants.memory_sizes))

        # for j, measures, entropy, behaviour in list_measures_entropies:
        #     measures_per_size[j, :, :] = measures.T
        #     entropies[j, :] = entropy
        #     behaviours[j, :] = behaviour

        for midx, measures, behaviour, cms in list_measures:
            measures_per_size[midx, :] = measures
            behaviours[:, midx, :] = behaviour
            list_cms.append(cms)

        # Average entropy among al digits.
        entropy.append(measures_per_size[:,constants.entropy_idx])

        # Average precision and recall as percentage
        precision.append(measures_per_size[:,constants.precision_idx]*100)
        recall.append(measures_per_size[:,constants.recall_idx]*100)
        accuracy.append(measures_per_size[:,constants.accuracy_idx]*100)

        all_precision.append(np.mean(behaviours[:, :, constants.precision_idx], axis=0) * 100)
        all_recall.append(np.mean(behaviours[:, :, constants.recall_idx], axis=0) * 100)
        all_cms.append(np.array(list_cms))
        no_response.append(np.sum(behaviours[:, :, constants.no_response_idx], axis=0))
        no_correct_response.append(np.sum(behaviours[:, :, constants.no_correct_response_idx], axis=0))
        no_correct_chosen.append(np.sum(behaviours[:, :, constants.no_correct_chosen_idx], axis=0))
        correct_chosen.append(np.sum(behaviours[:, :, constants.correct_response_idx], axis=0))
        response_size.append(np.mean(behaviours[:, :, constants.response_size_idx], axis=0))


        ##########################################################################################

    # Every row is training fold, and every column is a memory size.
    entropy = np.array(entropy)
    precision = np.array(precision)
    recall = np.array(recall)
    accuracy = np.array(accuracy)

    all_precision = np.array(all_precision)
    all_recall = np.array(all_recall)
    all_cms = np.array(all_cms)

    average_entropy = np.mean(entropy, axis=0)
    stdev_entropy = np.std(entropy, axis=0)
    average_precision = np.mean(precision, axis=0)
    stdev_precision = np.std(precision, axis=0)
    average_recall = np.mean(recall, axis=0)
    stdev_recall = np.std(recall, axis=0)
    average_accuracy = np.mean(accuracy, axis=0)
    stdev_accuracy = np.std(accuracy, axis=0)

    no_response = np.array(no_response)
    no_correct_response = np.array(no_correct_response)
    no_correct_chosen = np.array(no_correct_chosen)
    correct_chosen = np.array(correct_chosen)
    response_size = np.array(response_size)

    all_precision_average = np.mean(all_precision, axis=0)
    all_precision_stdev = np.std(all_precision, axis=0)
    all_recall_average = np.mean(all_recall, axis=0)
    all_recall_stdev = np.std(all_recall, axis=0)
    main_no_response = np.mean(no_response, axis=0)
    main_no_correct_response = np.mean(no_correct_response, axis=0)
    main_no_correct_chosen = np.mean(no_correct_chosen, axis=0)
    main_correct_chosen = np.mean(correct_chosen, axis=0)
    main_response_size = np.mean(response_size, axis=0)
    main_response_size_stdev = np.std(response_size, axis=0)

    best_memory_size = constants.memory_sizes[
        np.argmax(main_correct_chosen)]
    main_behaviours = [main_no_response, main_no_correct_response, \
        main_no_correct_chosen, main_correct_chosen, main_response_size]

    np.savetxt(constants.csv_filename('memory_average_precision', training_stage, experiment), precision, delimiter=',')
    np.savetxt(constants.csv_filename('memory_average_recall', training_stage, experiment), recall, delimiter=',')
    np.savetxt(constants.csv_filename('memory_average_accuracy', training_stage, experiment), accuracy, delimiter=',')
    np.savetxt(constants.csv_filename('memory_average_entropy', training_stage, experiment), entropy, delimiter=',')
    np.savetxt(constants.csv_filename('all_precision', training_stage, experiment), all_precision, delimiter=',')
    np.savetxt(constants.csv_filename('all_recall', training_stage, experiment), all_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_behaviours', training_stage, experiment), main_behaviours, delimiter=',')
    np.save(constants.data_filename('memory_cms', training_stage, experiment), all_cms)
    np.save(constants.data_filename('behaviours', training_stage, experiment), behaviours)
    plot_pre_graph(average_precision, average_recall, average_accuracy, average_entropy,\
        stdev_precision, stdev_recall, stdev_accuracy, stdev_entropy, training_stage )
    plot_pre_graph(all_precision_average, all_recall_average, None, average_entropy, \
        all_precision_stdev, all_recall_stdev, None, stdev_entropy, training_stage, 'overall')
    plot_size_graph(main_response_size, main_response_size_stdev, training_stage)
    plot_behs_graph(main_no_response, main_no_correct_response, main_no_correct_chosen,\
        main_correct_chosen, training_stage)
    print('Memory size evaluation completed!')
    return best_memory_size



def get_recalls(ams, msize, domain, min_value, max_value, trf, trl, tef, tel, fold, percent, training_stage):
    n_mems = constants.n_labels

    # To store precisión, recall, accuracy and entropies
    measures = np.zeros(constants.n_measures, dtype=np.float64)
    entropy = np.zeros(n_mems, dtype=np.float64)

    # Confusion matrix for calculating precision, recall and accuracy
    # per memory.
    cms = np.zeros((n_mems, 2, 2))
    TP = (0,0)
    FP = (0,1)
    FN = (1,0)
    TN = (1,1)

    # Confusion matrix for calculating overall precision and recall.
    cmatrix = np.zeros((2,2))

    # Registration in parallel, per label.
    Parallel(n_jobs=constants.n_jobs, require='sharedmem', verbose=50)(
        delayed(register_in_memory)(ams[label], features_list) \
            for label, features_list in split_by_label(zip(trf, trl)))

    print(f'Filling of memories done for idx {fold}')

    # Calculate entropies
    means = []
    for m in ams:
        entropy[m] = ams[m].entropy
        means.append(ams[m].mean)

    # Total number of differences between features and memories.
    mismatches = 0
    split_size = 500
    for mmatches, scms, cmatx in \
         Parallel(n_jobs=constants.n_jobs, verbose=50)(
            delayed(remember_by_memory)(fl_pairs, ams, entropy) \
            for fl_pairs in split_every(split_size, zip(tef, tel))):
        mismatches += mmatches
        cms  = cms + scms
        cmatrix = cmatrix + cmatx
    positives = conf_sum(cms, TP) + conf_sum(cms, FP)
    details = True
    if positives == 0:
        print('No memory responded')
        measures[constants.precision_idx] = 1.0
        details = False
    else:
        measures[constants.precision_idx] = memories_precision(cms)
    measures[constants.recall_idx] = memories_recall(cms)
    measures[constants.accuracy_idx] = memories_accuracy(cms)
    measures[constants.entropy_idx] = np.mean(entropy)
 
    if details:
        for i in range(n_mems):
            positives = cms[i][TP] + cms[i][FP]
            if positives == 0:
                print(f'Memory {i} filled with {percent}% in fold {fold} did not respond.')
    positives = cmatrix[TP] + cmatrix[FP]
    if positives == 0:
        print(f'System filled with {percent} in fold {fold} did not respond.')
        total_precision = 1.0
    else: 
        total_precision = cmatrix[TP] / positives
    total_recall = cmatrix[TP] / len(tel)
    mismatches /= len(tel)
    filename = constants.memory_conftrix_filename(percent, training_stage, fold)
    np.save(filename, cms)
    return measures, total_precision, total_recall, mismatches


def remember_by_memory(fl_pairs, ams, entropy):
    n_mems = constants.n_labels
    cms = np.zeros((n_mems, 2, 2), dtype='int')
    cmatrix = np.zeros((2,2), dtype='int')
    mismatches = 0
    for features, label in fl_pairs:
        mismatches += ams[label].mismatches(features)
        memories = []
        weights = {}
        for k in ams:
            recognized, weight = ams[k].recognize(features)
            if recognized:
                memories.append(k)
                weights[k] = weight
            # For calculation of per memory precision and recall
            cms[k][TP] += (k == label) and recognized
            cms[k][FP] += (k != label) and recognized
            cms[k][TN] += not ((k == label) or recognized)
            cms[k][FN] += (k == label) and not recognized
        if (len(memories) == 0):
            cmatrix[FN] += 1
        else:
            l = get_label(memories, weights, entropy)
            if l == label:
                cmatrix[TP] += 1
            else:
                cmatrix[FP] += 1
    return mismatches, cms, cmatrix

def get_means(d):
    n = len(d.keys())
    means = np.zeros((n, ))
    for k in d:
        rows = np.array(d[k])
        mean = rows.mean()
        means[k] = mean

    return means


def get_stdev(d):
    n = len(d.keys())
    stdevs = np.zeros((n, ))
    for k in d:
        rows = np.array(d[k])
        std = rows.std()
        stdevs[k] = std

    return stdevs    
    

# def test_recalling_fold(n_memories, mem_size, domain, fold, experiment, occlusion = None, bars_type = None, tolerance = 0):
#     # Create the required associative memories.
#     ams = dict.fromkeys(range(n_memories))
#     for j in ams:
#         ams[j] = AssociativeMemory(domain, mem_size, tolerance)

#     suffix = constants.filling_suffix
#     filling_features_filename = constants.features_name() + suffix        
#     filling_features_filename = constants.data_filename(filling_features_filename, fold)
#     filling_labels_filename = constants.labels_name + suffix        
#     filling_labels_filename = constants.data_filename(filling_labels_filename, fold)

#     suffix = constants.testing_suffix
#     testing_features_filename = constants.features_name(experiment, occlusion, bars_type) + suffix        
#     testing_features_filename = constants.data_filename(testing_features_filename, fold)
#     testing_labels_filename = constants.labels_name + suffix        
#     testing_labels_filename = constants.data_filename(testing_labels_filename, fold)

#     filling_features = np.load(filling_features_filename)
#     filling_labels = np.load(filling_labels_filename)
#     testing_features = np.load(testing_features_filename)
#     testing_labels = np.load(testing_labels_filename)

#     filling_max = filling_features.max()
#     testing_max = testing_features.max()
#     fillin_min = filling_features.min()
#     testing_min = testing_features.min()

#     maximum = filling_max if filling_max > testing_max else testing_max
#     minimum = fillin_min if fillin_min < testing_min else testing_min

#     total = len(filling_features)
#     percents = np.array(constants.memory_fills)
#     steps = np.round(total*percents/100.0).astype(int)

#     stage_recalls = []
#     stage_entropies = {}
#     stage_mprecision = {}
#     stage_mrecall = {}
#     total_precisions = []
#     total_recalls = []
#     mismatches = []

#     i = 0
#     for j in range(len(steps)):
#         k = steps[j]
#         features = filling_features[i:k]
#         labels = filling_labels[i:k]

#         recalls, measures, entropies, total_precision, total_recall, mis_count = get_recalls(ams, mem_size, domain, minimum, maximum, \
#             features, labels, testing_features, testing_labels, fold)

#         # A list of tuples (position, label, features)
#         stage_recalls += recalls

#         # An array with entropies per memory
#         stage_entropies[j] = entropies

#         # An array with precision per memory
#         stage_mprecision[j] = measures[constants.precision_idx,:]

#         # An array with recall per memory
#         stage_mrecall[j] = measures[constants.recall_idx,:]

#         # 
#         # Recalls and precisions per step
#         total_recalls.append(total_recall)
#         total_precisions.append(total_precision)

#         i = k

#         mismatches.append(mis_count)

#     return fold, stage_recalls, stage_entropies, stage_mprecision, \
#         stage_mrecall, np.array(total_precisions), np.array(total_recalls), np.array(mismatches)

def test_recalling(domain, mem_size, training_stage):
    n_memories = constants.n_labels
    memory_fills = constants.memory_fills
    testing_folds = constants.training_stages
    # All recalls, per memory fill and fold.
    # all_memories = {}
    # All entropies, precision, and recall, per fold, and fill.
    total_entropies = np.zeros((testing_folds, len(memory_fills)))
    total_precisions = np.zeros((testing_folds, len(memory_fills)))
    total_recalls = np.zeros((testing_folds, len(memory_fills)))
    total_accuracies = np.zeros((testing_folds, len(memory_fills)))
    sys_precisions = np.zeros((testing_folds, len(memory_fills)))
    sys_recalls = np.zeros((testing_folds, len(memory_fills)))
    total_mismatches = np.zeros((testing_folds, len(memory_fills)))

    list_results = []
    for fold in range(testing_folds):
        results = test_recalling_fold(n_memories, mem_size, domain, fold, training_stage)
        list_results.append(results)
    # for fold, memories, entropy, precision, recall, accuracy, \
    for fold, entropy, precision, recall, accuracy, \
        sys_precision, sys_recall, mismatches in list_results:
        # all_memories[fold] = memories
        total_precisions[fold] = precision
        total_recalls[fold] = recall
        total_accuracies[fold] = accuracy
        total_mismatches[fold] = mismatches
        total_entropies[fold] = entropy
        sys_precisions[fold] = sys_precision
        sys_recalls[fold] = sys_recall
    main_avrge_entropies = np.mean(total_entropies,axis=0)
    main_stdev_entropies = np.std(total_entropies, axis=0)
    main_avrge_mprecision = np.mean(total_precisions,axis=0)
    main_stdev_mprecision = np.std(total_precisions,axis=0)
    main_avrge_mrecall = np.mean(total_recalls,axis=0)
    main_stdev_mrecall = np.std(total_recalls,axis=0)
    main_avrge_maccuracy = np.mean(total_accuracies,axis=0)
    main_stdev_maccuracy = np.std(total_accuracies,axis=0)
    main_avrge_sys_precision = np.mean(sys_precisions,axis=0)
    main_stdev_sys_precision = np.std(sys_precisions,axis=0)
    main_avrge_sys_recall = np.mean(sys_recalls,axis=0)
    main_stdev_sys_recall = np.std(sys_recalls,axis=0)
    
    np.savetxt(constants.csv_filename('main_average_precision', training_stage), \
        main_avrge_mprecision, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_recall', training_stage), \
        main_avrge_mrecall, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_accuracy', training_stage), \
        main_avrge_maccuracy, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_entropy', training_stage), \
        main_avrge_entropies, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_precision', training_stage), \
        main_stdev_mprecision, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_recall', training_stage), \
        main_stdev_mrecall, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_accuracy', training_stage), \
        main_stdev_maccuracy, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_entropy', training_stage), \
        main_stdev_entropies, delimiter=',')
    np.savetxt(constants.csv_filename('main_total_recalls', training_stage), \
        main_avrge_sys_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_total_precision', training_stage), \
        main_avrge_sys_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_total_mismatches', training_stage), \
        total_mismatches, delimiter=',')

    plot_pre_graph(main_avrge_mprecision*100, main_avrge_mrecall*100, main_avrge_maccuracy*100, main_avrge_entropies,\
        main_stdev_mprecision*100, main_stdev_mrecall*100, main_stdev_maccuracy*100, main_stdev_entropies, training_stage, 'recall-', \
            xlabels = constants.memory_fills, xtitle = _('Percentage of memory corpus'))
    plot_pre_graph(main_avrge_sys_precision*100, main_avrge_sys_recall*100, None, main_avrge_entropies, \
        main_stdev_sys_precision*100, main_stdev_sys_recall*100, None, main_stdev_entropies, training_stage, 'total_recall-', \
            xlabels = constants.memory_fills, xtitle = _('Percentage of memory corpus'))

    bfp = best_filling_percentage(main_avrge_sys_precision, main_avrge_sys_recall)
    print('Best filling percent: ' + str(bfp))
    print('Filling evaluation completed!')
    return bfp

def best_filling_percentage(precisions, recalls):
    n = 0
    i = 0
    avg = -float('inf')
    for precision, recall in zip(precisions, recalls):
        new_avg = (precision + recall) / 2.0
        if avg < new_avg:
            n = i
            avg = new_avg
        i += 1
    return constants.memory_fills[n]

def test_recalling_fold(n_memories, mem_size, domain, fold, training_stage):
    # Create the required associative memories.
    ams = dict.fromkeys(range(n_memories))
    for j in ams:
        ams[j] = AssociativeMemory(domain, mem_size, tolerance=0)

    suffix = constants.filling_suffix
    filling_features_filename = constants.features_name + suffix        
    filling_features_filename = constants.data_filename(filling_features_filename, training_stage, fold)
    filling_labels_filename = constants.labels_name + suffix        
    filling_labels_filename = constants.data_filename(filling_labels_filename, training_stage, fold)

    suffix = constants.testing_suffix
    testing_features_filename = constants.features_name + suffix        
    testing_features_filename = constants.data_filename(testing_features_filename, training_stage, fold)
    testing_labels_filename = constants.labels_name + suffix        
    testing_labels_filename = constants.data_filename(testing_labels_filename, training_stage, fold)

    filling_features = np.load(filling_features_filename)
    filling_labels = np.load(filling_labels_filename)
    testing_features = np.load(testing_features_filename)
    testing_labels = np.load(testing_labels_filename)

    filling_max = filling_features.max()
    testing_max = testing_features.max()
    fillin_min = filling_features.min()
    testing_min = testing_features.min()

    maximum = filling_max if filling_max > testing_max else testing_max
    minimum = fillin_min if fillin_min < testing_min else testing_min

    filling_features = msize_features(filling_features, mem_size, minimum, maximum)
    testing_features = msize_features(testing_features, mem_size, minimum, maximum)

    total = len(filling_labels)
    percents = np.array(constants.memory_fills)
    steps = np.round(total*percents/100.0).astype(int)

    fold_entropies = []
    fold_precision = []
    fold_recall = []
    fold_accuracy = []
    total_precisions = []
    total_recalls = []
    mismatches = []

    start = 0
    for percent, end in zip(percents, steps):
        features = filling_features[start:end]
        labels = filling_labels[start:end]

        # recalls, measures, step_precision, step_recall, mis_count = get_recalls(ams, mem_size, domain, \
        measures, step_precision, step_recall, mis_count = get_recalls(ams, mem_size, domain,
            minimum, maximum, features, labels, testing_features, testing_labels, fold, percent, training_stage)

        # A list of tuples (position, label, features)
        # fold_recalls += recalls
        # An array with average entropy per step.
        fold_entropies.append(measures[constants.entropy_idx])
        # Arrays with precision, recall and accuracy per step
        fold_precision.append(measures[constants.precision_idx])
        fold_recall.append(measures[constants.recall_idx])
        fold_accuracy.append(measures[constants.accuracy_idx])
        # Overall recalls and precisions per step
        total_recalls.append(step_recall)
        total_precisions.append(step_precision)
        mismatches.append(mis_count)
        start = end
    # Use this to plot current state of memories
    # as heatmaps.
    # plot_memories(ams, es, fold)
    fold_entropies = np.array(fold_entropies)
    fold_precision = np.array(fold_precision)
    fold_recall = np.array(fold_recall)
    fold_accuracy = np.array(fold_accuracy)
    total_precisions = np.array(total_precisions)
    total_recalls = np.array(total_recalls)
    mismatches = np.array(mismatches)
    return fold, fold_entropies, fold_precision, \
        fold_recall, fold_accuracy, total_precisions, total_recalls, mismatches

# def test_recalling(domain, mem_size, experiment, occlusion = None, bars_type = None, tolerance = 0):
#     n_memories = constants.n_labels

#     all_recalls = {}
#     all_entropies = {}
#     all_mprecision = {}
#     all_mrecall = {}
#     total_precisions = np.zeros((constants.training_stages, len(constants.memory_fills)))
#     total_recalls = np.zeros((constants.training_stages, len(constants.memory_fills)))
#     total_mismatches = np.zeros((constants.training_stages, len(constants.memory_fills)))

#     xlabels = constants.memory_fills
#     list_results = Parallel(n_jobs=constants.n_jobs, verbose=50)(
#         delayed(test_recalling_fold)(n_memories, mem_size, domain, fold, experiment, occlusion, bars_type, tolerance) \
#             for fold in range(constants.training_stages))

#     for fold, stage_recalls, stage_entropies, stage_mprecision, stage_mrecall,\
#         total_precision, total_recall, mismatches in list_results:
#         all_recalls[fold] = stage_recalls
#         for msize in stage_entropies:
#             all_entropies[msize] = all_entropies[msize] + [stage_entropies[msize]] \
#                 if msize in all_entropies.keys() else [stage_entropies[msize]]
#             all_mprecision[msize] = all_mprecision[msize] + [stage_mprecision[msize]] \
#                 if msize in all_mprecision.keys() else [stage_mprecision[msize]]
#             all_mrecall[msize] = all_mrecall[msize] + [stage_mrecall[msize]] \
#                 if msize in all_mrecall.keys() else [stage_mrecall[msize]]
#             total_precisions[fold] = total_precision
#             total_recalls[fold] = total_recall
#             total_mismatches[fold] = mismatches

#     for fold in all_recalls:
#         list_tups = all_recalls[fold]
#         tags = []
#         memories = []
#         for (idx, label, features) in list_tups:
#             tags.append((idx, label))
#             memories.append(np.array(features))
        
#         tags = np.array(tags)
#         memories = np.array(memories)
#         memories_filename = constants.memories_name(experiment, occlusion, bars_type, tolerance)
#         memories_filename = constants.data_filename(memories_filename, fold)
#         np.save(memories_filename, memories)
#         tags_filename = constants.labels_name + constants.memory_suffix
#         tags_filename = constants.data_filename(tags_filename, fold)
#         np.save(tags_filename, tags)
    
#     main_avrge_entropies = get_means(all_entropies)
#     main_stdev_entropies = get_stdev(all_entropies)
#     main_avrge_mprecision = get_means(all_mprecision)
#     main_stdev_mprecision = get_stdev(all_mprecision)
#     main_avrge_mrecall = get_means(all_mrecall)
#     main_stdev_mrecall = get_stdev(all_mrecall)
    
#     np.savetxt(constants.csv_filename('main_average_precision',experiment, occlusion, bars_type, tolerance), \
#         main_avrge_mprecision, delimiter=',')
#     np.savetxt(constants.csv_filename('main_average_recall',experiment, occlusion, bars_type, tolerance), \
#         main_avrge_mrecall, delimiter=',')
#     np.savetxt(constants.csv_filename('main_average_entropy',experiment, occlusion, bars_type, tolerance), \
#         main_avrge_entropies, delimiter=',')

#     np.savetxt(constants.csv_filename('main_stdev_precision',experiment, occlusion, bars_type, tolerance), \
#         main_stdev_mprecision, delimiter=',')
#     np.savetxt(constants.csv_filename('main_stdev_recall',experiment, occlusion, bars_type, tolerance), \
#         main_stdev_mrecall, delimiter=',')
#     np.savetxt(constants.csv_filename('main_stdev_entropy',experiment, occlusion, bars_type, tolerance), \
#         main_stdev_entropies, delimiter=',')
#     np.savetxt(constants.csv_filename('main_total_recalls',experiment, occlusion, bars_type, tolerance), \
#         total_recalls, delimiter=',')
#     np.savetxt(constants.csv_filename('main_total_mismatches',experiment, occlusion, bars_type, tolerance), \
#         total_mismatches, delimiter=',')

#     plot_pre_graph(main_avrge_mprecision*100, main_avrge_mrecall*100, main_avrge_entropies,\
#         main_stdev_mprecision*100, main_stdev_mrecall*100, main_stdev_entropies, 'recall-', \
#             xlabels = xlabels, xtitle = _('Percentage of memory corpus'), action = experiment,
#             occlusion = occlusion, bars_type = bars_type, tolerance = tolerance)

#     plot_pre_graph(np.average(total_precisions, axis=0)*100, np.average(total_recalls, axis=0)*100, \
#         main_avrge_entropies, np.std(total_precisions, axis=0)*100, np.std(total_recalls, axis=0)*100, \
#             main_stdev_entropies, 'total_recall-', \
#             xlabels = xlabels, xtitle = _('Percentage of memory corpus'), action=experiment,
#             occlusion = occlusion, bars_type = bars_type, tolerance = tolerance)

#     print('Test completed')


def get_all_data(prefix, domain):
    data = None

    for stage in range(constants.training_stages):
        filename = constants.data_filename(prefix, stage)
        if data is None:
            data = np.load(filename)
        else:
            newdata = np.load(filename)
            data = np.concatenate((data, newdata), axis=0)

    return data

def characterize_features(domain, experiment, occlusion = None, bars_type = None):
    """ Produces a graph of features averages and standard deviations.
    """
    features_prefix = constants.features_name(experiment, occlusion, bars_type)
    tf_filename = features_prefix + constants.testing_suffix

    labels_prefix = constants.labels_name
    tl_filename = labels_prefix + constants.testing_suffix

    features = get_all_data(tf_filename, domain)
    labels = get_all_data(tl_filename, 1)

    d = {}
    for i in constants.all_labels:
        d[i] = []

    for (i, feats) in zip(labels, features):
        # Separates features per label.
        d[i].append(feats)

    means = {}
    stdevs = {}
    for i in constants.all_labels:
        # The list of features becomes a matrix
        d[i] = np.array(d[i])
        means[i] = np.mean(d[i], axis=0)
        stdevs[i] = np.std(d[i], axis=0)

    plot_features_graph(domain, means, stdevs, experiment, occlusion, bars_type)

def save_learn_params(mem_size, fill_percent, training_stage):
    name = constants.learn_params_name()
    filename = constants.data_filename(name, training_stage)
    np.save(filename, np.array([mem_size, fill_percent], dtype=int))  

def save_history(history, training_stage, prefix):
    """ Saves the stats of neural networks.

    Neural networks stats may come either as a History object, that includes
    a History.history dictionary with stats, or directly as a dictionary.
    """

    stats = {}
    stats['history'] = []
    for h in history:
        if type(h) is dict:
            stats['history'].append(h)
        else:
            stats['history'].append(h.history)

    with open(constants.json_filename(prefix, training_stage), 'w') as outfile:
        json.dump(stats, outfile)

    
##############################################################################
# Main section

def main(action, training_stage):#, occlusion = None, bar_type= None, tolerance = 0):
    """ Distributes work.

    The main function distributes work according to the options chosen in the
    command line.
    """

    if (action == constants.TRAIN_NN):
        # Trains the neural networks.
        training_percentage = constants.nn_training_percent
        model_prefix = constants.model_name
        stats_prefix = constants.stats_model_name        

        history = convnet.train_networks(training_stage, training_percentage, model_prefix, action)
        save_history(history, training_stage, stats_prefix)
    elif (action == constants.GET_FEATURES):
        # Generates features for the memories using the previously generated
        # neural networks.
        training_percentage = constants.nn_training_percent
        am_filling_percentage = constants.am_filling_percent
        model_prefix = constants.model_name
        features_prefix = constants.features_name #'features'#constants.features_name(action)
        labels_prefix = constants.labels_name
        data_prefix = constants.data_name

        history = convnet.obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix,
            training_percentage, am_filling_percentage, action, training_stage)
        save_history(history, training_stage, features_prefix)
    elif (action == constants.GET_FEATURES_IAM):
        # Generates features for the memories using the previously generated
        # neural networks.
        #training_percentage = constants.nn_training_percent #0.57
        #am_filling_percentage = constants.am_filling_percent #0.33
        model_prefix = constants.model_name # 'model' generated when trainning
        #Here I can modify the prefix to add the stage
        features_prefix = constants.features_name #features       
        data_prefix = constants.data_name  # 'data'
        action = constants.GET_FEATURES # 0
        #learning stage starting in 0
        learning_stage = 0
        constants.training_stage = training_stage

        convnet.obtain_features_iam(model_prefix, features_prefix, data_prefix,
                action)
     
        #save_history(history, features_prefix)
    elif action == constants.CHARACTERIZE:
        # Generates graphs of mean and standard distributions of feature values,
        # per digit class.
        characterize_features(constants.domain, action)
    elif action == constants.OPTIMIZATION:
        # Optimization of iota, kappa, tolerance and size memory using the SMAC algorithm
        run_optimization(constants.domain, action, training_stage)
    elif action == constants.INCREASE:
        # Increase the data in emnist using the iam dataset
        constants.training_stage = training_stage
        increase_data(constants.domain, action)
    elif (action == constants.EXP_1) or (action == constants.EXP_2):
        # The domain size, equal to the size of the output layer of the network.
        best_memory_size = test_memories(constants.domain, action, training_stage)
        print(f'Best memory size: {best_memory_size}')
        best_filling_percent = test_recalling(constants.domain, best_memory_size, training_stage)
        print(f'Best filling percent: {best_filling_percent}')
        save_learn_params(best_memory_size, best_filling_percent, training_stage)
    # elif (action == constants.EXP_3):
    #     test_recalling(constants.domain, constants.partial_ideal_memory_size, action)
    # elif (action == constants.EXP_4):
    #     convnet.remember(action)
    # elif (constants.EXP_5 <= action) and (action <= constants.EXP_10):
    #     # Generates features for the data sections using the previously generate
    #     # neural network, introducing (background color) occlusion.
    #     training_percentage = constants.nn_training_percent
    #     am_filling_percentage = constants.am_filling_percent
    #     model_prefix = constants.model_name
    #     features_prefix = constants.features_name(action, occlusion, bar_type)
    #     labels_prefix = constants.labels_name
    #     data_prefix = constants.data_name

    #     history = convnet.obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix,
    #         training_percentage, am_filling_percentage, action, occlusion, bar_type)
    #     save_history(history, features_prefix)
    #     characterize_features(constants.domain, action, occlusion, bar_type)
    #     test_recalling(constants.domain, constants.partial_ideal_memory_size,
    #         action, occlusion, bar_type, tolerance)
    #     convnet.remember(action, occlusion, bar_type, tolerance)



if __name__== "__main__" :
    """ Argument parsing.
    
    Basically, there is a parameter for choosing language (-l), one
    to train and save the neural networks (-n), one to create and save the features
    for all data (-f), one to characterize the initial features (-c), and one to run
    the experiments (-e).
    """

    num_stages = constants.num_stages_learning
    stages = []
    for i in range(num_stages):
        stages.append(str(i))

    parser = argparse.ArgumentParser(description='Associative Memory Experimenter.')
    parser.add_argument('-l', nargs='?', dest='lang', choices=['en', 'es'], default='en',
                       help='choose between English (en) or Spanish (es) labels for graphs.')
    
    parser.add_argument('-stage', nargs='?', choices=stages, dest='training_stage', help='Training stage, if is the first time running the code set the training stage to 0', required=True)

    #parser.add_argument('-t', nargs='?', dest='tolerance', type=int,
    #                    help='run the experiment with the tolerance given (only experiments 5 to 12).')
    
    #group = parser.add_mutually_exclusive_group(required=False)
    #group.add_argument('-o', nargs='?', dest='occlusion', type=float, 
    #                    help='run the experiment with a given proportion of occlusion (only experiments 5 to 12).')
    #group.add_argument('-b', nargs='?', dest='bars_type', type=int, 
    #                    help='run the experiment with chosen bars type (only experiments 5 to 12).')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n', action='store_const', const=constants.TRAIN_NN, dest='action',
                        help='train the neural networks, separating NN and AM training data (Separate Data NN).')
    group.add_argument('-f', action='store_const', const=constants.GET_FEATURES, dest='action',
                        help='get data features using the separate data neural networks.')
    group.add_argument('-fiam', action='store_const', const=constants.GET_FEATURES_IAM, dest='action',
                        help='get data features for the data set iam using the separate data neural networks. Before this step you have to train your model with option -n')
    group.add_argument('-c', action='store_const', const=constants.CHARACTERIZE, dest='action',
                        help='characterize the features from partial data neural networks by class.')
    group.add_argument('-s', action='store_const', const=constants.OPTIMIZATION, dest='action',
                        help='run smac optimization algoritm for optimize F1 score trought memory size, tolerance, kappa and iota. Run after training (option -n) and get features (option -f)')
    group.add_argument('-i', action='store_const', const=constants.INCREASE, dest='action',
                        help='Increase the amount of data using the iam dataset. Run after the first cicle of training (option -n), get features (option -f), optimization (option -s) and get features iam (option -fiam)')
    group.add_argument('-e', nargs='?', dest='nexp', type=int, 
                        help='run the experiment with that number, using separate data neural networks.')

    args = parser.parse_args()
    lang = args.lang
    #occlusion = args.occlusion
    #bars_type = args.bars_type
    #tolerance = args.tolerance
    action = args.action
    nexp = args.nexp
    training_stage = args.training_stage

    
    if lang == 'es':
        es = gettext.translation('ame', localedir='locale', languages=['es'])
        es.install()

    # if not (occlusion is None):
    #     if (occlusion < 0) or (1 < occlusion):
    #         print_error("Occlusion needs to be a value between 0 and 1")
    #         exit(1)
    #     elif (nexp is None) or (nexp < constants.EXP_5) or (constants.EXP_8 < nexp):
    #         print_error("Occlusion is only valid for experiments 5 to 8")
    #         exit(2)
    # elif not (bars_type is None):
    #     if (bars_type < 0) or (constants.N_BARS <= bars_type):
    #         print_error("Bar type must be a number between 0 and {0}"\
    #                     .format(constants.N_BARS-1))
    #         exit(1)
    #     elif (nexp is None) or (nexp < constants.EXP_9):
    #         print_error("Bar type is only valid for experiments 9 to {0}"\
    #                     .format(constants.MAX_EXPERIMENT))
    #         exit(2)


    # if tolerance is None:
    #     tolerance = 0
    # elif (tolerance < 0) or (constants.domain < tolerance):
    #         print_error("tolerance needs to be a value between 0 and {0}."
    #             .format(constants.domain))
    #         exit(3)
    # elif (nexp is None) or (nexp < constants.EXP_5):
    #     print_error("tolerance is only valid from experiments 5 on")
    #     exit(2)

    if action is None:
        # An experiment was chosen
        if (nexp < constants.MIN_EXPERIMENT) or (constants.MAX_EXPERIMENT < nexp):
            print_error("There are only {1} experiments available, numbered consecutively from {0}."
                .format(constants.MIN_EXPERIMENT, constants.MAX_EXPERIMENT))
            exit(1)
        main(nexp, training_stage )#, occlusion, bars_type, tolerance)
    else:
        # Other action was chosen
        main(action, training_stage)

    
    
