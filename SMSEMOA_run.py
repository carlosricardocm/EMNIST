# SMAC3
#from ConfigSpace import Configuration, ConfigurationSpace
#from smac.scenario.scenario import Scenario
#from smac.facade.smac_bb_facade import SMAC4BB as BBFacade
# EAM
import constants
import convnet
from associative import AssociativeMemory
# Other
import os
import sys
import gc
import argparse
import numpy as np
#from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import random
from optproblems import Problem
from evoalgos.algo import SMSEMOA
from evoalgos.individual import ESIndividual

# Directorio de salida
smac_outdir = "SMS-EMOA_results"
# Historico de configuraciones intentadas
statsfilename = smac_outdir+"/EMNIST_configurations_new2023.csv"

if not os.path.exists(smac_outdir):
    os.makedirs(smac_outdir)
if not os.path.exists(os.path.join(os.getcwd(), statsfilename)):
    with open(os.path.join(os.getcwd(), statsfilename), "w+") as outf:
        outf.write("OBJ_1 = (-1)*Suma de Precisions, OBJ_2 = (-1)*Suma de Recalls, F1 (Promedio), F1 (desviación estándar), Tamaño de memoria, Tolerancia, Sigma, Iota, Kappa, Precisión (promedio), Precisión (desviación estándar), Recall (promedio), Recall (desviación estándar), Entropía (promedio), Entropía (desviación estándar)\n")


def get_label(memories, entropies = None):

    # Random selection
    if entropies is None:
        i = random.atddrange(len(memories))
        return memories[i]
    else:
        i = memories[0]
        entropy = entropies[i]

        for j in memories[1:]:
            if entropy > entropies[j]:
                i = j
                entropy = entropies[i]

    return i

# Función auxiliar a la función objetivo
# Crea un sistema de memoria w-ams usando los parametros dados y lo evalua
def get_wams_results(msize, domain, lpm, trf, tef, trl, tel, tolerance, sigma, iota, kappa):
    # Round the values
    max_value = trf.max()
    other_value = tef.max()
    max_value = max_value if max_value > other_value else other_value

    min_value = trf.min()
    other_value = tef.min()
    min_value = min_value if min_value < other_value else other_value

    trf_rounded = np.round((trf-min_value) * (msize - 1) / (max_value-min_value)).astype(np.int16)
    tef_rounded = np.round((tef-min_value) * (msize - 1) / (max_value-min_value)).astype(np.int16)

    n_labels = constants.n_labels
    nmems = int(n_labels/lpm)
    print("Num de memorias: {}".format(nmems))

    measures = np.zeros((constants.n_measures, nmems), dtype=np.float64)
    entropy = np.zeros((nmems, ), dtype=np.float64)
    behaviour = np.zeros((constants.n_behaviours, ))

    # Confusion matrix for calculating precision and recall per memory.
    cms = np.zeros((nmems, 2, 2))
    TP = (0,0)
    FP = (0,1)
    FN = (1,0)
    TN = (1,1)

    # Create the required associative memories.
    ams = dict.fromkeys(range(nmems))
    #tolerance = 5
    #sigma = 0.1
    #iota = 0.3
    #kappa = 1.5
    for j in ams:
        ams[j] = AssociativeMemory(domain, msize, tolerance,
                    sigma, iota, kappa)  #AssociativeMemory(domain, msize)

    # Registration
    for features, label in zip(trf_rounded, trl):
        i = int(label/lpm)
        ams[i].register(features)

    # Calculate entropies
    for j in ams:
        entropy[j] = ams[j].entropy

    # Recognition
    response_size = 0

    for features, label in zip(tef_rounded, tel):
        correct = int(label/lpm)

        memories = []
        for k in ams:
            recognized, weight = ams[k].recognize(features)

            # For calculation of per memory precision and recall
            if (k == correct) and recognized:
                cms[k][TP] += 1
            elif k == correct:
                cms[k][FN] += 1
            elif recognized:
                cms[k][FP] += 1
            else:
                cms[k][TN] += 1

            # For calculation of behaviours, including overall precision and recall.
            if recognized:
                memories.append(k)

        response_size += len(memories)
        if len(memories) == 0:
            # Register empty case
            behaviour[constants.no_response_idx] += 1
        elif not (correct in memories):
            behaviour[constants.no_correct_response_idx] += 1
        else:
            l = get_label(memories, entropy)
            if l != correct:
                behaviour[constants.no_correct_chosen_idx] += 1
            else:
                behaviour[constants.correct_response_idx] += 1

    behaviour[constants.mean_responses_idx] = response_size /float(len(tef_rounded))
    all_responses = len(tef_rounded) - behaviour[constants.no_response_idx]
    all_precision = (behaviour[constants.correct_response_idx])/float(all_responses)
    all_recall = (behaviour[constants.correct_response_idx])/float(len(tef_rounded))

    print("Sanity check {} {}".format(constants.correct_response_idx, behaviour))
    print("Sanity check {} {} {}".format(all_responses,all_precision,all_recall))

    behaviour[constants.precision_idx] = all_precision
    behaviour[constants.recall_idx] = all_recall

    precision_sum = 0.0
    recall_sum = 0.0
    F1_sum = 0.0
    precisions = []
    recalls = []
    F1s = []

    for i in range(nmems):
        print(cms[i])
        if (cms[i][TP] + cms[i][FP]) > 0:
            pre = cms[i][TP] /(cms[i][TP] + cms[i][FP])
            rec = cms[i][TP] /(cms[i][TP] + cms[i][FN])
        else:
            pre = 0.0
            rec = 0.0
        if (pre + rec) > 0:
            F1 = (2.0 * pre * rec) / (pre + rec)
        else:
            F1 = 0.0
        measures[constants.precision_idx,i] = pre
        measures[constants.recall_idx,i] = rec
        precisions.append(pre)
        recalls.append(rec)
        precision_sum += pre
        recall_sum += rec
        F1_sum += F1
        F1s.append(F1)


    #score = (-1)*(F1_sum) # el objetivo de todo esto es minimizar este valor
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    entropies = np.array(entropy)
    F1s = np.array(F1s)

    # Guardamos resultados en archivo statsfilename
    print("Score = -1*(Suma de F1s), F1 (Promedio), F1 (desviación estándar),  Tamaño de memoria, Tolerancia, Sigma, Iota, Kappa, Precisión (promedio), Precisión (desviación estándar), Recall (promedio), Recall (desviación estándar), Entropía (promedio), Entropía (desviación estándar)")
    outdata1 = [(-1*precision_sum), (-1*recall_sum), F1s.mean(), F1s.std(), msize, tolerance, sigma, iota, kappa, precisions.mean(), precisions.std(), recalls.mean(), recalls.std(), entropies.mean(), entropies.std()]
    outdata2 = []
    for val in outdata1:
        outdata2.append(str(val))
    run_results_str = ",".join(outdata2)
    run_results_str += "\n"
    print(run_results_str)
    with open(os.path.join(os.getcwd(), statsfilename), 'a+') as outf:
        outf.write(run_results_str)

    return ( (-1)*(precision_sum), (-1)*(recall_sum), measures, entropy, behaviour)




# Funcion objetivo
# Recibe una configuracion y devuelve un valor real que consiste en
# Para cada memoria sumar las precisiones y recalls:
# (-1)*(SumaDeTodasLasPrecisiones + SumadeTodoslosRecall)
# Este valor es mínimo cuando el precision y recall de todas las memorias es 1
def evaluate_memory_config(config) -> float:  #(self, config: Configuration, seed: int) -> float:
#def test_memories(domain, prefix, experiment):
    domain = constants.domain
    prefix = constants.partial_prefix
    experiment = 1
    sigma = 0.2
    iota = config[0]
    kappa = config[1]
    msize = config[2]
    tolerance = config[3]

    # Las cotas de parametros seran implementadas por la función de optimización
    if iota < 0.0 or kappa < 0.0 or msize < 0.0 or tolerance < 0.0:
        return 0.0, 0.0
    if iota > 10  or kappa > 10.0 or msize > 300.0 or tolerance > 64:
        return 0.0, 0.0
    msize = int(math.floor(msize))
    tolerance = int(math.floor(tolerance))

    labels_x_memory = constants.labels_per_memory[experiment] # = 1
    n_memories = int(constants.n_labels/labels_x_memory) # = n_labels

    if prefix == constants.partial_prefix:
        suffix = constants.filling_suffix
    elif prefix == constants.full_prefix:
        suffix = constants.training_suffix

    i = constants.training_stages - 1
    #for i in range(constants.training_stages):
    gc.collect()

    training_features_filename = prefix + constants.features_name + suffix
    training_features_filename = constants.data_filename(training_features_filename, i)
    training_labels_filename = prefix + constants.labels_name + suffix
    training_labels_filename = constants.data_filename(training_labels_filename, i)

    suffix = constants.testing_suffix
    testing_features_filename = prefix + constants.features_name + suffix
    testing_features_filename = constants.data_filename(testing_features_filename, i)
    testing_labels_filename = prefix + constants.labels_name + suffix
    testing_labels_filename = constants.data_filename(testing_labels_filename, i)

    training_features = np.load(training_features_filename)
    training_labels = np.load(training_labels_filename)
    testing_features = np.load(testing_features_filename)
    testing_labels = np.load(testing_labels_filename)

    # Each memory has precision and recall
    #measures_per_size = np.zeros((1, n_memories, constants.n_measures), dtype=np.float64)

    # An entropy value per memory size and memory.
    #entropies = np.zeros((1, n_memories), dtype=np.float64)
    #behaviours = np.zeros((1, constants.n_behaviours))

    precision_sum, recall_sum, measures, entropy, behaviour = get_wams_results(msize, domain, labels_x_memory, \
                    training_features, testing_features, training_labels, testing_labels, \
                    tolerance, sigma, iota, kappa)
    return precision_sum, recall_sum


problem = Problem(evaluate_memory_config, num_objectives=2, max_evaluations=20000, name="WEAM_EMNIST")
dim = 4 # parametro para sbx
popsize = 20
population = []
init_step_sizes = [0.25]
for _ in range(popsize):
    genes = []
    genes.append(random.uniform(0.0, 5.0)) # iota
    genes.append(random.uniform(0.0, 5.0)) # kappa
    genes.append(random.randint(0, 300))   # memsize
    genes.append(random.randint(0, 64))    # tolerance
    population.append(ESIndividual(genome=genes,
                                   learning_param1=1.0/math.sqrt(dim),
                                   learning_param2=0.0,
                                   strategy_params=init_step_sizes,
                                   recombination_type="none",
                                   num_parents=2))

ea = SMSEMOA(problem, population, popsize, num_offspring=20)
ea.run()
for individual in ea.population:
    print(individual)
