# SMAC3
from ConfigSpace import Configuration, ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter
from smac.scenario.scenario import Scenario
from smac.facade.smac_bb_facade import SMAC4BB as BBFacade
from smac.facade.smac_hpo_facade import SMAC4HPO as HPOFacade
from smac.initial_design.default_configuration_design import DefaultConfiguration
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
import random

hl = 2.0 # cota superior de parametros iota, kappa
maxtol = 64 # cota superior para parametro de tolerancia
maxmemsize = 300 # cota superior del tamaño de memoria
experiment = 1
labels_x_memory = constants.labels_per_memory[experiment] # = 1
n_memories = int(constants.n_labels/labels_x_memory) # = n_labels

# Solucion inicial para empezar la busqueda
prior_memsize = 23
prior_tolerance = 17
prior_kappa = 0.00151258507990818
prior_iota = 0.601543557593993

# Directorio de salida
smac_outdir = "SMAC_results"
smac_filename = "EMNIST_HPOFacade_with_priors.csv"
# Historico de configuraciones intentadas
statsfilename = os.path.join(smac_outdir, smac_filename)
headstr = "Score = (-1)*Suma de F1s, F1 (Promedio), F1 (desviación estándar), Precisión (promedio), Precisión (desviación estándar), Recall (promedio), Recall (desviación estándar), Entropía (promedio), Entropía (desviación estándar), "
for j in range(n_memories):
    headstr += "EAM{0} tolerancia, EAM{0} sigma, EAM{0} iota, EAM{0} kappa, EAM{0} Tamaño, EAM{0} F1, EAM{0} Precision, EAM{0} Recall, EAM{0} Entropia, ".format(j)

if not os.path.exists(smac_outdir):
    os.makedirs(smac_outdir)
if not os.path.exists(os.path.join(os.getcwd(), statsfilename)):
    with open(os.path.join(os.getcwd(), statsfilename), "w+", encoding="utf-8") as outf:
        outf.write(headstr[:-2]+"\n")


# SMAC necesita 4 componentes:
# Espacio de configuracion
# Funcion objetivo
# Escenario
# Fachada



#config = {}
#for j in range(n_memories):
#    config[] = (0, maxtol)
#    config[str(j)+"_sigma"] = (0.0, hl)
#    config[str(j)+"_iota"] = (0.0, hl)
#    config[str(j)+"_kappa"] = (0.0, hl)
#    config[str(j)+"_memory_size"] = (1, 500)
#cs = ConfigurationSpace(config)



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
def get_wams_results(config, domain, lpm, trf, tef, trl, tel): #tolerance, sigma, iota, kappa):
    # Round the values
    max_value = trf.max()
    other_value = tef.max()
    max_value = max_value if max_value > other_value else other_value

    min_value = trf.min()
    other_value = tef.min()
    min_value = min_value if min_value < other_value else other_value

    n_labels = constants.n_labels
    nmems = int(n_labels/lpm)

    print("Num de memorias: {}".format(nmems))

    measures = np.zeros((constants.n_measures, nmems), dtype=np.float64)
    entropy = np.zeros((nmems, ), dtype=np.float64)
    behaviour = np.zeros((constants.n_behaviours, ))

    # Confusion matrix for calculating F1, precision and recall per memory.
    cms = np.zeros((nmems, 2, 2))
    TP = (0,0)
    FP = (0,1)
    FN = (1,0)
    TN = (1,1)

    # Create the required associative memories.
    ams = dict.fromkeys(range(nmems))
    max_msize = 0 # para normalizacion


    tolerance = config["tolerance"]
    sigma = 0.2 #config[str(j)+"_sigma"]
    iota = config["iota"]
    kappa = config["kappa"]
    msize = config["memory_size"]
    for j in ams:
        if msize > max_msize:
            max_msize = msize
        ams[j] = AssociativeMemory(domain, msize, tolerance, sigma, iota, kappa)

    trf_rounded = np.round((trf-min_value) * (max_msize - 1) / (max_value-min_value)).astype(np.int16)
    tef_rounded = np.round((tef-min_value) * (max_msize - 1) / (max_value-min_value)).astype(np.int16)
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

    precision_sum = 0.0 # defunct
    recall_sum = 0.0 # defunct
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


    score = (-1)*(F1_sum) # el objetivo de smac es minimizar este valor

    # Parseamos la configuración en forma de renglón para el csv
    outconfig = []
    print("Núm memoria, tolerancia, sigma, iota, kappa, tamaño memoria, F1, Precision, Recall, Entropía")
    for j in ams:
        #tolerance = config[str(j)+"_tolerance"]
        #sigma = 0.2 #config[str(j)+"_sigma"]
        #iota = config[str(j)+"_iota"]
        #kappa = config[str(j)+"_kappa"]
        #msize = config[str(j)+"_memory_size"]

        row = [tolerance, sigma, iota, kappa, msize, F1s[j], precisions[j], recalls[j], entropy[j]]
        print(",".join([str(j)]+[str(r) for r in row]))
        outconfig += row

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    entropies = np.array(entropy)
    F1s = np.array(F1s)

    # Guardamos resultados en archivo statsfilename
    #print("\n\nScore = -1*(Suma de F1s), F1 (Promedio), F1 (desviación estándar), Precisión (promedio), Precisión (desviación estándar), Recall (promedio), Recall (desviación estándar), Entropía (promedio), Entropía (desviación estándar)")
    outdata1 = [score, F1s.mean(), F1s.std(), precisions.mean(), precisions.std(), recalls.mean(), recalls.std(), entropies.mean(), entropies.std()]
    outdata2 = []
    for val in outdata1 + outconfig:
        outdata2.append(str(val))
    run_results_str = ",".join(outdata2)
    run_results_str += "\n"
    with open(os.path.join(os.getcwd(), statsfilename), 'a+') as outf:
        outf.write(run_results_str)

    print("Score {}".format(score))
    return (score, measures, entropy, behaviour)




# Funcion objetivo
# Recibe una configuracion y devuelve un valor real que consiste en
# Para cada memoria sumar las precisiones y recalls:
# (-1)*(SumaDeTodasLasPrecisiones + SumadeTodoslosRecall)
# Este valor es mínimo cuando el precision y recall de todas las memorias es 1
def evaluate_memory_config(config: Configuration) -> float:  #(self, config: Configuration, seed: int) -> float:
#def test_memories(domain, prefix, experiment):
    domain = constants.domain
    prefix = constants.partial_prefix
    experiment = 1
    #tolerance = config["tolerance"]
    #sigma = config["sigma"]
    #iota = config["iota"]
    #kappa = config["kappa"]
    #msize = config["memory_size"]

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

    score, measures, entropy, behaviour = get_wams_results(config, domain, labels_x_memory, \
                    training_features, testing_features, training_labels, testing_labels)#, \
                    #tolerance, sigma, iota, kappa)
    return score









# Optimize tolerance, iota, kappa and memory size params
def optimize():
    print("optimize")
    # Generamos una configuración para todas las memoria, con el conocimiento a priori que contamos
    config = []
    config.append(UniformIntegerHyperparameter("tolerance", lower=0, upper=maxtol, default_value=prior_tolerance)) #(1, 10), default=4))
    #config.append(UniformFloatHyperparameter(str(j)+"_sigma", lower=0.0, upper=hl, default_value=prior_sigma))
    config.append(UniformFloatHyperparameter("iota", lower=0.0, upper=hl, default_value=prior_iota))
    config.append(UniformFloatHyperparameter("kappa", lower=0.0, upper=hl, default_value=prior_kappa))
    config.append(UniformIntegerHyperparameter("memory_size", lower=0, upper=maxmemsize, default_value=prior_memsize))

    cs = ConfigurationSpace()
    cs.add_hyperparameters(config)
    print(config)

    # 3. Escenario
    # Seleccionar variables de ambiente
    scenario = Scenario({
    "cs":cs,
    #"output-directory": smac_outdir,
    "run_obj": "quality",
    #"wallclock_limit": 7*86400, #863400 = 24hrs = 2 *  12 hrs = 12*60*60 secs
    #"n-workers":32,  # Use 32 workers
    "deterministic":"true", # hace que solo 1 semilla se pruebe en cada ejecucion de la funcion objetivo
    "runcount-limit": 80, #Num evaluations
    "n_trials":"1",  # Evaluated max 500 trials
    })

    smac = HPOFacade(scenario=scenario, tae_runner=evaluate_memory_config, initial_design=DefaultConfiguration) #target_function=evaluate_memory_config)
    best_found_config = smac.optimize()   
    #best_found_config = best_found_config.replace("Configuration(values=", "").replace(")","")
    #parsed_best_found_config = eval(best_found_config)
    #print("Best parsed config params")
    #print(parsed_best_found_config)
    return best_found_config

#smac = HPOFacade(scenario=scenario, tae_runner=evaluate_memory_config, initial_design=DefaultConfiguration) #target_function=evaluate_memory_config)
#best_found_config = smac.optimize()
#print(best_found_config)
