# Copyright [2020] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
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

import csv
import gettext
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import constants
import convnet 
import collections
from collections import Counter


labels_to_char = [] 

def plot_freqs(frequencies, prefix):
    plt.clf()
    x = labels_to_char
    width = .5  # the width of the bars: can also be len(x) sequence
    plt.bar(x, frequencies, width)
    plt.xlabel(('Characters'))
    plt.ylabel(('Frequency'))
    plt.xticks(rotation=45,fontsize=4)
    graph_filename = constants.picture_filename(prefix + ('-english'), "0")
    plt.savefig(graph_filename, dpi=600)

def plot_matrix(matrix, prefix):
    plt.clf()
    plt.figure(figsize=(6.4, 4.8))
    max_value = np.max(matrix)
    tick_labels_x = labels_to_char
    tick_labels_y = labels_to_char          
    #tick_labels_x = labels_to_char[::-1]
    seaborn.heatmap(matrix/max_value, xticklabels=tick_labels_x,
        yticklabels=tick_labels_y, vmin=0.0, vmax=1.0, cmap='coolwarm')    
    plt.xlabel(('Second Character'))
    plt.xticks(rotation=45,fontsize=4)
    plt.xticks()
    plt.ylabel(('First Character'))
    plt.yticks(rotation=45,fontsize=4)
    filename = constants.picture_filename(prefix + ('-english'), "0")    
    plt.savefig(filename, dpi=600)


def unique(lists):
    lista = list(Counter(lists).keys())
    return lista
 

def bigram_matrix():
    
    global char_to_labels
    global labels_to_char
    
    all_data = convnet.get_data_iam(bigram=True)
    
    lista_chars = []   
    for line in all_data:
        for char in line:
            lista_chars.append(char)
        
    labels_to_char = unique(lista_chars)        
    total_unique_chars = len(labels_to_char)
   
   
    char_to_labels = {}
    for index, char in enumerate(labels_to_char):
        char_to_labels[char] = index 

    frequencies = np.zeros(total_unique_chars, dtype=np.double) 
    matrix = np.zeros((total_unique_chars, total_unique_chars), dtype=np.double) 

    
    for line in all_data:
        contador = 0
        previous = 0
        for char in line:
            label = char_to_labels[char]
            frequencies[label]  += 1
            if contador != 0:
                matrix[previous, label] += 1    
            previous = label
            contador += 1
    
 

    return matrix,frequencies


if __name__== "__main__" : 
     
    matrix, frequencies = bigram_matrix() 
    plot_matrix(matrix, 'bigrams')     
    filename = constants.csv_filename('bigrams', "0")
    np.savetxt(filename, matrix, fmt='%d', delimiter=',')
    totals = np.sum(matrix, axis=1)
    matrix = matrix / totals[:, None]
    filename = constants.data_filename('bigrams', "0")
    np.save(filename, matrix)    
    plot_freqs(frequencies, 'frecuencies')
    filename = constants.csv_filename('frequencies', "0")
    np.savetxt(filename, frequencies, fmt='%d', delimiter=',')
    frequencies = frequencies/np.sum(frequencies)
    filename = constants.data_filename('frequencies',"0")
    np.save(filename, frequencies)
    filename = constants.data_filename('ltochars', "0")
    _ltochars = np.array(labels_to_char)
    np.save(filename, _ltochars)
    filename = constants.data_filename('ctolabels', "0")
    _ctolabels = np.array(char_to_labels)
    np.save(filename, _ctolabels)
   
    
  

