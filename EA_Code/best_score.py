#!/usr/bin/env python

import numpy as np
import os
import pickle

#dir_path_data = '/home/iammoresentient/phd_lab/data/170427_k_10g10i10m50c'
dir_path_data = os.getcwd()

min_score = [] 

#load new data:
arr_best_scores = []
arr_best_inds = []
arr_end_scores = []
dir_to_check = dir_path_data 
files = os.listdir(dir_to_check)
#print(files)
for i in range(0,len(files)):
    filename = dir_to_check + '/' + files[i]
    if os.path.isfile(filename):
        if '.pickled' in files[i]:
            #print(filename)
            with open(filename, 'rb') as f:
                arr_to_unpickle = pickle.load(f)
            arr_best_score, arr_best_ind = arr_to_unpickle
            #temp_end_score = arr_best_score[-1]
            #arr_end_scores.append(temp_end_score)
            #arr_best_scores.append(arr_best_score)
            #arr_best_inds.append(arr_best_ind)
            
            if len(min_score) == 0 or arr_best_score[-1] < min_score[0]:
                min_score = [arr_best_score[-1], filename]
            #print(len(arr_to_unpickle), len(arr_best_score), len(arr_best_ind))
    else:
        break

print(min_score)
