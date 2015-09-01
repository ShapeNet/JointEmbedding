#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *



# input: weights array
# output: index of weight/item choosed
def weighted_choice(weights):
    total = sum(w for w in weights)
    r = random.uniform(0, total)
    upto = 0
    for i, w in enumerate(weights):
        if upto + w > r:
            return i
        upto += w


view_distr_distance = 3
view_distr_N = 1000000
for synset in all_shapenet_synset_set:
    if synset not in g_view_distribution_params:
        continue
    view_distr_filename = g_view_distribution_files[synset]
    view_distr_azimuth_weights = g_view_distribution_params[synset][0]
    view_distr_elevation_weights = g_view_distribution_params[synset][1]
    view_distr_tilt_deviation = g_view_distribution_params[synset][2]

    fout = open(view_distr_filename,'w')
    for _ in range(view_distr_N):
        azimuth_deg = weighted_choice(view_distr_azimuth_weights) * 22.5 + np.random.uniform(-11.25, 11.25) 
        elevation_deg = weighted_choice(view_distr_elevation_weights) * 10 - 85 + np.random.uniform(-5,5) 
        tilt_deg = np.random.normal(0,view_distr_tilt_deviation) 
        distance = view_distr_distance 
        fout.write('%f %f %f %f\n' % (azimuth_deg, elevation_deg, tilt_deg, distance))
    fout.close()
