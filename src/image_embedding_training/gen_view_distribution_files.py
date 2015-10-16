#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
from utilities_common import query_yes_no

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
for synset in g_shapenet_synset_set:
    if synset not in g_view_distribution_params:
        print 'Please define view distribution file generation parameters for synset %s in global_variables.py'%(synset)
        continue

    view_distr_filename = g_view_distribution_files[synset]
    if os.path.exists(view_distr_filename):
        if query_yes_no(('It seems that you already have view distribution file \"%s\" for synset %s, skip generating?')%(view_distr_filename, synset)):
            continue
        else:
            os.remove(view_distr_filename)
    
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
