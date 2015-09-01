#!/usr/bin/env python

import glob
import os
import sys
from functools import partial
from multiprocessing.dummy import Pool
import argparse
from subprocess import call

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(BASE_DIR)))
from global_variables import *

if __name__ == '__main__':
    num = g_fixedview_image_per_model
    renderRootFolder = g_fixedview_image_folder
    render_program = os.path.join(BASE_DIR, 'render_fixedview.py')
    shape_list = [line.strip().split(' ') for line in open(g_shape_list_file, 'r')]
    
    if not os.path.exists(renderRootFolder):
        os.mkdir(renderRootFolder,0777) 

    tcount = 0
    cmd = []

    # Loop over all the models.
    for shape_property in shape_list:
        shape_synset = shape_property[0]
        shape_md5 = shape_property[1]
        shape_file = os.path.join(g_shapenet_root_folder, shape_synset, shape_md5, 'model.obj')
        renderFolder = os.path.join(renderRootFolder, shape_md5)

        if not os.path.exists(renderFolder):
            cmd.append('mkdir -p %s' % renderFolder)
        elif len(glob.glob(os.path.join(renderFolder, '*.png'))) >= num:
            print 'skip', shape_md5
            continue

        command_per_model = '%s %s --background --python %s -- %s %s %s %s > /dev/null 2>&1' % (g_blender_executable_path, g_blank_blend_file_path, render_program, shape_file, renderFolder)
        cmd.append(command_per_model) # backup

    pool = Pool(12) # 5 concurrent commands at a time
    for i, returncode in enumerate(pool.imap(partial(call, shell=True), cmd)):
	print i
	if returncode != 0:
	    print("%d command failed: %d" % (i, returncode)) 
	    tcount = tcount + 1
	    print tcount
