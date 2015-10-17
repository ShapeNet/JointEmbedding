#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import bpy
import sys
import math
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *
from utilities_math import *
 
shape_file = sys.argv[-4]
shape_synset = sys.argv[-3]
shape_md5 = sys.argv[-2]
shape_view_params_file = sys.argv[-1]
syn_images_folder = os.path.join(g_syn_images_folder, shape_synset, shape_md5) 
view_params = [[float(x) for x in line.strip().split(' ')] for line in open(shape_view_params_file).readlines()]

if not os.path.exists(syn_images_folder):
    os.makedirs(syn_images_folder)

bpy.ops.import_scene.obj(filepath=shape_file) 

bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
#bpy.context.scene.render.use_shadows = False
#bpy.context.scene.render.use_raytrace = False

# disable material raytrace
for material_idx in range(len(bpy.data.materials)):
    bpy.data.materials[material_idx].use_transparency = False

camObj = bpy.data.objects['Camera']
# camObj.data.lens_unit = 'FOV'
# camObj.data.angle = 0.2

# YOUR CODE START HERE

for param in view_params:
    azimuth_deg = param[0]
    elevation_deg = param[1]
    theta_deg = param[2]
    rho = g_syn_camera_dist # fixed distance

    # clear default lights
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    # set environment lighting
    #bpy.context.space_data.context = 'WORLD'
    bpy.context.scene.world.light_settings.use_environment_light = True
    bpy.context.scene.world.light_settings.environment_energy = np.random.uniform(0, 1)
    bpy.context.scene.world.light_settings.environment_color = 'PLAIN'

    # set point lights
    for i in range(random.randint(g_syn_light_num_lowbound,g_syn_light_num_highbound)):
        light_azimuth_deg = np.random.uniform(0, 360)
        light_elevation_deg  = np.random.uniform(-90, 90)
        light_dist = np.random.uniform(g_syn_light_dist_lowbound, g_syn_light_dist_highbound)
        lx, ly, lz = obj_centened_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
        bpy.ops.object.lamp_add(type='POINT', location=(lx, ly, lz))
    for lamp_idx in range(len(bpy.data.lamps)):
        bpy.data.lamps[lamp_idx].data.energy = np.random.normal(2, 2)

    cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
    q1 = camPosToQuaternion(cx, cy, cz)
    q2 = camRotQuaternion(cx, cy, cz, theta_deg)
    q = quaternionProduct(q2, q1)
    camObj.location[0] = cx
    camObj.location[1] = cy 
    camObj.location[2] = cz
    camObj.rotation_mode = 'QUATERNION'
    camObj.rotation_quaternion[0] = q[0]
    camObj.rotation_quaternion[1] = q[1]
    camObj.rotation_quaternion[2] = q[2]
    camObj.rotation_quaternion[3] = q[3]
    syn_image_file = './%s_%s_a%03d_e%03d_t%03d_d%03d.png' % (shape_synset, shape_md5, round(azimuth_deg), round(elevation_deg), round(theta_deg), round(rho))
    bpy.data.scenes['Scene'].render.filepath = os.path.join(syn_images_folder, syn_image_file)
    bpy.ops.render.render( write_still=True )

