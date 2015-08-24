#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from subprocess import call

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
from global_variables import *

call(['wget', '-O', g_fine_tune_caffemodel, g_fine_tune_caffemodel_url])
