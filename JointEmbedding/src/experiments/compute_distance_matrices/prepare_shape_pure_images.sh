#!/bin/bash

# 1. Render pure images from 3D shape models
python renderFolder_fixedview.py

# 2. Crop and resize rendered images
python crop_resize_fixedview.py

# 3. Generate image filelist
python gen_filelist_fixedview.py
