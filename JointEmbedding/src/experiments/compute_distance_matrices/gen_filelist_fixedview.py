import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(BASE_DIR)))
from global_variables import *

img_root_dir = g_fixedview_image_folder
modelIds = [x.rstrip().split(' ')[0] for x in open(g_shape_list_file,'r')]
elevatio_degs = g_fixedview_elevation_degs
azimuth_degs = g_fixedview_azimuth_degs

pure_img_filelist = []
sampled_pure_img_filelist = [] # with elevation fixed
for k in range(len(modelIds)):
  print k
  modelId = modelIds[k]
  model_img_dir = os.path.join(img_root_dir, modelId)
  for e in elevatio_degs:
    for a in azimuth_degs:
      img_filename = 'image_a%03d_e%03d_t000_d003.png' % (a,e)
      img_path = os.path.join(model_img_dir, img_filename)
      #assert(os.path.isfile(img_path))
      pure_img_filelist.append(img_path)
      if e == g_fixedview_elevation_sample_deg:
          sampled_pure_img_filelist.append(img_path)

print len(pure_img_filelist)

fout = open(os.path.join(BASE_DIR, 'pure_img_filelist.txt'), 'w')
for name in pure_img_filelist:
  fout.write(name+'\n')
fout.close()

fout = open(os.path.join(BASE_DIR, 'sampled_pure_img_filelist.txt'), 'w')
for name in sampled_pure_img_filelist:
  fout.write(name+'\n')
fout.close()
