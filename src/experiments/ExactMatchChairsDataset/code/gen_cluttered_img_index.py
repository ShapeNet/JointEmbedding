import numpy as np

imgs = [x.rstrip().split('/')[-1] for x in open('../exact_match_chairs_img_filelist.txt','r')]
clutter_imgs = [x.rstrip() for x in open('../exact_match_chairs_cluttered_img_filelist.txt','r')]

clutter_img_index = []

for clutter_img in clutter_imgs:
  clutter_img_index.append(imgs.index(clutter_img))

print clutter_img_index
np.savetxt('exact_match_chairs_cluttered_img_indicies_0to314.txt',clutter_img_index,fmt='%d')
