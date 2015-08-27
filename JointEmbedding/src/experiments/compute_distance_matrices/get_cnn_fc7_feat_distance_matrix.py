from scipy.spatial.distance import *
import scipy.io as sio
import numpy as np
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))

test_image_embedding = np.load(os.path.join(BASEDIR, 'exact_match_fc7_feat.npy')) # 315 * 4096
test_image_embedding = test_image_embedding.squeeze()
print test_image_embedding.shape


pure_image_embedding = np.load(os.path.join(BASEDIR, 'pure_img_fc7_feat.npy')) # 677700 * 4096
pure_image_embedding = pure_image_embedding.squeeze()
print pure_image_embedding.shape

#
# WITHOUT VIEW ORACLE
#
selected = []
for x in range(6777):
  for y in range(20):
    selected.append(x*100+y+40)

pure_image_embedding = pure_image_embedding[selected,:]
print pure_image_embedding.shape

D = np.zeros((test_image_embedding.shape[0], 677700/5))
for k in range(test_image_embedding.shape[0]):
  print ">> ", k
  test = test_image_embedding[k,:]
  test = test[np.newaxis,:]
  print test.shape, pure_image_embedding.shape
  d = cdist(test, pure_image_embedding)
  print d.shape
  D[k,:] = d
print D.shape
#np.save('D_fc7.npy',D)

imageModelDist = np.zeros((test_image_embedding.shape[0], 6777))
for k in range(test_image_embedding.shape[0]):
  print ">> ", k
  distToModel = D[k,:] # 1*677700
  print distToModel.shape
  distToModel = distToModel.reshape((20,6777), order='F')
  print distToModel.shape
  distToModel = np.min(distToModel,0)
  print distToModel.shape
  imageModelDist[k,:] = distToModel.squeeze()

np.savetxt(os.path.join(BASEDIR, 'CNN_fc7_distMatrix_withoutViewOracle.txt'), imageModelDist)
