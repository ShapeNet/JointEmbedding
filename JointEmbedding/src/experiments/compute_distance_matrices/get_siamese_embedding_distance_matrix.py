from scipy.spatial.distance import *
import scipy.io as sio
import numpy as np
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))

test_image_embedding = np.load(os.path.join(BASEDIR,'exact_match_chairs_siamese_embedding.npy')) # 315 * 512
test_image_embedding = test_image_embedding.squeeze()
print test_image_embedding.shape


pure_image_embedding = np.load(os.path.join(BASEDIR, 'pure_img_siamese_embedding.npy')) # 677700/5 * 512
pure_image_embedding = pure_image_embedding.squeeze()
print pure_image_embedding.shape


#
# WITHOUT VIEW ORACLE
#
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
#np.save('D.npy',D)

imageModelDist = np.zeros((test_image_embedding.shape[0], 6777))
for k in range(315):#test_image_embedding.shape[0]):
  print ">> ", k
  distToModel = D[k,:] # 1*677700
  print distToModel.shape
  #distToModel = distToModel.reshape((20,6777)) # WRONG!
  distToModel = distToModel.reshape((20,6777), order='F') # WRONG!
  print distToModel.shape
  distToModel = np.min(distToModel,0)
  print distToModel.shape
  imageModelDist[k,:] = distToModel.squeeze()

np.savetxt(os.path.join(BASEDIR, 'siamese_distMatrix_withoutViewOracle.txt'), imageModelDist)
