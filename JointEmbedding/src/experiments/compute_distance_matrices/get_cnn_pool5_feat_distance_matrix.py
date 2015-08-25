from scipy.spatial.distance import *
import scipy.io as sio
import numpy as np

#mat = sio.loadmat('../exact_match_closest_view_indices_1to100.mat')
#print mat
#closest_view_indicies = mat['closest_view_indices']
#closest_view_indicies = closest_view_indicies.squeeze()
#print closest_view_indicies
#print closest_view_indicies.shape

'''
test_image_embedding = np.load('exact_match_pool5_feat.npy') # 315 * 9216
test_image_embedding = test_image_embedding.squeeze()
print test_image_embedding.shape


pure_image_embedding = np.load('pure_img_pool5_feat.npy') # 677700/5 * 9216
pure_image_embedding = pure_image_embedding.squeeze()
print pure_image_embedding.shape
'''

#
# WIHT VIEW ORACLE
#
'''
D = np.zeros((test_image_embedding.shape[0], 6777))
for k in range(test_image_embedding.shape[0]):
  selected = [(closest_view_indicies[k]-1)+100*x for x in range(6777)]
  test = test_image_embedding[k,:]
  test = test[np.newaxis,:]
  pure = pure_image_embedding[selected,:]
  print test.shape, pure.shape
  d = cdist(test, pure)
  print d.shape
  D[k,:] = d
print D.shape

np.savetxt('CNN_distMatrix_withViewOracle.txt', D)
'''

#
# WITHOUT VIEW ORACLE
#
'''
selected = []
for x in range(6777):
  for y in range(20):
    selected.append(x*100+y+40)
pure_image_embedding = pure_image_embedding[selected,:]
print pure_image_embedding.shape
'''

'''
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

np.save('D.npy',D)
'''
D = np.load('D.npy')

#imageModelDist = np.zeros((test_image_embedding.shape[0], 6777))
imageModelDist = np.zeros((315, 6777))
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

np.savetxt('CNN_pool5_distMatrix_withoutViewOracle.txt', imageModelDist)
