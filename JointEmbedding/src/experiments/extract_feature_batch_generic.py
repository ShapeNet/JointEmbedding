import numpy as np
import scipy.io as sio
import caffe
import math
import argparse
from skimage.transform import resize
from multiprocessing import Pool
from save_mat_for_lmdb import *
'''
@auhtor: rqi
may 22, 2015
'''
parser = argparse.ArgumentParser(description="Extract neural network features for GENERIC input.")
parser.add_argument('-i', '--input_npy_file', help='Input numpy file, contains an array of N*C*H*W', required=True)
parser.add_argument('-d', '--deploy_file', help='Model deploy file.', required=True)
parser.add_argument('-p', '--params_file', help='Model param file', required=True)
parser.add_argument('-b', '--batch_size', help='Test batch size (must be consistent with deploy file).', required=True)
parser.add_argument('--feat_dim', help='Feature dimension. e.g. 4096', required=True)
parser.add_argument('--feat_name', help='Feature name. e.g. fc7', required=True)
parser.add_argument('--save_file', help='Save file name.', required=True)
parser.add_argument('--save_file_format', help='Save file format (npy, mat or txt). (default=npy)', default='npy')
parser.add_argument('--gpu_index', help='GPU index (default=0).', default=0)
args = parser.parse_args()

gpu_index = int(args.gpu_index)

# INIT NETWORK
net = caffe.Net(args.deploy_file,args.params_file, caffe.TEST)
#net.set_phase_test()
caffe.set_mode_gpu()
caffe.set_device(gpu_index)

first_layer_data = np.load(args.input_npy_file)
N = np.shape(first_layer_data)[0]
print np.shape(first_layer_data)

assert(args.save_file_format == 'mat' or args.save_file_format == 'npy' or args.save_file_format == 'txt')

## BATCH FORWARD 
BATCH_SIZE = int(args.batch_size) # CONSISTENT with deploy.prototxt, do NOT change

all_feats = np.zeros((N,int(args.feat_dim),1,1))
print np.shape(all_feats)
batch_feats = np.zeros((BATCH_SIZE, int(args.feat_dim), 1, 1))

batch_num = int(math.ceil(N/float(BATCH_SIZE)))
print batch_num
for k in range(batch_num):
    start_idx = BATCH_SIZE * k
    end_idx = min(BATCH_SIZE * (k+1), N)
    print 'batch: ', k, 'idx: ', start_idx, end_idx

    input_data = []
    for j in range(start_idx, end_idx):
        input_data.append(first_layer_data[j,:,:,:])
        #print np.shape(first_layer_data[j,:,:,:])

    probs = net.forward_all(data=np.asarray(input_data), blobs=[args.feat_name])
    feats = net.blobs[args.feat_name].data
    feat_shape = np.shape(feats)
    feat_dim = feat_shape[1]# * feat_shape[2] * feat_shape[3]
    print feat_shape, feat_dim
    assert(feat_dim == int(args.feat_dim))
    assert(np.shape(feats)[0] == BATCH_SIZE)
    for j in range(BATCH_SIZE):
      #batch_feats[j,:,:,:] = feats[j,:,:,:].reshape(feat_dim,1,1)
      batch_feats[j,:,:,:] = feats[j,:].reshape(feat_dim,1,1)
    #print np.shape(batch_feats)
    all_feats[start_idx:end_idx,:,:,:] = batch_feats[0:end_idx-start_idx,:,:]

print np.shape(all_feats.squeeze())
try:
  if args.save_file_format == 'txt':
    np.savetxt(args.save_file+'.txt', all_feats.squeeze(), fmt='%.8e', delimiter=' ')
  elif args.save_file_format == 'npy':
    np.save(args.save_file+'.npy', all_feats)
  elif args.save_file_format == 'mat':
    save_mat_for_lmdb(args.save_file, np.array([N,int(args.feat_dim),1,1]), all_feats.squeeze(), 1)
except:
  save_mat_for_lmdb(args.save_file, np.array([N,int(args.feat_dim),1,1]), all_feats.squeeze(), 1)
