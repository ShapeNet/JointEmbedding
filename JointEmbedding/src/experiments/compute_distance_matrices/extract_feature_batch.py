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
'''
parser = argparse.ArgumentParser(description="Extract neural network features for IMAGE input.")
parser.add_argument('-i', '--img_filelist', help='Image file list.', required=True)
parser.add_argument('--img_root', help='Image file root dir.', default='')
parser.add_argument('-d', '--deploy_file', help='Model deploy file.', required=True)
parser.add_argument('-p', '--params_file', help='Model param file', required=False)
parser.add_argument('-b', '--batch_size', help='Test batch size (must be consistent with deploy file).', required=True)
parser.add_argument('--feat_dim', help='Feature dimension. e.g. 4096', required=True)
parser.add_argument('--feat_name', help='Feature name. e.g. fc7', required=True)
parser.add_argument('--save_file', help='Save file name.', required=True)
parser.add_argument('--save_file_format', help='Save file format (npy, mat or txt). (default=npy)', default='npy')
parser.add_argument('--gpu_index', help='GPU index (default=0).', default=0)
parser.add_argument('--resize_dim', help='Resize img to N*N, default not resize.', default=0)
args = parser.parse_args()

gpu_index = int(args.gpu_index)
resize_dim = int(args.resize_dim)
#input_file = '/orions3-zfs/projects/haosu/Image2Scene/data/pretrained_models/data_chair_7k_views_hog_subset.txt'
#fin = open(input_file, 'r')
#[N,C,H,W] = [int(x) for x in fin.readline().rstrip().split(' ')]
# DEPLOY FILE
#model_deploy_file = '/orions3-zfs/projects/haosu/Image2Scene/data/model_chair_view_prediction_7k_siamese/hog_dim_reduction/deploy.prototxt'
#model_params_file = '/orions3-zfs/projects/haosu/Image2Scene/data/model_chair_view_prediction_7k_siamese/hog_dim_reduction/snapshots_iter_35000.caffemodel'

# INIT NETWORK
net = caffe.Classifier(args.deploy_file,args.params_file)
net.set_phase_test()
net.set_mode_gpu()
net.set_mean('data', np.load('/orions3-zfs/projects/rqi/Data/deploy/imagenet_mean.npy'))
net.set_raw_scale('data', 255) # absolutely necessary
net.set_channel_swap('data', (2,1,0)) #not necessary for gray imgs
net.set_device(gpu_index)

img_filenames = [args.img_root+'/'+x.rstrip() for x in open(args.img_filelist, 'r')]
N = len(img_filenames)

assert(args.save_file_format == 'mat' or args.save_file_format == 'npy' or args.save_file_format == 'txt')
#if args.save_file_format == 'mat':
#  assert(N * int(args.feat_dim) <= 200000000)

## BATCH FORWARD 

BATCH_SIZE = int(args.batch_size) # CONSISTENT with deploy.prototxt, do NOT change

def load_and_resize_img(img_filename):
  im = caffe.io.load_image(img_filename)
  if resize_dim > 0:
    im = resize(im, (resize_dim, resize_dim))
  return im

all_feats = np.zeros((N,int(args.feat_dim),1,1))
print np.shape(all_feats)
batch_feats = np.zeros((BATCH_SIZE, int(args.feat_dim), 1, 1))

batch_num = int(math.ceil(N/float(BATCH_SIZE)))
print batch_num
for k in range(batch_num):
    start_idx = BATCH_SIZE * k
    end_idx = min(BATCH_SIZE * (k+1), N)
    print 'batch: ', k, 'idx: ', start_idx, end_idx

    # prepare batch input data
    #p = Pool()
    #input_data = p.map(load_and_resize_img, [img_filenames[j] for j in range(start_idx, end_idx)])
    #p.close()
    #p.join()
    input_data = []
    for j in range(start_idx, end_idx):
        im = caffe.io.load_image(img_filenames[j])
        if resize_dim > 0 and np.shape(im)[0] != resize_dim:
          #print 'resize'
          im = resize(im, (resize_dim, resize_dim))
          assert(np.shape(im)[0] == resize_dim)
        input_data.append(im)

    probs = net.predict(input_data, oversample=False)
    feats = net.blobs[args.feat_name].data
    feat_shape = np.shape(feats)
    feat_dim = feat_shape[1] * feat_shape[2] * feat_shape[3]
    print feat_shape, feat_dim
    assert(feat_dim == int(args.feat_dim))
    assert(np.shape(feats)[0] == BATCH_SIZE)
    for j in range(BATCH_SIZE):
      batch_feats[j,:,:,:] = feats[j,:,:,:].reshape(feat_dim,1,1)
    #print np.shape(batch_feats)
    all_feats[start_idx:end_idx,:,:,:] = batch_feats[0:end_idx-start_idx,:,:]

# TODO: the header may have some problem! sometimes the last mat file has header'N as 0
# SAVE EXTRACTED FEATS
def save_mat_for_lmdb(filename_prefix, header_all, data_all, suffix):
  batch_element_num = 200000000;
  data = data_all;
  header = header_all;
  if np.size(data_all) > batch_element_num:
     batch_N = batch_element_num / np.size(data_all,1) # batch/feat_dim
     data = data[0:batch_N,:]
     header[0] = batch_N
     sio.savemat(filename_prefix+str(suffix), {'header':header, 'data':data})
     header = header_all
     header[0] = header_all[0] - batch_N
     data = data_all[batch_N:,:]
     save_mat_for_lmdb(filename_prefix, header, data, suffix+1);
  else:
     sio.savemat(filename_prefix+str(suffix), {'header':header, 'data':data})
 
print np.shape(all_feats.squeeze())
try:
  if args.save_file_format == 'txt':
    np.savetxt(args.save_file+'.txt', all_feats.squeeze(), fmt='%.8e', delimiter=' ')
  elif args.save_file_format == 'npy':
    np.save(args.save_file+'.npy', all_feats)
  elif args.save_file_format == 'mat':
    save_mat_for_lmdb(args.save_file, np.array([N,int(args.feat_dim),1,1]), all_feats.squeeze(), 1)
    #sio.savemat(args.save_file+'.mat', {'header':np.array([N, int(args.feat_dim), 1, 1]), 'data':all_feats})
except:
  save_mat_for_lmdb(args.save_file, np.array([N,int(args.feat_dim),1,1]), all_feats.squeeze(), 1)
#save_mat_for_lmdb(args.save_file, np.array([N,int(args.feat_dim),1,1]), all_feats.squeeze(), 1)
