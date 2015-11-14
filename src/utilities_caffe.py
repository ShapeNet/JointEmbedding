#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import math
import shutil
import datetime
import numpy as np
from multiprocessing import Pool
from google.protobuf import text_format

#https://github.com/BVLC/caffe/issues/861#issuecomment-70124809
import matplotlib 
matplotlib.use('Agg')

def _array4d_idx_to_datum_string(array4d_idx):
    import caffe
    array4d = array4d_idx[0]
    idx = array4d_idx[1]
    global_idx = array4d_idx[2]
    array = array4d[idx, :, :, :]
    datum = caffe.io.array_to_datum(array.astype(float), global_idx)
    return datum.SerializeToString()      

def extract_cnn_features(img_filelist, img_root, prototxt, caffemodel, feat_name, output_path=None, output_type=None, caffe_path=None, mean_file=None, gpu_index=0, pool_size=8):   
    if caffe_path:
        sys.path.append(os.path.join(caffe_path, 'python'))
    import caffe
    from caffe.proto import caffe_pb2
    
    imagenet_mean = np.array([104, 117, 123])
    if mean_file:
        imagenet_mean = np.load(mean_file)
        net_parameter = caffe_pb2.NetParameter()
        text_format.Merge(open(prototxt, 'r').read(), net_parameter)
        if len(net_parameter.input_dim) != 0:
            input_shape = net_parameter.input_dim
        else:
            input_shape = net_parameter.input_shape
        imagenet_mean = caffe.io.resize_image(imagenet_mean.transpose((1, 2, 0)), input_shape[2:]).transpose((2, 0, 1))
    
    # INIT NETWORK
    caffe.set_mode_gpu()
    caffe.set_device(gpu_index)
    net = caffe.Classifier(prototxt,caffemodel,
        mean=imagenet_mean,
        raw_scale=255,
        channel_swap=(2, 1, 0))
    
    img_filenames = [os.path.abspath(img_filelist)]
    if img_filelist.endswith('.txt'):
        img_filenames = [img_root+'/'+x.rstrip() for x in open(img_filelist, 'r')]
    N = len(img_filenames)
    
    if output_path != None and os.path.exists(output_path):
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        else:
            os.remove(output_path)
        
    ## BATCH FORWARD 
    batch_size = int(net.blobs['data'].data.shape[0])
    batch_num = int(math.ceil(N/float(batch_size)))
    print 'batch_num:', batch_num

    def compute_feat_array(batch_idx):    
        start_idx = batch_size * batch_idx
        end_idx = min(batch_size * (batch_idx+1), N)
        print datetime.datetime.now().time(), '- batch: ', batch_idx, 'of', batch_num, 'idx range:[', start_idx, end_idx, ']'
    
        input_data = []
        for img_idx in range(start_idx, end_idx):
            im = caffe.io.load_image(img_filenames[img_idx])
            input_data.append(im)
        while len(input_data) < batch_size:
            input_data.append(input_data[0])
        net.predict(input_data, oversample=False)
        feat_array = net.blobs[feat_name].data
        return feat_array
                
    if output_type == None:
        feat_list = []
        for batch_idx in range(batch_num):
            start_idx = batch_size * batch_idx
            end_idx = min(batch_size * (batch_idx+1), N)
            batch_count = end_idx - start_idx
            feat_array = compute_feat_array(batch_idx)
            for n in range(batch_count):
                feat_list.append(feat_array[n, ...])
            return feat_list
    elif output_type == 'txt':
        with open(output_path, 'w') as output_file:
            for batch_idx in range(batch_num):
                start_idx = batch_size * batch_idx
                end_idx = min(batch_size * (batch_idx+1), N)
                batch_count = end_idx - start_idx
                feat_array = compute_feat_array(batch_idx)
                for n in range(batch_count):
                    output_file.write(' '.join([str(x) for x in feat_array[n, ...].flat])+'\n')
    elif output_type == 'lmdb':
        import lmdb
        env = lmdb.open(output_path, map_size=int(1e12))
        pool = Pool(pool_size)
        for batch_idx in range(batch_num):
            feat_array = compute_feat_array(batch_idx)        
            start_idx = batch_size * batch_idx
            end_idx = min(batch_size * (batch_idx+1), N)
            array4d_idx = [(feat_array, idx, idx+start_idx) for idx in range(end_idx-start_idx)]
            datum_strings = pool.map(_array4d_idx_to_datum_string, array4d_idx)
            with env.begin(write=True) as txn:
                for idx in range(end_idx-start_idx):
                    txn.put('{:0>10d}'.format(start_idx+idx), datum_strings[idx])        
        env.close();
                
def stack_caffe_models(prototxt, base_model, top_model, stacked_model, caffe_path=None):
    if caffe_path:
        sys.path.append(os.path.join(caffe_path, 'python'))
    import caffe
    
    net = caffe.Net(prototxt, caffe.TEST)

    print 'Copying trained layers from %s...'%(base_model)
    net.copy_from(base_model)

    print 'Copying trained layers from %s...'%(top_model)
    net.copy_from(top_model)
    
    print 'Saving stacked model to %s...'%(top_model)
    net.save(stacked_model)
