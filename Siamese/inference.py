import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io
import sys

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples                
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe.proto import caffe_pb2 
# iclude lib that's installed locally
sys.path.append('/n/shokuji/dd/cecilia/.local/lib/python2.6/site-packages/')
import lmdb

plt.rcParams['figure.figsize'] = (12, 16)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os

# load pre-trained model and the deploy file
caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'projects/sketchto3D/sketch_test.prototxt',
                caffe_root + 'projects/sketchto3D/caffemodel/sketch__iter_10000.caffemodel',
                caffe.TEST)

from sklearn.neighbors import NearestNeighbors

# utility function: plot multiple images arranging in tiles.
def plot_images(images, tile_shape, cmap = cm.Greys_r):
    assert images.shape[0] <= (tile_shape[0]* tile_shape[1])
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  nrows_ncols = tile_shape ) 
    for i in range(images.shape[0]):
        grd = grid[i]
        grd.imshow(images[i], cmap=cmap)
        
        
# feed test data forward to get the feature vectors

num_data = 54000
feat_s1 = np.zeros((num_data, 64))
feat_s2 = np.zeros((num_data, 64))
feat_v1 = np.zeros((num_data, 64))
feat_v2 = np.zeros((num_data, 64))

sketch1 = np.zeros((num_data, 128, 128))
sketch2 = np.zeros((num_data, 128, 128))

view1 = np.zeros((num_data, 128, 128))
view2 = np.zeros((num_data, 128, 128))

print ("start evaluating")

for i in range (0, num_data):
    out = net.forward()
    sketch1[i, :] = net.blobs['sketch1'].data[0]
    sketch2[i, :] = net.blobs['sketch_pair'].data[0]
    view1[i, :] = net.blobs['view1'].data[0]
    view2[i, :] = net.blobs['view_pair'].data[0]
    
    feat_s1[i, :] = net.blobs['feat'].data[0]
    feat_s2[i, :] = net.blobs['feat_p'].data[0]
    feat_v1[i, :] = net.blobs['feat_v'].data[0]
    feat_v2[i, :] = net.blobs['feat_vp'].data[0]

print ("finish evaluating")

nbrs_s = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(feat_s1)
distances_s, indices_s = nbrs.kneighbors(feat_s2)
np.save('distances_s.npy','distances_s')
np.save('indices_s.npy','indices_s')


nbrs_v = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(feat_v1)
distances_v, indices_v = nbrs.kneighbors(feat_v2)
np.save('distances_v.npy','distances_v')
np.save('indices_v.npy','indices_v')


nbrs_sv = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(feat_v1)
distances_sv, indices_sv = nbrs.kneighbors(feat_s1)
np.save('distances_sv.npy','distances_sv')
np.save('indices_sv.npy','indices_sv')


