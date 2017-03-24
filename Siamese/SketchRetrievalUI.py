import sys
from PyQt4 import QtGui, QtCore
import numpy as np
import qimage2ndarray
import scipy.misc
import scipy.ndimage.filters
import scipy.io
import os
from sklearn.neighbors import NearestNeighbors
import caffe
from caffe.proto import caffe_pb2 

project_root = ''

K = 12 # 12 nearest neighbors
sketch_feature = np.load( project_root + 'trained_feature/feat_train_sketch.npy' )
nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(sketch_feature)
caffe.set_mode_cpu()
net = caffe.Net(project_root + 'sketch_test_img.prototxt',
                project_root + 'caffemodel/sketch_all_iter_50000.caffemodel',
                caffe.TEST)

class Retriever() :
    def retrieve(self, image) :
        #   To do: 
        #       Retrieve top 12 model's view path for the input image.
        #       Input image is a 128x128 binary array
        input_img = 1-image.transpose().reshape((1, 1, 128, 128)).astype(float)
        net.blobs['data'].data[...] = input_img
        output = net.forward()
        query_feature = output['feat'][0]
        distances, indices = nbrs.kneighbors(query_feature)
        path_list = []
        for i in range(K) :
            cur_img = indices[0, i]
            cur_img_path = project_root + 'sketchImg/0' + str(cur_img) + '.png'
            path_list.append( cur_img_path )
        return path_list
        
class ModelView(QtGui.QLabel) :
    def __init__(self, parent, num) :
        QtGui.QLabel.__init__(self, parent)
        self.num = num
        self.pixmap = QtGui.QPixmap(128, 128)
        self.loadImage('')
    def loadImage(self, path) :
        if path == '' :
            self.pixmap.fill(QtGui.QColor(0, 0, 0))
        else :
            self.pixmap = QtGui.QPixmap(path)
        qp = QtGui.QPainter(self.pixmap)
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor(255, 255, 255))
        pen.setWidth(2)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        qp.setPen(pen)
        qp.drawText(10, 20, str(self.num))
        self.setPixmap(self.pixmap)
        
class SketchCanvas(QtGui.QLabel) :
    def __init__(self, parent, w, h, pw, ew) :
        QtGui.QLabel.__init__(self, parent)
        self.pixmap = QtGui.QPixmap(w, h)
        self.setPixmap(self.pixmap)
        self.last_x = 0
        self.last_y = 0
        self.w = w
        self.h = h
        self.pw = pw
        self.ew = ew
        self.clearCanvas()
        self.mode = 0
    def clearCanvas(self) :
        self.pixmap.fill(QtGui.QColor(0, 0, 0))
        self.setPixmap(self.pixmap)
    def mousePressEvent(self, e) :
        self.last_x = e.x()
        self.last_y = e.y()
        if e.button() == QtCore.Qt.LeftButton :
            self.mode = 0
        if e.button() == QtCore.Qt.RightButton :
            self.mode = 1
    def mouseMoveEvent(self, e):
        qp = QtGui.QPainter(self.pixmap)
        pen = QtGui.QPen()
        if self.mode == 0 :
            pen.setColor(QtGui.QColor(255, 255, 255))
            pen.setWidth(self.pw)
        else :
            pen.setColor(QtGui.QColor(0, 0, 0))
            pen.setWidth(self.ew)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        qp.setPen(pen)
        qp.drawLine(self.last_x, self.last_y, e.x(), e.y())
        self.last_x = e.x()
        self.last_y = e.y()
        self.setPixmap(self.pixmap)

class SketchRetrivalUI(QtGui.QFrame) :
    def __init__(self) :
        QtGui.QWidget.__init__(self)
        bw = 2        
        h = 128*6+bw*7
        w = 128*2+bw*2+h
        self.move(-800, -800)
        self.resize(w, h)
        style_str = 'border:'+str(bw)+'px solid rgb(0, 255, 0)'
        self.canvas = SketchCanvas(self, h-bw*2, h-bw*2, 4, 32)
        self.canvas.setStyleSheet(style_str)
        self.views = []
        for i in range(12) :
            ix = i % 2
            iy = i / 2
            self.views.append(ModelView(self, i+1))
            self.views[-1].move(h-bw+(128+bw)*ix, (128+bw)*iy)
            self.views[-1].setStyleSheet(style_str)
        self.retriever = Retriever()
        
    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape :
            self.close()   
        if e.key() == QtCore.Qt.Key_Space :
            self.canvas.clearCanvas()
        
    def mouseReleaseEvent(self, e) :
        self.retrieve()
            
    def retrieve(self) :
        image = self.canvas.pixmap.toImage()
        image = np.array(qimage2ndarray.rgb_view(image))
        image = image[:, :, 0]
        image = scipy.ndimage.filters.gaussian_filter(image, 2)
        image = scipy.misc.imresize(image, (128, 128))  
        image = image > 50.0
        path_list = self.retriever.retrieve(image)
        for i in range(12) :
            self.views[i].loadImage(path_list[i])
        
a = QtGui.QApplication(sys.argv)
w = SketchRetrivalUI()
w.setWindowTitle("SketchRetrievalGUI") 
w.show() 
 
sys.exit(a.exec_())
