#coding=utf-8
import os
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow as tf

class ImportGraph():
    def __init__(self, loc):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc+'.meta',clear_devices=True)
            saver.restore(self.sess, loc)
            self.classpro = tf.get_collection('pro')[0]
            self.pred = tf.get_collection('predict')[0]
    def get_classpro(self,data):
        return self.sess.run(self.classpro,feed_dict={"datap:0":data})
    def get_pred(self,data):
        return self.sess.run(self.pred, feed_dict={"datap:0": data})
def test(image,*args):
    print(image.shape)
    cv.imshow("test",image)
    print(args)
    cv.waitKey(0)
    os.system('pause')
def image_resize(img,scaleFactor):
    return cv.resize(img,(int(img.shape[1]*(1/scaleFactor)),int(img.shape[0]*(1/scaleFactor))),interpolation=cv.INTER_CUBIC)
def isLogo(image):
    model_path = "detect_model/image_model"
    model1 = ImportGraph(model_path)
    pred = model1.get_pred(image)
    classproget = model1.get_classpro(image)
    return pred,classproget
def get_image():
    data = []
    origin = []
    for img in os.listdir("./testdata"):
        img_file1 = cv.imread(os.path.join('./testdata',img))
        img_file1 = cv.resize(img_file1, (400, 300), interpolation=cv.INTER_CUBIC)
        img_file = cv.cvtColor(img_file1,cv.COLOR_BGR2GRAY)
        img_file = cv.resize(img_file, (400, 300), interpolation=cv.INTER_CUBIC)
       # print('file{}'.format(img_file.shape))
        img_file = img_file[:,:,np.newaxis]
        data.append(img_file)
        origin.append(img_file1)
    data = np.array(data)
    data = data/255
    return data,origin
def pyramid(image,scale=1.2,minsize=(30,45)):
    yield image #执行到yield就返回一个图像，调用不会执行，返回一个iterable对象，调用next才会开始执行
    while True:
        image = image_resize(image,scale)
        if image.shape[0]<minsize[1] or image.shape[1]<minsize[0]:
            break
        yield image
def sliding_window(image,stepsize,windowsize): #其他以x,y形式存储
    for y in range(0,image.shape[0],stepsize):
        for x in range(0,image.shape[1],stepsize):
            if y+windowsize[1]>image.shape[0] or x+windowsize[0]>image.shape[1]:
                continue
            yield (x,y,image[y:y+windowsize[1],x:x+windowsize[0]])
def get_box(image):
    all_box = []
    #image = cv.resize(image, (200, 280), interpolation=cv.INTER_CUBIC)
    y = image.shape[0]
    x = image.shape[1]
    for img_file in pyramid(image): #先把所有的窗口检测出来，然后再去识别
        patch = []
        box = []
        now_x = img_file.shape[1]
        now_y = img_file.shape[0] #行和列区别
        #print(y,x,now_y,now_x)
        for img_cut in sliding_window(img_file,35,(33,35)):
            cutedimage = img_cut[2]
            sub_box = []
            sub_box.append(img_cut[1])
            sub_box.append(img_cut[1] + 35)
            sub_box.append(img_cut[0])
            sub_box.append(img_cut[0]+33)
            box.append(sub_box)
            cutedimage = cv.resize(cutedimage, (28, 28), interpolation=cv.INTER_CUBIC)
            cutedimage = cutedimage[:, :, np.newaxis]
            patch.append(cutedimage)
        patch = np.array(patch)
        if patch.shape[0]!=0:
            predict,proba = isLogo(patch)
        for predict1,proba1,box1 in zip(predict,proba,box):
            if predict1==0:
                continue
            if proba1[1]<0.95:
                continue
            box1[0] = int(box1[0]*(y/now_y))
            box1[1] = int(box1[1] * (y / now_y))
            box1[2] = int(box1[2] * (x / now_x))
            box1[3] = int(box1[3] * (x / now_x))
            box1.append(proba1[1])
            all_box.append(box1)
    all_box = np.array(all_box)
    print('all_box {}'.format(all_box.shape))
    return all_box
def del_Roi(box,overlapThresh):

    if box.shape[0] == 0:
        return []
    y1 = box[:,0]
    y2 = box[:,1]
    x1 = box[:,2]
    x2 = box[:,3]
    scores = box[:,4]
    area = (x2-x1+1)*(y2-y1+1)
    #print(area)
    idxs = np.argsort(scores) #反向,从小到大返回下标
    #print(idxs)
    pick = []
    while(len(idxs)>0):
        last = len(idxs)-1
        i = idxs[last]
        pick.append(i)
        xx1 = np.minimum(x1[i],x1[idxs[:last]])
        yy1 = np.minimum(y1[i],y1[idxs[:last]])
        xx2 = np.maximum(x2[i],x2[idxs[:last]])
        yy2 = np.maximum(y2[i],y2[idxs[:last]])
        w = np.maximum(0,xx2-xx1+1)
        h = np.maximum(0,yy2-yy1+1)
        overlap = (w*h)/area[idxs[:last]]
        idxs = np.delete(idxs,np.concatenate(([last],np.where(overlap>overlapThresh)[0]))) #得到第一个数组，返回下标
    return box[pick]
def draw(image,box,originimage):
    if box.shape[0]==0:
        return []
    model2 = ImportGraph('model/image_modelreco')
    ha = {
        0:'Adidas' ,
        1:'Apple',
        2:'BMW',
        3:'Citroen',
        4:'Nike'
    }
    for sub_box in box:
        x1 = int(sub_box[2])
        y1 =  int(sub_box[0])
        x2 = int(sub_box[3])
        y2 = int(sub_box[1])
        image2 = image[y1:y2,x1:x2]
        image2 = cv.resize(image2, (28, 28), interpolation=cv.INTER_CUBIC)
        image2 = image2[:, :, np.newaxis]
        all_image = []
        all_image.append(image2)
        all_image = np.array(all_image)
        pred = model2.get_pred(all_image)
        print(ha[pred[0]])
        print(x1,y1,x2,y2)
        cv.rectangle(originimage, (x1,y1), (x2,y2),  (100, 100, 255), 1)
        xx = x1
        yy = y1
        if x1==0:
            xx = x1+30
        if y1==0:
            yy = y1+30
        cv.putText(originimage,ha[pred[0]],(xx,yy),cv.FONT_HERSHEY_SIMPLEX,1,(100,100,255),2)
        #print(x1,y1,x2,y2,image.shape)
    cv.namedWindow("test",0)
    cv.imshow("test",originimage)
    cv.waitKey(0)
if __name__=='__main__':
    data,origin = get_image() #(?,28,28,1)
    for image,originimage in zip(data,origin):
        #cv.imshow("what",image)
        #cv.waitKey(0)
        box = get_box(image)
        box = del_Roi(box,0.2)
        print(box.shape)
        draw(image,box,originimage)
