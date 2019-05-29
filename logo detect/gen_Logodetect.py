
#coding=utf-8
import cv2 as cv
import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
datas_placeholder = tf.placeholder(tf.float32,[None,28,28,1],name='datap')
labels_placeholder = tf.placeholder(tf.int32,[None],name='labelp')
dropout_placeholder = tf.placeholder(tf.float32,name='dropoutp')
model_path = "detect_model/image_model"

def devide():
    data = []
    labels = []
    for img in os.listdir("./data/positive3/"):
        img_file = cv.imread(os.path.join("./data/positive3/",img),0)
        resized_image = cv.resize(img_file,
                                  (28, 28), interpolation=cv.INTER_CUBIC)
        #img_file = img_file[:, :, np.newaxis]
        #cv.imwrite(os.path.join("./data/positive3/", img), resized_image)
        resized_image = resized_image[:, :, np.newaxis]
        data.append(resized_image)
        labels.append(1)
    #print(len(data))
    num = 0
    for img in os.listdir("./data/negative3/"):
        img_file = cv.imread(os.path.join("./data/negative3/",img),0)
        #
        resized_image = cv.resize(img_file,
                                  (28, 28), interpolation=cv.INTER_CUBIC)
        #cv.imwrite(os.path.join("./data/negative3/", img), resized_image)
        resized_image = resized_image[:, :, np.newaxis]
        data.append(resized_image)
        labels.append(0)
        num =num+1
    #print(num)
    data = np.array(data) #增加维度，默认axix=0
    print(data.shape)
    labels = np.array(labels)
    data = data/255
    #os.system("pause")
    #print(data.shape,labels.shape)
    #from sklearn.preprocessing import LabelBinarizer
    #le = LabelBinarizer()
    #labels = le.fit_transform(labels) #变成数字表示
    #
    X, testX, y, testY = train_test_split(data, labels, test_size=0.1, stratify=labels, random_state=42)
    # print(len(X),len(testX))
    trainimageindex = np.arange(len(X))
    np.random.shuffle(trainimageindex)
    traindata = X[trainimageindex]
    trainlabel = y[trainimageindex]
    testimageindex = np.arange(len(testX))
    np.random.shuffle(testimageindex)
    testdata = testX[testimageindex]
    testlabel = testY[testimageindex]
    return traindata, testdata, trainlabel, testlabel


#卷积核初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape,mean=0.0,stddev=0.1)
    return tf.Variable(initial)

#偏置初始化
def bias_variable(shape):
    initial = tf.constant(0,1,shape=shape)
    return tf.Variable(initial)

# 卷积层封装
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# 池化层封装
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


def build_cnn():
    #5个卷积核，卷积核大小5×5
    #print(datas_placeholder.shape)
    W_conv1 = weight_variable([5,5,1,6])
    #3前置,卷积核输出
    b_conv1 = bias_variable([6])
    layer1_conv = conv2d(datas_placeholder,W_conv1,b_conv1)
    layer2_pool = maxpool2d(layer1_conv)
    W_conv2 = weight_variable([5,5,6,12])
    b_conv2 = bias_variable([12])
    layer3_conv = conv2d(layer2_pool,W_conv2,b_conv2)
    layer4_pool = maxpool2d(layer3_conv)
    #全连接层
    pool_shape = layer4_pool.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    layer5_flat = tf.reshape(layer4_pool, [-1, nodes])
    #layer5_flat = tf.reshape(layer4_pool,shape=[-1,4*4*10])
    W_fc1 = weight_variable([nodes,196])
    b_fc1 = bias_variable([196])
    h_fc1 = tf.nn.relu(tf.matmul(layer5_flat, W_fc1) + b_fc1)
    #droput网络
    h_fc1_drop = tf.nn.dropout(h_fc1, dropout_placeholder)
    #输出层
    W_fc2 = weight_variable([196,2])
    b_fc2 = bias_variable([2])
    out = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2)
    return out
   # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
    #return y_conv
def get_batch(data,label,batch_size):
    for start_index in range(0,len(data)-batch_size+1,batch_size):
        slice_index = slice(start_index,start_index+batch_size)
        yield data[slice_index],label[slice_index]
if __name__ =='__main__':
    datas,testdata,labels,testlabels = devide()
    #print(datas.shape,labels.shape)
    num_classes =2
    #labels = tf.one_hot(labels,num_classes)
   # print(labels.shape)
    #os.system("pause")
    train = True
    logits = build_cnn()
    logits2 = tf.nn.softmax(logits)
    predicted_labels = tf.arg_max(logits, 1)
    #probability = logits[:,predicted_labels]
    tf.add_to_collection('predict', predicted_labels)
    tf.add_to_collection('pro',logits2)
    #prediction = tf.nn.softmax(logits)
    # 定义损失函数,计算均值
    losses = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.one_hot(labels_placeholder, num_classes))
    mean_loss = tf.reduce_mean(losses)
    # 定义优化函数
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    # 确定优化目标
    train_op = optimizer.minimize(mean_loss)

    # 获得预测正确的结果，获取每一行下标最大值
    # correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    # 求准确率
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()
    if train:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch_size = 202
            for step in range(200):
                for train_data_batch,train_label_batch in get_batch(datas,labels,batch_size):
                    train_feed_dict = {
                        datas_placeholder: train_data_batch,
                        labels_placeholder: train_label_batch,
                        dropout_placeholder: 0.25
                    }
                    _,mean_loss_val = sess.run([train_op,mean_loss],feed_dict=train_feed_dict)
                    if step % 20== 0:
                        print("step = {}\tmean loss = {}".format(step, mean_loss_val))
            saver.save(sess,model_path)

    else:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)
            test_feed_dict = {
                datas_placeholder: testdata,
                labels_placeholder: testlabels,
                dropout_placeholder: 0
            }
            predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
            #print(predicted_labels_val)
            for real_label, predicted_label in zip(testlabels, predicted_labels_val):
                # 将label id转换为label名
                print("{} => {}".format(real_label, predicted_label))







