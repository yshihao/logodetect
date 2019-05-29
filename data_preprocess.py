#coding=utf-8
import cv2 as cv
import tensorflow as tf
import numpy as np
import os
import pandas as pd
df = pd.read_csv("./flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt",header=None,sep=" ")
df.drop(df.columns[-1],axis=1,inplace=True)
df.columns = ["name", "labels", "subset", "x1", "y1", "x2", "y2"]
df.head()
#print(df)
# df = df.sample(frac=1).reset_index(drop=True) #重置索引
def image_resize(img,scaleFactor):
    return cv.resize(img,(int(img.shape[1]*(1/scaleFactor)),int(img.shape[0]*(1/scaleFactor))),interpolation=cv.INTER_CUBIC)
def pyramid(image,scale=1.1,minsize=(25,25)):
    yield image #执行到yield就返回一个图像，调用不会执行，返回一个iterable对象，调用next才会开始执行
    while True:
        image = image_resize(image,scale)
        if image.shape[0]<minsize[1] or image.shape[1]<minsize[0]:
            break;
        yield image
def crop_img(id):
    image = os.path.join("./flickr_logos_27_dataset/flickr_logos_27_dataset_images/",df.loc[id]["name"])
    image = cv.imread(image)
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    re_image = gray_image[df["y1"].iloc[id]:df["y2"].iloc[id],df["x1"].iloc[id]:df["x2"].iloc[id]]
    #cv.imshow("test",image)
    #cv.waitKey(0)
    return gray_image,re_image
def sliding_window(image,stepsize,windowsize): #其他以x,y形式存储
    for y in range(0,image.shape[0],stepsize):
        for x in range(0,image.shape[1],stepsize):
            yield (x,y,image[y:y+windowsize[1],x:x+windowsize[0]])
def read_data(id):
    df = pd.read_csv("./flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt",header=None,sep=" ")
    df.drop(df.columns[-1],axis=1,inplace=True) #沿着行执行
    df.columns = ["name", "labels", "subset", "x1", "y1", "x2", "y2"]
    df.head()
    #print(df)
   # df = df.sample(frac=1).reset_index(drop=True) #重置索引
    df = df.reset_index(drop=True)
    #print(df.loc[id]['x1'],df['x1'].iloc[id])
   # print(df)
    image = os.path.join('./flickr_logos_27_dataset/flickr_logos_27_dataset_images',df.loc[id]['name'])
    image1 = cv.imread(image)
    #cv.imshow('test',image1)
    #cv.waitKey(0)
    cv.rectangle(image1,(df['x1'].iloc[id],df['y1'].iloc[id]),(df['x2'].iloc[id],df['y2'].iloc[id]),(0,255,0),1)
    cv.imshow('test2', image1)
    cv.waitKey(0)
def test(x1,y1,x2,y2,nx1,ny1,nx2,ny2):
    length = x2-x1+1
    width = y2-y1+1
    nlength = nx2-nx1+1
    nwidth = ny2-ny1+1
    left = min(x1,nx1)
    right = max(x2,nx2)
    upper = min(y1,ny1)
    down = max(y2,ny2)
    if right-left>=length+nlength or down-upper >=width+nwidth:
        return 0
    inter_length = length+nlength-(right-left)
    inter_width = width+nwidth-(down-upper)
    inter_area = inter_length*inter_width
    origin_area = length*width
    box_area = nlength*nwidth
    if inter_area/origin_area <0.85:
        return 0
    if box_area /origin_area >3:
        return 0
    return 1
def test2(x1,y1,x2,y2,nx1,ny1,nx2,ny2):
    length = x2-x1
    width = y2-y1
    nlength = nx2-nx1
    nwidth = ny2-ny1
    left = min(x1,nx1)
    right = max(x2,nx2)
    upper = min(y1,ny1)
    down = max(y2,ny2)
    if right-left>=length+nlength or down-upper >=width+nwidth:
        return 0
    inter_length = length+nlength-(right-left)
    inter_width = width+nwidth-(down-upper)
    inter_area = inter_length*inter_width
    origin_area = length*width
    box_area = nlength*nwidth
    if inter_area/origin_area >0.1:
        return 0
    return 1
def pre_process():
    u = [i for i in range(842)]
    v = [i for i in range(2443,2596)]
    h =np.concatenate((u,v))
    snum = 0
    num = 0
    for id in u:
        origin_image,image2 = crop_img(id)
        y1 = df["y1"].iloc[id]
        y2 =df["y2"].iloc[id]
        x1 = df["x1"].iloc[id]
        x2 = df["x2"].iloc[id]
        w = origin_image.shape[0]

        if snum>1200:
            break
        for simage in pyramid(origin_image):
            #if snum > 1000:
                #break
            w1 = simage.shape[0]
            y11 =int(y1*(w1/w))
            y21 =int(y2*(w1/w))
            x11 =int(x1*(w1/w))
            x21 = int(x2*(w1/w))
            for sub_image in sliding_window(simage,30,(28,28)):
                if snum > 1200:
                    break
                ny1 = sub_image[1]
                nx1 = sub_image[0]
                ny2 = sub_image[1]+28
                nx2 = sub_image[0]+28
                ans = test(x11,y11,x21,y21,nx1,ny1,nx2,ny2)
                ans2 = test2(x11,y11,x21,y21,nx1,ny1,nx2,ny2)
                if ans==1 and ans2==1:
                    print('what')
                if ans==1:

                    num = num +1
                    #image_name = str(id)+str(snum) + "_" + df.iloc[id]["labels"] + ".jpg"
                    #cv.imwrite(os.path.join("./data/positive2/", image_name), sub_image[2])
                    #print(id)
                    #cv.imshow('test2', sub_image[2])
                    #cv.waitKey(0)

                if ans2==1 :
                    if num>0:
                        snum = snum+1
                        num = num-1
                        image_name = str(id) + str(snum) + "_" + df.iloc[id]["labels"] + ".jpg"
                        #cv.imwrite(os.path.join("./data/negative2/", image_name), sub_image[2])

            #os.system('pause')

        #cv.imshow('teset',origin_image)
        '''
        try:
            resized_image = cv.resize(image2,
                                       (28, 28), interpolation=cv.INTER_CUBIC)
        except:
            print(id)
            continue
        #cv.imshow('test2', resized_image)
        #cv.waitKey(0)
        #name =  df.iloc[id]["labels"]
        #if not name=='Apple' or name=='Adidas' or name=='BMW' or name=='Citroen':
        #   print(id)
        image_name = str(id) + "_" + df.iloc[id]["labels"] + ".jpg"
        cv.imwrite(os.path.join("./data/positive/", image_name), resized_image)
        '''
    print(snum)
def pre_process2(): #生产负样本
    u = [i for i in range(842)]
    v = [i for i in range(2443, 2596)]
    h = np.concatenate((u, v))

    train_data_index = [i for i in range(842,2442)]
    train_data_index2 = [i for i in range(2596,4535)]
    train_data_indexa = np.concatenate((train_data_index,train_data_index2))
    np.random.shuffle(train_data_indexa)
    np.random.shuffle(h)
    #train_data_indexa = train_data_indexa[:1000]
    num = 0
    for i in h:
        if num > 1000:
            break;
        origin,_ = crop_img(i)
        y = origin.shape[0]
        x = origin.shape[1]
        x1 = df['x1'].iloc[i]
        x2 = df['x2'].iloc[i]
        y1 = df['y1'].iloc[i]
        y2 = df['y2'].iloc[i]
        if y1 > y - y2:
            y3 = 0
            y4 = y1
            # y4=int((y1+y2)/2)
        else:
            y3 = y2
            y4 = y
        if x1 > x - x2:
            x3 = 0
            x4 = x1
            # x4 = int((x1+x2)/2)
        else:
            x3 = x2
            x4 = x
        if y4 - y3 < 20 or x4 - x3 < 20:
            continue;
        gray_image = origin[y3:y4, x3:x4]
        resized_image = cv.resize(gray_image,
                                  (28, 28), interpolation=cv.INTER_CUBIC)
        # cv.rectangle(gray_image, (0,0), (df['x1'].iloc[i], df['y2'].iloc[i]),(0, 255, 0), 1)
        cv.imwrite(os.path.join("./data/negative2/", 'negative_' + str(i) + '.jpg'), resized_image)
        num = num + 1
        #cv.imshow('test2', resized_image)
        #cv.waitKey(0)
    for i in train_data_indexa:
        if num>1000:
            break;
        gray_image,_ = crop_img(i)
        y=gray_image.shape[0]
        x=gray_image.shape[1]
        x1 = df['x1'].iloc[i]
        x2 = df['x2'].iloc[i]
        y1 = df['y1'].iloc[i]
        y2 = df['y2'].iloc[i]
        if y1>y-y2:
            y3=0
            y4 = y1
            #y4=int((y1+y2)/2)
        else:
            y3=y2
            y4=y
        if x1>x-x2:
            x3=0
            x4 = x1
            #x4 = int((x1+x2)/2)
        else:
            x3 = x2
            x4 = x
        if y4-y3<15 or x4-x3<15:
            continue;
        gray_image = gray_image[y3:y4,x3:x4]
        resized_image = cv.resize(gray_image,
                                  (28, 28), interpolation=cv.INTER_CUBIC)
        #cv.rectangle(gray_image, (0,0), (df['x1'].iloc[i], df['y2'].iloc[i]),(0, 255, 0), 1)
        cv.imwrite(os.path.join("./data/negative2/", 'negative2_'+str(i)+'.jpg'), resized_image)
        num = num+1
        #cv.imshow('test2', resized_image)
        #cv.waitKey(0)
if __name__=='__main__':
    pre_process()
