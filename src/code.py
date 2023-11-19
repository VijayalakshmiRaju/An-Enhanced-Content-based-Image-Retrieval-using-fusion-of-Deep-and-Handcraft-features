# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:35:10 2023

@author: Pavi
"""

from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from keras.layers import add, Activation,LeakyReLU
from tensorflow_addons.layers import GroupNormalization
#from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.layers import concatenate
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#from sklearn.cluster import KMeans
from keras.utils import to_categorical
import faiss
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from PIL import Image
from matplotlib import pyplot as plt
def conv2d_unit(x, filters, kernels, strides=1):
    x = Conv2D(filters, kernels, padding='same', strides=strides, activation='linear', kernel_regularizer=l2(5e-4))(x)
    x = GroupNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def inception(layer_in, f1, f2_in, f2_out, f3_in, f3_out, last_layer=False):
# 1x1 conv
    conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
# 3x3 conv
    conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
# 5x5 conv
    conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
# 3x3 max pooling
    x = add([layer_in, conv3])
    x = Activation('linear')(x)
    if last_layer==True:
        return conv1,x,conv5
# concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5], axis=-1)
    return layer_out, x
def residual_block(inputs, filters):
    x = conv2d_unit(inputs, filters, (1, 1))
    x = conv2d_unit(x, 2 * filters, (3, 3))
    x = add([inputs, x])
    x = Activation('linear')(x)
    return x
def stack_residual_block(inputs, filters, n):
    x = residual_block(inputs, filters)
    for i in range(n - 1):
        x = residual_block(x, filters)
    return x
def darknet_base(inputs):
    x = conv2d_unit(inputs, 32, (3, 3))
    x = conv2d_unit(x, 64, (3, 3), strides=2)
    #x = stack_residual_block(x, 32, n=1)
    dep_con,x=inception(x,48, 32, 64, 16, 16)
    dep_con = conv2d_unit(dep_con, 64, (1, 1), strides=2)
   
    x = conv2d_unit(x, 128, (3, 3), strides=2)
    x = stack_residual_block(x, 64, n=1)
    dep_con1,x=inception(x,96, 64, 128, 24, 32)
    dep_con1 = conv2d_unit(dep_con1, 128, (1, 1))
    dep_con = concatenate([dep_con,dep_con1], axis=-1)
   
    dep_con = conv2d_unit(dep_con, 192, (1, 1), strides=2)
   
    x = conv2d_unit(x, 256, (3, 3), strides=2)
    x = stack_residual_block(x, 128, n=7)
   
    dep_con1,x=inception(x,192, 128, 256, 32, 64)
    dep_con1 = conv2d_unit(dep_con1, 256, (1, 1))
   
    dep_con = concatenate([dep_con,dep_con1], axis=-1)
   
    dep_con = conv2d_unit(dep_con, 448, (1, 1), strides=2)

    x = conv2d_unit(x, 512, (3, 3), strides=2)
    x = stack_residual_block(x, 256, n=7)
   
    dep_con1,x=inception(x,384, 256, 512, 64, 128)
    dep_con1 = conv2d_unit(dep_con1, 512, (1, 1))
    dep_con = concatenate([dep_con,dep_con1], axis=-1)
   
    dep_con = conv2d_unit(dep_con, 960, (1, 1), strides=2)

    x = conv2d_unit(x, 1024, (3, 3), strides=2)
    x = stack_residual_block(x, 512, n=3)

    conv1,conv3,conv5=inception(x,768, 512, 1024, 128, 256,True)
    dep_con = concatenate([conv1,conv3,conv5,dep_con], axis=-1)
    dep_con1 = conv2d_unit(dep_con, 1024, (1, 1))
    return dep_con1

def darknet():
    inputs = Input(shape=(256, 256, 3))
    x = darknet_base(inputs)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(12, activation='softmax')(x)

    model = Model(inputs, x)

    return model


class dictionary:
    def __init__(self):
        p = Path('Dataset/train/')
        dirs = p.glob('*')
        sift_feature=[]
        self.sift=cv2.SIFT_create()
        for folder_name in dirs:
            label = os.path.basename(str(folder_name))
            for image_path in folder_name.glob('*.jpg'):
                image = cv2.imread(str(image_path))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img1=cv2.resize(gray,(256,256))
                kp,features=self.sift.detectAndCompute(img1,None)
                sift_feature.append(features)
        sift_feature=np.vstack(sift_feature)
        x=sift_feature.shape[1]
        self.kmeans = faiss.Kmeans(x,k=800)
        self.kmeans.train(sift_feature)

    def quantify_image(self,image):
        kp,features=self.sift.detectAndCompute(image,None)
        dist,label=self.kmeans.index.search(features,1)
        return label
    def getHandcraftFeatureVector(self,img_path):
        image = cv2.imread(str(img_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img1=cv2.resize(gray,(256,256))
        dict_feature=self.quantify_image(img1)
        hist,_=np.histogram(dict_feature,bins=np.arange(801))
        hist=hist.reshape(-1)
        hist = hist/np.linalg.norm(hist)
        return hist

class GN_Inception_DarkNet_53:
    def __init__(self):
       
        im_size=256
        images=[]
        labels=[]
        path="Dataset/train"
        dir_list = os.listdir(path)
        for i in dir_list:
            data_path=path+'/'+str(i)
            filenames=[i for i in os.listdir(data_path)]
            for f in filenames:
                img=cv2.imread(data_path+'/'+f)
                img=cv2.resize(img,(im_size,im_size))
                images.append(img)
                labels.append(i)
        path="Dataset/test"
        dir_list = os.listdir(path)
        for i in dir_list:
            data_path=path+'/'+str(i)
            filenames=[i for i in os.listdir(data_path)]
            for f in filenames:
                img=cv2.imread(data_path+'/'+f)
                img=cv2.resize(img,(im_size,im_size))
                images.append(img)
                labels.append(i)
        images=np.array(images)
        images=images.astype('float32')/255.0
        y_labelencoder=LabelEncoder()
        y=y_labelencoder.fit_transform(labels)
        y=np.array(y)
        Y=to_categorical(y)
        y_mappings={index:labels for index,labels in enumerate(y_labelencoder.classes_)}
        images,Y=shuffle(images,Y,random_state=42)
        train_x,test_x,train_y,test_y=train_test_split(images,Y,test_size=0.20,random_state=42)      
        self.DarkNet=darknet()
        METRICS = [
              tf.keras.metrics.BinaryAccuracy(name='accuracy'),
              tf.keras.metrics.Precision(name='precision'),
              tf.keras.metrics.Recall(name='recall'),  
              tf.keras.metrics.AUC(name='auc')
        ]
        self.DarkNet.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='categorical_crossentropy',
            metrics=METRICS
        )        
        self.DarkNet.fit(train_x,train_y,epochs=80,validation_split=0.2)
        Input(shape=(256, 256, 3))
        trim_layer=self.DarkNet.get_layer('avg_pool').output
        self.gndarknet=Model(inputs=self.DarkNet.input,outputs=trim_layer)
    def extract_deep_feature(self,img_path):
        img=Image.open(img_path)
        img = img.resize((256, 256)).convert("RGB")
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        feature_vector = self.gndarkNet.predict(x)
        feature_vector = feature_vector/np.linalg.norm(feature_vector)
        return feature_vector
    def extract_weight(self,img_path):
        img=Image.open(img_path)
        img = img.resize((256, 256)).convert("RGB")
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        feature_vector = self.DarkNet.predict(x)
        return feature_vector

class cbir:
    def __init__(self):
        self.handcraft=dictionary()
        self.deepextract=GN_Inception_DarkNet_53()
    def getFeatureVector(self, img_path):
        deep_feature=self.deepextract.extract_deep_feature(self,img_path)
        dict_feature=self.handcraft.quantify_image(str(img_path))
        feature_vector=np.concatenate((deep_feature,dict_feature),axis=None)
        feature_vector=feature_vector.reshape((1,-1))
        return feature_vector  
    def extract_train_feature(self,path):
        p = Path(path)
        dirs = p.glob('*')
        label_dict = {'burj_khalifa': 0, 'chichen_itza': 1, 'christ_the_reedemer': 2, 'eiffel_tower': 3, 'great_wall_of_china': 4, 'machu_pichu': 5, 'pyramids_of_giza': 6, 'roman_colosseum': 7, 'statue_of_liberty': 8, 'stonehenge': 9, 'taj_mahal': 10, 'venezuela_angel_falls': 11}
        feature_vector = []
        labels = []
        files = []
        self.wondertrain = pd.DataFrame(columns=['feature','file', 'index'])
        for folder_name in dirs:
            label = os.path.basename(str(folder_name))
            for image_path in folder_name.glob('*.jpg'):
                files.append(str(image_path))
                labels.append(label_dict[label])

        self.wondertrain['file']=files
        self.wondertrain['index']=labels
        self.wondertrain['feature'] = self.wondertrain.apply(lambda row: self.getFeatureVector(self.deepextract.gndarknet, row['file']), axis=1)
        return
    def retrieve(self,path):
        test_vector=self.deepextract.getFeatureVector(path)
        prob=self.deepextract.extract_weight(path)
        feature_vector=self.wondertrain['feature'].to_numpy()
        label=self.wondertrain['index'].to_numpy()
        dists1=[]
        for i in range(len(feature_vector)):
            weight=1-prob[0][label[i]]
            dist=weight*np.linalg.norm(feature_vector[i]-test_vector)
        dists1.append(dist)
        ids1=np.argsort(dists1)[:10]
        scores1=[(dists1[id],self.wondertrain['file'].values[id]) for id in ids1]
        return scores1
if __name__ == "__main__":
    extract_class=cbir()
    extract_class.extract_train_feature("Dataset/train/")
    img=input("enter the query img path :")
    retrieved_img=extract_class.retrieve(str(img))
    plt.figure(figsize=(50,50))
    query_img=cv2.imread(img)
    plt.subplot(6,2,1)
    plt.title("Query Image")
    plt.imshow(query_img)
    plt.axis('off')
    for i in range(len(retrieved_img)):
        ret_img=cv2.imread(retrieved_img[i][1])
        plt.subplot(6,2,i+1)
        plt.imshow(ret_img)
        plt.axis('off')
