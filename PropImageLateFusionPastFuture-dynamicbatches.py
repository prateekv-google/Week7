#!/usr/bin/env python
# coding: utf-8



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import concatenate
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50,VGG16
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import IPython.display as display
import datetime, os
from Transformer import *
import time
from keras import backend as K
# Create the TensorBoard callback,
# which we will drive manually
modelname='PropImageLateFusion_FriWeek7'


# In[2]:


def batch_merge(x):
    return tf.reshape(x,[-1,150,150,3])

def batch_unmerge(x):
    return tf.reshape(x,[-1,10,128])
'''
def batch_unmerge(x):
    x=K.reshape(x,[-1,10,128])
    return x
# In[3]:

'''
print("\nAdding positional encodings to the Inputs before passing them to Transformers")
class TokenAndPositionEmbedding1(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding1, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    def call(self, x):
        maxlen=10
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


# In[4]:


def createVGGWithImgTransformer(maxlen=10,img_embed_dim=128,img_num_heads=8,img_ff_dim=32):
    inpImg=Input(shape=(maxlen,150,150,3))
    model=tf.keras.Sequential()
    model.add(inpImg)

    inpImgR=Lambda(batch_merge)
    model.add(inpImgR)
    
    model_resnet = VGG16(include_top=False, weights=None, input_shape=(150,150,3))
    model.add(model_resnet)
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    inpEmbR=Lambda(batch_unmerge)
    model.add(inpEmbR)
    return model

def createPropTransformer(maxlen=10,prop_embed_dim=32,prop_num_heads=32,prop_ff_dim=32):
    inpProp=keras.Input(batch_shape=(None,maxlen,2))
    model=tf.keras.Sequential()
    inpPropop = inpProp
    model.add(inpPropop)
    model.add(Dense(prop_embed_dim))
    embedding_layer = TokenAndPositionEmbedding1(maxlen,prop_embed_dim)
    model.add(embedding_layer)
    transformer_prop1 = TransformerBlock(prop_embed_dim, prop_num_heads, prop_ff_dim)
    #transformer_prop2 = TransformerBlock(prop_embed_dim, prop_num_heads, prop_ff_dim)
    model.add(transformer_prop1)
    #model.add(transformer_prop2)    
    print(model.summary())
    return model

def getModelForCompile():
    modelImg=createVGGWithImgTransformer()
    modelProp=createPropTransformer()
    combinedInput = concatenate([modelImg.output, modelProp.output])
    x = Dense(64, activation='relu')(combinedInput)
    x = Dense(64, activation='relu')(x)
    x=Dense(2)(x)
    ImgPropMergeModel = Model(inputs=[modelImg.input, modelProp.input], outputs=x)
    print(ImgPropMergeModel.summary())
    return ImgPropMergeModel

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    ImgPropMergeModel = getModelForCompile()
    loss_object = tf.keras.losses.MeanAbsoluteError()
    coptimizer = tf.keras.optimizers.Adam(lr=1e-4)
    ImgPropMergeModel.compile(optimizer=coptimizer,loss=loss_object)
    print("Done....phew")

hf=h5py.File('/shared/data/PropImagePastFutureFivePercent.h5','r')
hfv=h5py.File('/shared/data/PropImagePastFutureFivePercent-Val.h5','r')

print(len(hf.keys())/3)
print(len(hfv.keys())/3)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '/shared/logs/' + modelname+current_time + '/train'
test_log_dir = '/shared/logs/' + modelname+current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


ImgPropMergeModel.save("/shared/Models/"+modelname)


EPOCHS=100
N=3000
bestTrainLoss=10000
bestIndex=0
bestTestLoss=10000
bestTIndex=0
valct=0
ValLoss=[]
TrainLoss=[]
BB=64
strt_time=0
for n in range(0,N*EPOCHS):
    currentn=np.mod(n,N)
    if np.mod(n,30)==0:
        display.clear_output(wait=True)
        print("------------------\n")
        print(n*1.0/(N))
        print("Best Train Loss")
        print(bestTrainLoss)
        print("At")
        print(bestIndex)
        print("Best Test Loss")
        print(bestTestLoss)
        print("At")
        print(bestTIndex)

    cImg=np.array(hf['Ximg_'+str(currentn)])
    cProp=np.array(hf['Xprop_'+str(currentn)])
    cTarget=np.array(hf['Ytarget_'+str(currentn)])
    ImgPropMergeModel.train_on_batch([cImg,cProp],cTarget)

    if np.mod(n,30)==0:
        #ImgPropMergeModel.save("/shared/Models/distributeModel")
        predictions = ImgPropMergeModel([cImg, cProp], training=False)
        currentloss = loss_object(cTarget, predictions)
        currentloss = currentloss.numpy()
        print("Current loss is")
        print(currentloss)
        TrainLoss.append(currentloss)
        if currentloss<bestTrainLoss:
            bestTrainLoss=currentloss
            bestIndex=n*1.0/(N)
            #ImgPropMergeModel.save("/shared/Models/DasTakaTrain")
        
        with train_summary_writer.as_default():
            tf.summary.scalar('trainloss', currentloss, step=n)
            tf.summary.scalar('global_time_step',time.process_time()-strt_time,step=n)
        
        strt_time = time.process_time()
    
    #Compute the validation loss per sample
     
    if np.mod(n,N)==0 and n>0:
        np.savez('logs.npz',ValLoss=ValLoss,TrainLoss=TrainLoss)
        testlossmat=np.zeros((300,))
        for k in range(0,300):
            display.clear_output(wait=True)
            print(k)
            cImg=np.array(hfv['Ximg_'+str(k)])
            cProp=np.array(hfv['Xprop_'+str(k)])
            cTarget=np.array(hfv['Ytarget_'+str(k)])
            predictions = ImgPropMergeModel([cImg, cProp], training=False)
            currentloss = loss_object(cTarget, predictions)
            currentloss = currentloss.numpy()
            testlossmat[k]=currentloss
        testloss=np.mean(testlossmat)
        ValLoss.append(testloss)

        print("Current Test Loss")
        print(testloss)
        print("Best Test Loss")
        print(bestTestLoss)
        if testloss<bestTestLoss:
            bestTestLoss=testloss
            bestTIndex=n*1.0/(N)
            ImgPropMergeModel.save("/shared/Models/"+modelname)
        with train_summary_writer.as_default():
            tf.summary.scalar('testloss', testloss, step=int(n/N))
hf.close()
hfv.close()

