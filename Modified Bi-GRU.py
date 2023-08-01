# -*- coding: utf-8 -*-
from re import split
from keras import layers
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Bidirectional,GRU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def ANN_data_transfer_IQ(x,taps,dim):
    # x=x[:,np.newaxis]
    z=np.int(len(x)-taps)
    z_IQ = np.int(z/dim)
    y=np.zeros([z_IQ+1,taps])
    for i in range(0,z+dim,dim):
        for k in range(taps):
            y_IQ = np.int(i/dim)
            y[y_IQ,k] = x[i+k]
    return y

# parameters
n_steps = 11
epoch=100
batch_size=200
ratio=0.3
taps = n_steps
dims = 1
n_features = 1
Nonlinear_order = 3
retrain=1
## name of file
Extra = '1'
path = 'Rx\\'
name_Tx = ['C:\PAM4_data.txt']
name_Rx = ['C:\Rx.txt']
name_Sa = ['Rx_GRU_train_',Extra,'.txt']
name_Sa = ''.join(name_Sa)
## load data
x_Tx    = np.loadtxt(name_Tx)
x_Rx    = np.loadtxt(name_Rx)
#prepare the data for LSTM
x_Tx_length=len(x_Tx)
x_Rx_length=len(x_Rx)
x_train_1=x_Rx[0:np.int(x_Rx_length*ratio)]
y_train_1=x_Tx[0:np.int(x_Tx_length*ratio)]
x_test_1=x_Rx[np.int(x_Rx_length*ratio): ]
y_test_1=x_Tx[np.int(x_Tx_length*ratio): ]

# train sequence and label
x_train=ANN_data_transfer_IQ(x_train_1,taps*dims,dims)
y_train=y_train_1[np.int((taps-1)/2):np.int(-(taps-1)/2)]
X=ANN_data_transfer_IQ(x_Rx,taps*dims,dims)
x_test=ANN_data_transfer_IQ(x_test_1,taps*dims,dims)
y_test=y_test_1[np.int((taps-1)/2):np.int(-(taps-1)/2)]

# # reshape from [samples, timesteps] into [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Callback
reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.9,verbose=1,
                            patience=5, min_lr=0, mode='auto')
EarlyStop = EarlyStopping(monitor='val_mae', min_delta=0.000001,
                                            patience=epoch*0.5, verbose=1, mode='auto')
if retrain==True:
    modelinputs = keras.Input(shape=(n_steps, n_features),name='input_layer')
    fowardGRUinput = keras.layers.Lambda(lambda x: x[:,0:int((taps-1)/2+1),:])(modelinputs)
    backwardGRUinput = keras.layers.Lambda(lambda x: x[:,int((taps-1)/2): ,:])(modelinputs)
    forwardGRUlayer = GRU(100, activation='tanh', recurrent_activation = 'sigmoid',return_sequences=False)(fowardGRUinput)
    backwardGRUlayer = GRU(100, activation='tanh', recurrent_activation = 'sigmoid',return_sequences=False,go_backwards=True)(backwardGRUinput)
    bi_GRU = keras.layers.Concatenate()([forwardGRUlayer,backwardGRUlayer])
    y = keras.layers.Dense(1)(bi_GRU)
    model = keras.Model(inputs=modelinputs,outputs=y,name = 'bi_nn')

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()
    # fit model
    HisI=model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size,
            validation_data=(x_test, y_test), callbacks=[reduce_lr,EarlyStop])
    # model.save('GRU',True,True)
else:
    model = keras.models.load_model('GRU'+'1485')
# demonstrate prediction
x_equalization=model.predict(X,batch_size=batch_size)
##
newpath ='.\\NN'
# if not os.path.exists(newpath):
#     os.makedirs(newpath)
np.savetxt(newpath+'\\'+name_Sa,x_equalization,fmt='%1.7e')

plt.figure()
np.savetxt('epoch.txt',HisI.epoch,fmt='%1.7e')
np.savetxt('train_loss.txt',HisI.history['loss'],fmt='%1.7e')
np.savetxt('val_loss.txt',HisI.history['val_loss'],fmt='%1.7e')
plt.plot(HisI.epoch,HisI.history['loss'],'r') 
plt.plot(HisI.epoch,HisI.history['val_loss'],'b')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(['Train loss ', 'Validation loss of '])
plt.show()