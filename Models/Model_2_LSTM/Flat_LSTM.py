from keras.models import Model
from keras.layers import Dense,LSTM,Input,Dropout,Bidirectional,Reshape,Concatenate
import csv
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
''' LOAD DATA '''

data = []

with open('weekly_tabulated_data_normalized_2010_2019.csv')as f:
    reader = csv.reader(f)
    ct = 0
    for row in reader:
        if ct == 0:
            ct+=1
            continue
        data.append(row)
        
''' HYPERPARAMETERS '''

dengue_window = 4
lag = 8
#locset={} #basic
locset = {7,8,12,14,19} #zero population areas
#locset = {0,1,2,3,4,5,6,9,10,11,13,15,16,17,18} #fucking around
#locset = {}
#locset = {3,7,8,12,14,18,19} #dengue hotspots
#locset = {7,8,11,12,14,16,18,19} 

''' PROCESS DATA '''

#0=train
#1=test
#x0 = input
#x1 = output

train_test_split = [data[0:881-522],data[881-522:]]
in_out_data = [[[],[]],[[],[]]]

for i in range(2):
    curr_split_set = train_test_split[i]
    curr_in_set = in_out_data[i][0]
    curr_out_set = in_out_data[i][1]
    size = len(curr_split_set)
    
    ''' output '''
    
    for t in range(0,size-lag-dengue_window):
        
        curr_out_set.append(float(curr_split_set[t+lag+dengue_window-1][1]))
    
    ''' input '''
    
    for loc in range(20):
        if loc in locset:
            continue
        
        ''' dis input '''
        
        dis_in = []
        
        for t in range(0,size-lag-dengue_window):
            
            temp = float(curr_split_set[t+dengue_window-1][loc+2])
            rf = float(curr_split_set[t+dengue_window-1][loc+22])
            pop = float(curr_split_set[t+dengue_window-1][loc+42])
           
            dis_in.append([temp,rf,pop])
            
        curr_in_set.append(dis_in)
    
        
    ''' dengue input '''
    
    dengue_in = []
    
    for t in range(0,size-lag-dengue_window):
        
        lstm_inpt = []
        
        for i in range(dengue_window):
            
            lstm_inpt.append([float(curr_split_set[t+i+lag-1][1])])
        
        dengue_in.append(lstm_inpt)
        
    curr_in_set.append(dengue_in)
    
'''
train_input = np.array(in_out_data[0][0])
train_output = np.array(in_out_data[0][1])
test_input = np.array(in_out_data[1][0])
test_output = np.array(in_out_data[1][1])
#print(np.shape(train_input))
'''
train_input = in_out_data[0][0]
train_output = in_out_data[0][1]
test_input = in_out_data[1][0]
test_output = in_out_data[1][1]
#print(train_input)

''' ACTIVATION FUNCTIONS '''

def relu(x): return keras.activations.relu(x,alpha = 0.01,threshold = 0)
def elu(x): return keras.activations.elu(x,alpha = 1)

act = relu

''' OPTIMIZERS '''

sgd = keras.optimizers.SGD(lr=0.01,decay=1e-6,momentum = 0.9,nesterov = True,clipnorm=1)
adam = keras.optimizers.Adam(lr=0.001,clipnorm = 1)

opt = sgd

''' LOSSES '''

def custom_loss(true,pred):
    def momentum(x):
        res = x[1:,:] - x[:-1,:]
        t1 = [[0.1]] # dummy for dimensions
        return tf.concat([t1, res], 0)
    def force(x):
        res = x[2:] - x[1:-1,:] - x[1:-1,:] + x[:-2,:]
        t1 = [[0.1]] # dummy
        return tf.concat([t1, t1, res], 0)
    def MSE(x,y):
        return K.square(x-y)
    return MSE(momentum(true), momentum(pred)) + MSE(true, pred) + MSE(force(true), force(pred))

loss_fn = custom_loss

''' MODEL '''

district_nn = 16 #16
l1 = 1 #1
l2 = 1 #1
d = 50 #200

input_layer = []
for loc in range(20):
    if loc in locset:
        continue
    input_layer.append(Input(shape=(3,)))
input_layer.append(Input(shape = (dengue_window,1)))

district_input = Input(shape=(3,))
district_dense = Dense(district_nn,activation = act)(district_input)

district_section = Model(inputs = district_input,outputs = district_dense)

lstm_input = Input(shape=(dengue_window,1))
lstm_1 = Bidirectional(LSTM(l1,activation = act,return_sequences = True))(lstm_input)
lstm_2 = Bidirectional(LSTM(l2,activation = act))(lstm_1)

lstm_section = Model(inputs = lstm_input,outputs = lstm_2)

conc_layer = []

for i in range(0,len(input_layer)-1):
    conc_layer.append(district_section(input_layer[i]))
conc_layer.append(lstm_section(input_layer[len(input_layer)-1]))    
    
merge = keras.layers.concatenate(conc_layer)

dense = Dense(d,activation = act )(merge)

prediction = Dense(1,activation = 'linear')(dense)

model = Model(inputs = input_layer,outputs = prediction)

''' COMPILE AND TRAIN '''

early_stopping_monitor = keras.callbacks.EarlyStopping(patience=50)
model.compile(optimizer = opt,loss = loss_fn)
model.fit(train_input,train_output,
          validation_data=(test_input,test_output),
          callbacks=[early_stopping_monitor],
          epochs = 1000)

''' EVALUATE '''

def get_lag(lst):
        curr_max = -100
        ind = 0
        for i in range(len(lst)):
            if lst[i]>curr_max:
                curr_max = lst[i]
                ind = i
        return ind-10

predictions = model.predict(test_input)
predictions = predictions.flatten()

print(model.evaluate(test_input,test_output)) 
print(get_lag(plt.xcorr(predictions,test_output)[1]))

plt.plot(predictions)
plt.plot(test_output)
plt.show()
#plt.figure(figsize=(20,10))