import csv
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import keras
import copy
full_dat = []

with open("weekly_tabulated_data_normalized_2010_2019.csv") as file:
    reader = csv.reader(file)
    ct = 0
    for row in reader:
        if ct == 0:
            ct += 1
            continue
        full_dat.append(row)

train_split = full_dat[0:881-522]
test_split = full_dat[881-522:]
popset = {7,8,12,14,19} #zero population areas
#popset = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19} #fucking around
#popset = {}
#popset = {3,7,8,12,14,18,19} #dengue hotspots
window = 4
lag = 8

train_input = []
train_output = []
for loc in range(1,21):
    if loc-1 in popset:
        continue
    dat = []
    for t in range(0,len(train_split)):
        if t+window+lag-1 == len(train_split):
            break
        dat2 = []
        top_temp = -20
        top_rain = -20
        low_temp = 20
        low_rain = 20
        for i in range(window):
            dat2.append(float(train_split[t+i][loc+1]))
            dat2.append(float(train_split[t+i][loc+21]))
            dat2.append(float(train_split[t+i][1]))
            if top_temp<float(train_split[t+i][loc+1]):
                top_temp=float(train_split[t+i][loc+1])
            if top_rain<float(train_split[t+i][loc+21]):
                top_rain=float(train_split[t+i][loc+21])
            if low_temp>float(train_split[t+i][loc+1]):
                low_temp=float(train_split[t+i][loc+1])
            if low_rain>float(train_split[t+i][loc+21]):
                low_rain=float(train_split[t+i][loc+21])   
            if i == window-1:
                dat2.append(float(train_split[t+i][loc+41]))
                dat2.append(top_temp)
                dat2.append(top_rain)
                dat2.append(low_temp)
                dat2.append(low_rain)
        dat.append(dat2)
    train_input.append(dat)

for t in range(0,len(train_split)):
    if t +window+lag-1 == len(train_split):
        break
    train_output.append(float(train_split[t+window+lag-1][1])-float(train_split[t+window-1][1]))

T0 = []
test_input = []
test_output = []
for loc in range(1,21):
    if loc-1 in popset:
        continue
    dat = []
    for t in range(0,len(test_split)):
        if t+window+lag-1 == len(test_split):
            break
        dat2 = []
        top_temp = -20
        top_rain = -20
        low_temp = 20
        low_rain = 20
        for i in range(window):
            dat2.append(float(test_split[t+i][loc+1]))
            dat2.append(float(test_split[t+i][loc+21]))
            dat2.append(float(test_split[t+i][1]))
            if top_temp<float(test_split[t+i][loc+1]):
                top_temp = float(test_split[t+i][loc+1])
            if top_rain<float(test_split[t+i][loc+21]):
                top_rain=float(test_split[t+i][loc+21])
            if low_temp>float(test_split[t+i][loc+1]):
                low_temp = float(test_split[t+i][loc+1])
            if low_rain>float(test_split[t+i][loc+21]):
                low_rain=float(test_split[t+i][loc+21])
            if i == window-1: 
                dat2.append(float(test_split[t+i][loc+41]))
                dat2.append(top_temp)
                dat2.append(top_rain)
                dat2.append(low_temp)
                dat2.append(low_rain)
        dat.append(dat2)
    test_input.append(dat)

for t in range(0,len(test_split)):
    if t +window+lag-1 == len(test_split):
        break
    test_output.append(float(test_split[t+window+lag-1][1])-float(test_split[t+window-1][1]))
    T0.append(float(test_split[t+window-1][1]))



input_layer = []
for i in range(0,20):
    if i in popset:
        continue
    input_layer.append(Input(shape=(window*3+5,)))
district_network_1 = Dense(24,activation = 'sigmoid')
#district_network_2 = Dense(4,activation = 'sigmoid')
conc_layer = []
for i in range(0,len(input_layer)):
    conc_layer.append(district_network_1(input_layer[i]))
    #conc_layer.append(district_network_2(district_network_1(input_layer[i])))
vector = keras.layers.concatenate(conc_layer,axis=-1)
total_net_1 = Dense(128,activation = 'relu')(vector)
drop_1 = Dropout(0.1)(total_net_1)
total_net_2 = Dense(112,activation = 'relu')(drop_1)
drop_2 = Dropout(0.1)(total_net_2)
total_net_3 = Dense(96,activation = 'relu')(drop_2)
drop_3 = Dropout(0)(total_net_3)
#total_net_4 = Dense(64,activation = 'relu')(total_net_3)
output = Dense(1,activation = 'linear')(drop_3)
model = Model(inputs = input_layer,outputs = output)
sgd = keras.optimizers.SGD(lr=0.01,decay=1e-6,momentum = 0.9,nesterov = True)
model.compile(optimizer=sgd,
              loss='mean_squared_error',
              metrics=['accuracy'])
model.save_weights('model.h5')

trials = 20
best_loss = 1
model_copy = None
history_copy = None
early_stopping_monitor = keras.callbacks.callbacks.EarlyStopping(patience=25)
compare = list(map(lambda x,y: x+y, T0 , test_output))
file = []        
for i in range(trials):
    model.load_weights('model.h5')
    history = model.fit(train_input, train_output, validation_data=(test_input,test_output),epochs=1000,callbacks=[early_stopping_monitor],verbose=1)
    y = model.evaluate(test_input,test_output)
    if y[0] < best_loss:
        best_loss = y[0]
        history_copy = copy.deepcopy(history.history)
        model_copy= keras.models.clone_model(model)
        #model_copy.build((None, (20-len(popset)),(window*3+1))) # replace 10 with number of variables in input layer
        model_copy.compile(optimizer=sgd, loss='mean_squared_error')
        model_copy.set_weights(model.get_weights())
    curr_prediction = model.predict(test_input)
    curr_prediction = list(map(lambda x,y:x+y,T0,curr_prediction.flatten()))
    def get_lag(lst):
        curr_max = -100
        ind = 0
        for i in range(len(lst)):
            if lst[i]>curr_max:
                curr_max = lst[i]
                ind = i
        return ind-10
    curr_lag = get_lag(plt.xcorr(curr_prediction,compare)[1])
    curr_row = []
    curr_row.append(y[0])
    curr_row.append(curr_lag)
    file.append(curr_row)
predictions = model_copy.predict(test_input)
predictions = list(map(lambda x,y: x+y, T0 , predictions.flatten()))
test_output = list(map(lambda x,y: x+y, T0 , test_output))
plt.plot(predictions)
plt.plot(test_output)
#plt.plot(history_copy['loss'])
#plt.plot(history_copy['val_loss'])
#training_graph = model.predict(train_input)
#plt.plot(training_graph)
#plt.plot(train_output)
print(plt.xcorr(predictions,test_output)[1])
y = model_copy.evaluate(test_input,test_output)
print(best_loss)
plt.show()

with open("score_table.csv","a",newline= "")as ret:
    writer = csv.writer(ret)
    #writer.writerow(["Loss","Lag"])
    for row in file:
        writer.writerow(row)