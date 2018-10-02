# preprocessing my data

###################################################################################################################
import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
# =============================================================================
# df1 = pd.read_csv("crypto_data/LTC-USD.csv",names =
#                  ["time","low","high","open","close","volume"])
# 
# #just to check for file
# #print(df.head())
# 
# #preprocessing data
# =============================================================================

#define values
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = 'LTC-USD'
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

#setting up tagets
def classify(current,future):
    if float(future)> float(current):
        return 1
    else:
        return 0 
    
#preprocess func
def preprocess_df(df):
    df = df.drop(["future"],1) 

    for col in df.columns: 
        if col != "target":  
            df[col] = df[col].pct_change()  
            df.dropna(inplace=True)  
            df[col] = preprocessing.scale(df[col].values)  
    df.dropna(inplace=True)  

    sequential_data = []  
    prev_days = deque(maxlen=SEQ_LEN)  

    for i in df.values:  
        prev_days.append([n for n in i[:-1]])  
        if len(prev_days) == SEQ_LEN:  
            sequential_data.append([np.array(prev_days), i[-1]])  

    random.shuffle(sequential_data)  
    
    
    buys = []
    sells = []
    
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq,target])
        elif target == 1:
            buys.append([seq,target])
            
            
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys),len(sells))
    
    buys = buys[:lower]
    sells = sells[:lower]
    
    
    sequential_data = buys+sells
    random.shuffle(sequential_data)
    
    X = []
    y = []
    
    for seq,target in sequential_data:
        X.append(seq)
        y.append(target)
        
    return np.array(X), y


main_df = pd.DataFrame() 

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  
for ratio in ratios:  
    print(ratio)
    dataset = f'crypto_data/{ratio}.csv'  
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume']) 

    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)  
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  

    if len(main_df)==0:  
        main_df = df  
    else:  
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)  
main_df.dropna(inplace=True)
#print(main_df.head())

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)

main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))
main_df.dropna(inplace=True)

print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head(20))


times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]


preprocess_df(main_df)



train_x, train_y = preprocess_df(main_df)

val_x, val_y = preprocess_df(validation_main_df)


print(f"train data: {len(train_x)} validation: {len(val_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {val_y.count(0)}, buys: {val_y.count(1)}")

###################################################################################################################


#setting up my layer and dropouts
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint    

model = Sequential()

model.add(LSTM(128,input_shape =(train_x.shape[1:]),return_sequences=True,activation="tanh"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,input_shape =(train_x.shape[1:]),return_sequences=True,activation="tanh"))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128,input_shape =(train_x.shape[1:],),activation="tanh"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr = 1e-3, decay= 1e-6)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=["accuracy"])

tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}" 
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', 
                             verbose=1, save_best_only=True, mode='max')) 

history = model.fit(train_x,train_y, batch_size=BATCH_SIZE,epochs=EPOCHS, validation_data=(val_x,val_y),
                    callbacks=[tensorboard,checkpoint])

model.save("{}.model".format(NAME))































