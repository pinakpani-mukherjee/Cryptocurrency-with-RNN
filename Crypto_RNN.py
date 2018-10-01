import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
# =============================================================================
# df1 = pd.read_csv("crypto_data/LTC-USD.csv",names =
#                  ["time","low","high","open","close","volume"])
# 
# #just to check for file
# #print(df.head())
# 
# #preprocessing data
# =============================================================================


SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = 'LTC-USD'

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



#train_x, train_y = preprocess(main_df)

#val_x, val_y = preprocess(validation_main_df)



