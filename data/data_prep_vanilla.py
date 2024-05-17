import json
import pandas as pd

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math

#Return calculation
def returnCalculation (database,lag):
    """
    Function to calculate lag return from the input
    -----------------------------------------------
    Input: 
        database (array/pandas.Series) : price data 
        lag (int) : lag for return calculation
    
    Returns: 
        array : 

    """
    dimension=database.shape[0];dif=lag;Out=np.zeros([dimension-dif])
    for i in range(dimension-dif):
        Out[i]=(np.log(database[i+dif])-np.log(database[i]))
    return np.append(np.repeat(np.nan, dif),Out)

#STD Calculation
def sdCalculation (dailyReturns, lagSD):
    """
    Function to calculate lag standard deviation from the input
    -----------------------------------------------
    Input: 
        database (array/pandas.Series) : price data 
        lagSD (int) : lag for std calculation
    
    Returns: 
        array : 

    """
    dimension=dailyReturns.shape[0]; dif=lagSD; Out=np.zeros([dimension-dif])
    for i in range (dimension-dif):
        Out[i]=np.std(dailyReturns[i:i+lagSD],ddof=1)
    return np.append(np.repeat(np.nan, dif),Out)

def position_encoding_init(n_position, d_pos_vec):
    position_enc = np.array([
        [math.pi*(pos/(n_position-1)) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])
    return np.cos(position_enc)


def vanilla_transformer_database(data, timestep, lag, lagSD, test_size, purge_size): 
    """
    Function for preparing train and test data for hybrid transformer model.
    -----------------------------------------------
    Input: 
        data (dataframe) : price data 
        timestep (int) : sequential timestep for inputing to transformer model
        lag (int) : lag for return calculation
        lagSD (int) : lag for std calculation
        test_size (float) : test data size (fraction value)
        purge_size (int) : # of data points to purge from train data for preventing leakage

    Returns: 
        xtrain (tensor), xtest (tensor), ytrain (tensor), ytest (tensor), 
        train_index (array), test_index (array)

    """
    return_data = pd.DataFrame(index = data.index)
    for col in data.columns: 
        return_data[col + '_lag_return'] = returnCalculation(data[col], lag)
        return_data[col + '_lag_sd'] = sdCalculation(data[col], lagSD) 
        return_data[col + '_true_return'] = returnCalculation(data[col], 1) 
    return_data.dropna(inplace = True)

    xdata = return_data[[col for col in return_data.columns if 'true' not in col]].copy()
    ydata = return_data[[col for col in return_data.columns if 'true' in col]].copy() 

    scaler = StandardScaler().fit(xdata) 
    scaled_xdata = scaler.transform(xdata) 

    features = scaled_xdata.shape[1] 
    sample = scaled_xdata.shape[0] - timestep + 1 

    xdataTrainScaledRNN = np.zeros([sample, timestep, features]) 
    ydataTrainRNN = ydata.iloc[timestep - 1:, :]

    # adding positional encoding
    pos_encoding = position_encoding_init(timestep, xdata.shape[1])
    for i in range(sample): 
        xdataTrainScaledRNN[i, :, :] = scaled_xdata[i:(timestep + i)] + pos_encoding
    
    xtrain, xtest, ytrain, ytest = train_test_split(xdataTrainScaledRNN, ydataTrainRNN, test_size = test_size, shuffle = False)

    xtrain = xtrain[:-purge_size, :, :]
    ytrain = ytrain.iloc[:-purge_size, :]

    train_index = ytrain.index
    test_index = ytest.index
    return (
        tf.convert_to_tensor(xtrain, np.float32), 
        tf.convert_to_tensor(xtest, np.float32), 
        tf.convert_to_tensor(ytrain.values, np.float32), 
        tf.convert_to_tensor(ytest.values, np.float32), 
        train_index, test_index
    )
