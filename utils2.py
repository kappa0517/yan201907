import numpy as np
import pandas as pd


def detrend(file_name,order):

    data = pd.read_csv(file_name)
    x = df.index.values  # index
    y = df.iloc[:,-1].values # target time series
    poly = np.polyfit(x, y, deg = order)

    return poly

#-----------------------------------------------------------------------------------------------------------------------
def add_trend(x_dex,poly):
    # x_dex: np.array. index of predict value

    y = np.ployval(poly,x_dex)  # y trend predict
    return y

#-----------------------------------------------------------------------------------------------------------------------
def norm_op(data,op,norm_opt):

    '''This  funtion is to apply normalization for inputdata, and return new data
    Input:
              data: input data with the shape [N,1] for label, and [time,features] for features
              norm_opt: normalize type option, here have 3 different type 'norm','max_min','max_abs' now
              op: scalar, 1 for labels and 0 for features

    Output:
             new_data: new data after normalization have the same shape with input data
             if op == 1, also return k and b for recover use'''

    # for features data  
    if op == 0:
        if norm_opt == 'max_min':
            new_data = (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0)) 
        elif norm_opt == 'norm':
            new_data = (data-data.mean(axis=0))/(np.std(data,axis=0))
        elif norm_opt == 'max_abs':
            new_data = data/np.max(np.abs(data),axis=0)

        return new_data

    elif op == 1: # for labels data

        if norm_opt == 'norm':  # use norm distribution and so-called normalization
            b, k = np.mean(data), np.std(data)
        elif norm_opt == 'max_min': # makes data range in the interval [-1 1 ] with substrct the minimum value
            b, k = np.min(data), (np.max(data)-np.min(data))
        elif norm_opt == 'max_abs': # makes data range in the intreval [0 1] with divide the absolute maximum valiue
            b, k = 0,np.max(np.abs(data))

        return (data - b)/k,k,b

#-----------------------------------------------------------------------------------------------------------------------
def read_file(file_name,time_steps,pred_n,ratio):

    '''function is to get input from csv file(file_name),and we demand input file with
     features each colunme and the time steps each rank. then we re-form the time
     -series with time step length 'time_steps'

     input:
                file_name: name of input csv file
                tine_steps: time window's length
                ratio: ratio for train and test'''
    # here if data is too big, we can consider use generator,and when we use tf,
    # can use with tf.data.from_generator
    print('Read data from file:'+'\''+file_name+'\'.')
    data = pd.read_csv(file_name) # column 1 to -2 features; column -1 for labels

    # if have date-information we use
    data = np.array(data.iloc[:,1:])

    # if have no date infornation in the file, we use:
    #data = np.array(data)

    data_fea = norm_op(data[:,:-1],0,'max_abs')        #  input feature data with normlization
    data_lab,k,b = norm_op(data[:,-1],1,'max_abs')   # inout label data witg normlization

    features = []
    labels = []

    for i in range(data_fea.shape[0]- time_steps-pred_n+1):
        features.append(data_fea[i:i+ time_steps,:])
        labels.append(data_lab[i+time_steps:i+time_steps+pred_n]) 

    features= np.array(features,dtype=np.float32).transpose((0,2,1))
    labels = np.array(labels,dtype=np.float32)
    labels = labels.reshape(labels.shape[0],pred_n)

    dex = round(data.shape[0]* ratio)

    print('Have done.')
    print()
    # return train_X train_y test_X test_y k b
    return features[0:dex,:,:], labels[0:dex,:], features[dex:,:,:], labels[dex:,:],k,b

#-----------------------------------------------------------------------------------------------------------------------
def de_norm(data,k,b):
    '''
    de-normaliz of data, for prediction use
    Input:
        data: data will be recover'
        para: [2,1] list for k and b
    '''
    return data*k+b

#-----------------------------------------------------------------------------------------------------------------------
def exp_data(ts,pred_n,seq):

    '''this function is to generate features and labels of training and test input data, 
        and only for experimental use. Here the input is index of time series, and we
        construct the features and labels with following formula:

        features = [cos(seq),sin(seq)]
        labels = [sin(seq)*cos(seq)]

        with shape [sample_number,features_number,time_steps] for features and
        [sample_number] for labels

        And finally return features and labels'''


    features, labels = [], []
    for i in range(len(seq)-ts-pred_n+1):
        features.append([np.cos(seq[i: i + ts]),np.sin(seq[i: i + ts])])
        labels.append([np.cos(seq[i+ts:i+ts+pred_n])*np.sin(seq[i+ts:i+ts+pred_n])])

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32) 

#-----------------------------------------------------------------------------------------------------------------------
def new_data(n_train,n_test,ts,d_t):

    start = (n_train + ts) * d_t            # start value of input series
    end = start + (n_test + ts) * d_t    # end value of input series
    train_X, train_y = exp_data(np.linspace(0, start, n_train + ts, dtype=np.float32))
    test_X, test_y = exp_data(np.linspace(start, end, n_test + ts, dtype=np.float32))
    b,k = 0,1

    test_y = test_y.reshape(test_y.shape[0],test_y.shape[2])
    train_y = train_y.reshape(train_y.shape[0],train_y.shape[2])
    print('Generate experimental data.')
    print()
    return train_X,train_y,test_X,test_y,k,b

#-----------------------------------------------------------------------------------------------------------------------
def create_experiment_file(exp_file_name):

    print('Generate experiment csv file.')
    n_period = 10000
    nps = np.linspace(0,100,n_period)
    df = pd.DataFrame()
    idx = pd.date_range('2018-01-01', periods=n_period, freq='H')

    df['date'] = idx
    df['feature1'] = np.sin(nps)
    df['feature2'] = np.cos(nps)
    df['feature3'] = np.sin(nps)*np.cos(nps)
    df.set_index('date')

    df.to_csv(exp_file_name,index=False)
    print('Have saved file: \''+exp_file_name+'\'.')
    print()
 

 #-----------------------------------------------------------------------------------------------------------------------   
def load_config_file(file_name):
    
    ''' this function is to load configure parameters.
    '''
    
    print('Load configure parameters from file \''+file_name+'\'.')
    fid = open(file_name)
    flag,data_dict,para_dict = 0,{},{}

    for line in fid:
        
        if line.strip().startswith('@'):
            
            flag = 1
        
        
        if flag == 0:
            
            if not line.strip().startswith('#'):

                temp = line.split(':')
                if len(temp) > 1:

                    if  temp[1].strip().isdigit():
                         data_dict[temp[0].strip()] = int(temp[1])
                    elif temp[1].split(".")[0].strip().isdigit():
                        data_dict[temp[0].strip()] = float(temp[1])
                    else:
                        data_dict[temp[0].strip()] = temp[1].strip()
                        
        if flag == 1:
            
            if not line.strip().startswith('#'):

                temp = line.split(':')
                if len(temp) > 1:

                    if  temp[1].strip().isdigit():
                         para_dict[temp[0].strip()] = int(temp[1])
                    elif temp[1].split(".")[0].strip().isdigit():
                        para_dict[temp[0].strip()] = float(temp[1])
                    else:
                        para_dict[temp[0].strip()] = temp[1].strip()
                        

    fid.close()
    
    print('Have done.')
    print()
    print('Dictionary for data is:')
    
    for key in data_dict:
          print('{}: {}'.format(key,data_dict[key]))
            
    print()
    print('Dictionary for RNN model is:')
    for key in para_dict:
          print('{}: {}'.format(key,para_dict[key]))
            
    print()
    
    return data_dict,para_dict

#-----------------------------------------------------------------------------------------------------------------------
def create_configure_file(config_file):
    
    ''' this function is to create congifure file for generate parameters dictionary of RNN_model'''
    
    
    print('Generate congigure file ...')
    fid = open(config_file,'w')
    fid.write('# This is input config file for MTRP algorithm.\n')
    fid.write('\n')
    fid.write('\n')
    fid.write('# Following lines before line begin with character @ is parameters for data processing.\n')
    fid.write('\n')
    fid.write('# ts is timelength for generator data [int].\n')
    fid.write('ts  : 200\n')
    fid.write('\n')
    fid.write('# pred_n is number of predict points equal to same parameter in RNN dictionary [int]. \n')
    fid.write('pred_n : 2\n')
    fid.write('\n')
    fid.write('# n_train is number of training sample if use generator data [int].\n')
    fid.write('n_train : 10000\n')
    fid.write('\n')
    fid.write('# n_test is number of testing sample if use generator data [int].\n')
    fid.write('n_test : 1000\n')
    fid.write('\n')
    fid.write('# d_t is time interval for each generator data with same pace [float].\n')
    fid.write('d_t : 0.1\n')
    fid.write('\n')
    fid.write('# time_steps is time window length for sample [int].\n')
    fid.write('time_steps : 30\n')
    fid.write('\n')
    fid.write('# ratio is ratio of training data in all the dataset [float].\n')
    fid.write('ratio : 0.8\n')
    fid.write('\n')
    fid.write('# file_name is name of input data [string].\n')
    fid.write('file_name : /Users/kappa0517/Code/RNN_predict/data0416.csv\n')
    fid.write('\n')
    fid.write('\n')
    fid.write('@ Follow lines before the end of this file is parameters for RNN model.\n')
    fid.write('\n')
    fid.write('# hidden_size is the number of hidden units for each layer of RNN model [int].\n')
    fid.write('hidden_size : 30\n')
    fid.write('\n')
    fid.write('# num_layers is the number of layers of RNN model [int].\n')
    fid.write('num_layers : 1\n')
    fid.write('\n')
    fid.write('# batch_size is number for each batch [int], should be power of 2 is better.\n')
    fid.write('batch_size : 32\n')
    fid.write('\n')
    fid.write('# epoch is numbner for each sample using during the training [int].\n')
    fid.write('epoch : 5\n')
    fid.write('\n')
    fid.write('# learning_rate is rate of learning during the training [float].\n')
    fid.write('learning_rate : 0.06\n')
    fid.write('\n')
    fid.write('# shuffle_size is random number which should 2 times greater than batch size [int].\n')
    fid.write('shuffle_size : 10000\n')
    fid.write('\n')
    fid.write('# optimize is name of optimizer [string].\n')
    fid.write('optimize : Adam\n')
    fid.write('\n')
    fid.write('# train_step is training steps [int].\n')
    fid.write('train_step : 3000\n')
    fid.write('\n')
    fid.write('# pred_n is number of predict [int].\n')
    fid.write('pred_n : 2\n')
    fid.write('\n')
    fid.write('# save_path is the path where to save training model [string].\n')
    fid.write('save_path : ./model\n')
    fid.write('\n')
    fid.write('# save_model is the path where to save serving model for serving ues [string].\n')
    fid.write('save_model : ./serve_model\n')
    fid.write('\n')
    fid.write('# whather use attention [string].\n')
    fid.write('attn : False')
    fid.write('\n')
    fid.write('#attention length [int].\n')
    fid.write('\n')
    fid.write('attn_length : 10')
    fid.write('\n')
    fid.write('#whather use layNormal [string].\n')
    fid.write('\n')
    fid.write('layerNorm : False')
    fid.write('\n')   
    fid.close()
    print('Have saved configure file as \''+config_file+'\'.')
    print()