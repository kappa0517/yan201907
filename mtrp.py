#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2019/06/18 15:15:00

@author: YanChuan
"""

# Import package from the source...

print('Import package...')

import numpy as np
import functools
from pathlib import Path
import tensorflow as tf
from tensorflow.contrib import predictor
from utils import *
import time
import matplotlib.pyplot as plt
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

print('Have done!')


# setting logging information

logging.basicConfig(level = logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)

class get_model():
    
    def __init__(self,para):
        
        self.hidden_size = para['hidden_size']
        self.num_layers = para['num_layers']
        self.batch_size = para['batch_size']
        self.epoch = para['epoch']
        self.learning_rate = para['learning_rate']
        self.shuffle_size = para['shuffle_size']
        self.optimize = para['optimize']
        self.train_step = para['train_step']
        self.save_path = para['save_path']
        self.pred_n = para['pred_n']
    
    
    def input_data(self,features,labels,mode):
    
    # data.dataset is an important senior API of tensoeflow for construct deeplearning
    # algorithm. this API mainly use function such as from_tensor_slices, shuffle,map,
    # repeat,batch and generator make_one_shot_iterator().get_next().
    #
    # here mainly have 3 ways to input numerical data into dataset:
    # 1: from data(features and labels)
    # 2: from file(sucn as with the format of tfrecord)
    # 3: from tensor(tensor of features and tensor of labels)
    #
    # here we use .from_tensor_slices for input format, this way is suitable for data size 
    # not very big size, in this case can get data from generator.
    # reference:
    #                   https://tensorflow.google.cn/guide/performance/datasets
    #                   https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    
        ds = tf.data.Dataset.from_tensor_slices({"feature": features, "label": labels}) # not need shuffle when eval
        if mode == 'train':
            ds = ds.shuffle(self.shuffle_size).repeat(self.epoch).batch(self.batch_size)
        elif mode == 'eval' or mode == 'predict':
            ds = ds.repeat(1).batch(self.batch_size)

        rst = ds.make_one_shot_iterator().get_next() # tensor of features and labels
        return rst['feature'], rst['label']
    
    #-----------------------------------------------------------------------------------------------------------------
    def model_fn(self,  features,  labels,  mode):
    
    # construct the RNN net with num_layers layers and num_hidden units for each layer
    # and return tf.estimator.EstimatorSpec for using tf.estimator
    
    # one thing need to remember is will we need multipl-processing?
    
        if isinstance(features, dict):  # For serving
            features = features['feature']
        
        with tf.name_scope("RNN"):
            cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_size) 
                                            for _ in range(self.num_layers)])  
            outputs, _ = tf.nn.dynamic_rnn(cell, features, dtype=tf.float32)
            output = outputs[:, -1, :]
            
            # addtion a full connection layer at last 
            predictions = tf.contrib.layers.fully_connected(output, self.pred_n, activation_fn=None)
       
        # for Predict use
        if mode == tf.estimator.ModeKeys.PREDICT:
            
            predict_output = {'values': predictions}
            export_outputs = {'predictions': tf.estimator.export.PredictOutput(predict_output)}
        
            return tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = predict_output,
                export_outputs=export_outputs)

 
        # now mse, and can set this part optional
        with tf.name_scope("Loss"):
            loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
        
        with tf.name_scope("Train"):
            train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(),
                                                   optimizer=self.optimize, 
                                                   learning_rate=self.learning_rate)
            metrics = {"mae": tf.metrics.mean_absolute_error(labels, predictions)} 
            tf.summary.scalar('Loss', loss)
            
        if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:

            return tf.estimator.EstimatorSpec(mode=mode,
                                                loss=loss,
                                                train_op=train_op)
        
    #----------------------------------------------------------------------------------
    def serving_input_receiver_fn(self,xshape):

        number = tf.placeholder(dtype=tf.float32, shape=[None,xshape[0],xshape[1]], name='serve_input')
        receiver_tensors = {'serve_input': number}
        features =  number
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

        
    def pred_model(self,test_X,test_y,md,k,b):
        
        '''test_X: features of  test/predict data
        test_y: labels of test/predict data, for re_norm and for predict use
        md: estimator
        k: slope
        b: interpret'''
        
        
        results = md.predict(lambda:self.input_data(test_X,test_y,'eval'))
        rst = [result["pred"] for result in results]
        rst = np.array(rst)

        test_denorm = test_y*k+b
        rst_denorm = rst*k+b
        
        return rst,rst_denorm,test_denorm

    
    def plot_rst(self,test_y,rst):
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.plot(test_y, label="Actual Values", color='red')
        plt.plot(rst, label="Predicted Values", color='green', )

        plt.title('Result')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend(loc='best')
        plt.show()
      
        
if __name__ == '__main__':
    
    
    # generate input dictionary
    # for experiment use
    config_file = 'config.txt'
    csv_name = 'exp.csv'

    create_configure_file(config_file)
    data_dict,para_dict = load_config_file(config_file)
    create_experiment_file(csv_name)
    
    # get input data
    print('Get input data...')
    train_X,train_y,test_X,test_y,k,b = read_file(csv_name,data_dict['ts'],data_dict['pred_n'],data_dict['ratio'])
    data_shape = (train_X.shape[1],train_X.shape[2])
    print('Have done!')
    print()
    
    # train model
    print('Create estimator now...')
    Path(para_dict['save_path']).mkdir(exist_ok=True)
    estimator = tf.estimator.Estimator(get_model(para_dict).model_fn, para_dict['save_path'])
    print('Have done!')
    print()
    
    print('Training model now...')
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda :get_model(para_dict).input_data(train_X,train_y,'train'))

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda :get_model(para_dict).input_data(test_X,test_y,'eval'),
        exporters=[tf.estimator.LatestExporter(
            name="eval", 
            serving_input_receiver_fn=lambda :get_model(para_dict).serving_input_receiver_fn(data_shape),
            exports_to_keep=1,
            as_text=True)],
            steps=None)


    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec, 
        eval_spec=eval_spec)
    print('Have done!')
    print()
    
    
    # export model
    print('Export model file to '+para_dict['save_model']+'.')
    estimator.export_saved_model(para_dict['save_model'], 
                                 lambda :get_model(para_dict).serving_input_receiver_fn(data_shape))
    
    # Predict
    print('Prediction now...')
    Path(para_dict['save_path']).mkdir(exist_ok=True)
    estimator = tf.estimator.Estimator(get_model(para_dict).model_fn, para_dict['save_path'])
    results =estimator.predict(input_fn=lambda :get_model(para_dict).input_data(test_X,test_y,'predict'))
    rst = [result["values"] for result in results]
    rst = np.array(rst)

    #-----------------For Serving Path.iterdir() generate  all files below path
    subdirs = [x for x in Path(para_dict['save_model']).iterdir() if x.is_dir() and 'temp' not in str(x)]
    predict_fn = predictor.from_saved_model(str(sorted(subdirs)[-1]))

    #test_data = np.random.random((5,data_shape[0],data_shape[1]))
    test_data = test_X[:50,:,:]
    pred_rst = []
    tic = time.time()
    for i in range(test_data.shape[0]):

        pred = predict_fn({'serve_input': [test_data[i,:,:]]})['values']
        pred_rst.append(pred[0])

    pred_rst = np.array(pred_rst)
    rst2 = pred_rst*k+b
    toc = time.time()
    print('Have done!')
    print()
    print('Run time {} seconds.'.format(toc-tic))
    print()
    print('Predict result:')
    print(pred_rst)