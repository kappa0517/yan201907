#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:49:08 2019

@author: wangderi
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as ma

class DataTools():
    '''
    Data tools
    
    genarate training set and test set
    you can get a training set and a test set which from input data dividing by a ratio
    
    Args: 
        windows_size: RNN input sequence's length, if you have multi-indicators the windows_size will be one of them length,  int type
        Indicators: input Indicators, tuple type
        label: input label
        pred_step: output nums for RNN one step prediction
    
    Method: 
    genarate_data
    dataDivide
    
    StaticMethod:
    MinMaxNormalize
    res_normalize
    '''
    
    def __init__(self, window_size, pred_step, label=None, *Indicators):
        
        self.WINDOW_SIZE = window_size
        self.LABEL = label
        self.RATIO = 0.7
        self.INDICATORS = Indicators
        self.PRED_STEP = pred_step
        self.INPUT_X = None
        self.OUTPUT_Y = None
    
    @staticmethod
    def Extend(lst):
        
        extend_lst = list(lst[0])
        for i in range(len(lst)-1):
            extend_lst.extend(list(lst[i+1]))
        return extend_lst
    
    @staticmethod  
    def Plot(mode, model, window_size, pred_steps, Data, test_x=None, test_y=None, *Indicators):
        
        pred_input = [indicator[-window_size:] for indicator in Indicators]
        if mode == "Method1":
            test_input = test_x[-1, :, :].reshape(-1, len(pred_input), window_size).astype(np.float32)
            test_pred = model.predict(test_input)
            pred_input = np.array(pred_input).reshape(-1, len(pred_input), window_size).astype(np.float32)
            result = model.predict(pred_input)
            
        elif mode == "Method2":
            pred_input = np.stack(pred_input, 1).reshape(-1, 1, len(pred_input)*window_size).astype(np.float32)
            result = model.predict(pred_input)
            
        plt.figure(figsize=(18,8))
        plt.title("test")
        plt.plot(range(0, len(test_y[-1, :])), test_y[-1, :])
        plt.plot(range(0, len(test_y[-1, :])), test_pred)
            
        plt.figure(figsize=(18,8))
        plt.title("Prediction")
        plt.plot(range(0, len(Data[-window_size:])), Data[-window_size:])
        plt.plot(range(len(Data[-window_size:]), len(Data[-window_size:])+len(result)), result)
            
        return
        
    def genarate_data(self, mode="Method1"):
        
        input_x, output_y = [], []
        if mode == "Method1":
            if self.LABEL is None:
                for index in range(len(self.INDICATORS[0])-self.WINDOW_SIZE):
                    input_x.append([indicator[index:index+self.WINDOW_SIZE] for indicator in self.INDICATORS])
                print("No label input, dataDivide just return input_x or you can just use the 'genarare_data' to make input data")

                return input_x
            
            else:
                for index in range(len(self.INDICATORS[0])-self.WINDOW_SIZE-self.PRED_STEP):
                    input_x.append([indicator[index:index+self.WINDOW_SIZE] for indicator in self.INDICATORS])
                    if self.PRED_STEP == 1:
                        output_y.append([self.LABEL[index+self.WINDOW_SIZE]])
                    else:
                        output_y.append([self.LABEL[index+self.WINDOW_SIZE : index+self.WINDOW_SIZE+self.PRED_STEP]])
            self.INPUT_X, self.OUTPUT_Y = np.array(input_x, np.float32), np.array(output_y, np.float32).reshape(-1, self.PRED_STEP)
        
        elif mode == "Method2":           
            if self.LABEL is None:
                for index in range(len(self.INDICATORS[0])-self.WINDOW_SIZE):
                    input_x.append([self.Extend([indicator[index:index+self.WINDOW_SIZE] for indicator in self.INDICATORS])])
                print("No label input, dataDivide just return input_x or you can just use the 'genarare_data' to make input data")

                return input_x
            
            else:
                for index in range(len(self.INDICATORS[0])-self.WINDOW_SIZE-self.PRED_STEP):
                    input_x.append([self.Extend([indicator[index:index+self.WINDOW_SIZE] for indicator in self.INDICATORS])])
                    if self.PRED_STEP == 1:
                        output_y.append([self.LABEL[index+self.WINDOW_SIZE]])
                    else:
                        output_y.append([self.LABEL[index+self.WINDOW_SIZE : index+self.WINDOW_SIZE+self.PRED_STEP]])

            self.INPUT_X, self.OUTPUT_Y = np.array(input_x, np.float32), np.array(output_y, np.float32).reshape(-1, self.PRED_STEP)        
        else:
            print("No such method, you can only use 'method1' or 'method2'")
        
        return
    
    def dataDivide(self):
        
        lenth = len(self.INDICATORS[0]) - self.WINDOW_SIZE
        
        if self.LABEL is None:
            input_x = self.INPUT_X
            
            return input_x
        else:
            train_x, train_y = self.INPUT_X[ : int(lenth*self.RATIO), :, :], self.OUTPUT_Y[ : int(lenth*self.RATIO), :]
            test_x,  test_y = self.INPUT_X[int(lenth*self.RATIO) :, :, :], self.OUTPUT_Y[int(lenth*self.RATIO) :, :]

            return train_x, train_y, test_x, test_y
        
class MASON_MODEL():
    
    def __init__(self, save_path, num_hidden, num_epochs, buffer_size, batch_size, pred_step=1):
        
        self.num_hidden = num_hidden
        self.save_path = save_path
        self.num_epochs = num_epochs
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.pred_step = pred_step
        self.regressor = None
        self.train_x = None
        self.train_y = None
        
    def train_input_fn(self, train_x, train_y):
        
        self.train_x = train_x
        self.train_y = train_y
        dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        dataset = dataset.shuffle(self.buffer_size).repeat(self.num_epochs).batch(self.batch_size)
        features, labels = dataset.make_one_shot_iterator().get_next()

        return features, labels

    def eval_input_fn(self, test_x, test_y):
        
        dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        dataset = dataset.shuffle(self.buffer_size).repeat(self.num_epochs).batch(self.batch_size)
        features, labels = dataset.make_one_shot_iterator().get_next()

        return features, labels

    def pred_input_fn(self, input_x):
        
        dataset = tf.data.Dataset.from_tensor_slices(input_x)
        dataset = dataset.batch(self.batch_size)    
        features = dataset.make_one_shot_iterator().get_next()

        return features
    
    def Stack_Bi_RNNs(self, features, labels, mode):
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden*3, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden*3, forget_bias=1.0)
        features = tf.unstack(features, 3, 1)
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_rnn([lstm_fw_cell],[lstm_bw_cell], features, dtype=tf.float32)
        pred = tf.contrib.layers.fully_connected(outputs[-1], self.pred_step, activation_fn = None)
        learning_rate = 0.01
        
        return self.returnSpec(mode, pred, features, labels, learning_rate)
        
    def Bi_RNNs(self, features, labels, mode):
        
        gru = tf.contrib.rnn.GRUCell(self.num_hidden*6)
        lstm_cell = tf.contrib.rnn.LSTMCell(self.num_hidden*6)
        outputs, states  = tf.nn.bidirectional_dynamic_rnn(lstm_cell, gru, features, dtype=tf.float32)
        outputs = tf.concat(outputs, 2)
        outputs = tf.transpose(outputs, [1, 0, 2])
        pred = tf.contrib.layers.fully_connected(outputs[-1], self.pred_step, activation_fn = None)
        learning_rate = 0.01
        
        return self.returnSpec(mode, pred, features, labels, learning_rate)
        
    
    def RNNs(self, features, labels, mode):
        
        gru = tf.contrib.rnn.GRUCell(self.num_hidden*3)
        lstm_cell1 = tf.contrib.rnn.LSTMCell(self.num_hidden*6)
        lstm_cell2 = tf.contrib.rnn.LSTMCell(self.num_hidden*6)
        ln_lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.num_hidden*6)
        
        attn_lstm_cell1 = tf.contrib.rnn.AttentionCellWrapper(lstm_cell1, attn_length=10, state_is_tuple=True)
        attn_lstm_cell2 = tf.contrib.rnn.AttentionCellWrapper(lstm_cell2, attn_length=10, state_is_tuple=True)
        attn_gru_cell = tf.contrib.rnn.AttentionCellWrapper(gru, attn_length=10, state_is_tuple=True)
        
        mcell = tf.contrib.rnn.MultiRNNCell([ln_lstm_cell, attn_gru_cell, attn_lstm_cell1])
        outputs, states  = tf.nn.dynamic_rnn(mcell, features, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        pred = tf.contrib.layers.fully_connected(outputs[-1], self.pred_step, activation_fn = None)
        learning_rate = 0.01
        
        return self.returnSpec(mode, pred, features, labels, learning_rate)
        
    def returnSpec(self, mode, pred, features, labels, learning_rate):
        
        if mode == tf.estimator.ModeKeys.PREDICT:

            return tf.estimator.EstimatorSpec(mode, predictions={'predicted' : pred})

        loss =tf.losses.mean_squared_error(pred, labels)

        if mode == tf.estimator.ModeKeys.EVAL:
            metrics = {"MSE": tf.metrics.mean_absolute_error(labels, pred)}     
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        
    def createModel(self, Core="RNN"):
        
        if Core == "RNN":
            Model = self.RNNs
            print("Enable RNN Model")
            
        elif Core == "BiRNN":
            Model = self.Bi_RNNs
            print("Enable Bi-RNN Model")
        
        elif Core == "StackBiRNN":
            Model = self.Stack_Bi_RNNs
            print("Enable Bi-RNN Model")
            
        training_config = tf.estimator.RunConfig(model_dir=self.save_path, tf_random_seed=1732)
        self.regressor = tf.estimator.Estimator(model_fn=Model, config=training_config)
        
        return 
    
    def trainModel(self, train_x, train_y):
        
        self.regressor.train(input_fn=lambda:self.train_input_fn(train_x, train_y), steps=self.num_epochs)
        
        return
    
    def evalModel(self, test_x, test_y):
        
        self.regressor.evaluate(input_fn=lambda:self.eval_input_fn(test_x, test_y), steps=self.num_epochs)
        
        return
    
    def predict(self, input_x):
        
        results = self.regressor.predict(input_fn=lambda:self.pred_input_fn(input_x))
        preds = [result["predicted"] for result in results]
        if self.pred_step == 1:          
           
            return preds
        
        else:
            
            return preds[0]

if __name__ == "__main__":
      
    t = np.linspace(1, 100, 10000)
    noise = np.random.rand(10000)*0.1
    ipt1 = np.sin(2*np.pi*t) + noise
    ipt2 = np.cos(3*np.pi*t) + noise
    ipt3 = np.sin(1.5*np.pi*t) + noise
    ipt4 = np.cos(3.5*np.pi*t) + noise
    ipt5 = np.sin(2.5*np.pi*t) + noise
    Data = ipt1**2 + ipt4**3 + ipt1
    Indicators = (ipt1, ipt2, ipt3)
    newData = DataTools(300, 600, Data, *Indicators)
    newData.genarate_data("Method1")
    train_x, train_y, test_x, test_y = newData.dataDivide()
    print(train_x.shape)
    
    
    multi_output_lstm = MASON_MODEL("./aa7", 5,  2000, 100, 32, 600)
    multi_output_lstm.createModel("BiRNN")
    multi_output_lstm.trainModel(train_x, train_y)
    multi_output_lstm.evalModel(test_x, test_y)
    
    newData.Plot("Method1", multi_output_lstm, 300, 600, Data, test_x, test_y, *Indicators)