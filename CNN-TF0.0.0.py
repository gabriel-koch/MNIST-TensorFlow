# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:24:36 2018

@author: R210
"""

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    '''Model function for CNN'''
    #Input Layer
    #Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features['X'],[-1, 28, 28, 1])
    
    #Convolutional Layer 
    #Computes 3 features maps using 5x5 filter with sigmoid activation
    #Input Tensor Shape: [batch_size, 28, 28, 1]
    #Output Tensor Shape: [batch_size, 24, 24, 20]
    conv = tf.layers.conv2d(
            inputs = input_layer,
            filters = 20,
            kernel_size = [5,5],
            activation = tf.nn.sigmoid)
    
    #Pooling Layer 
    #Max pooling layer with a 2x2 flter and stride of 2
    #Input Tensor Shape: [batch_size, 24, 24, 20]
    #Output Tensor Shape: [batch_size, 12, 12, 20]
    pool = tf.layers.max_pooling2d(
            inputs = conv,
            pool_size = [2, 2],
            strides = 1)
    
    #Flatten tensor into a batch of vectors
    #Input Tensor Shape: [batch_size, 12, 12, 20]
    #Output Tensor Shape: [batch_size, 12 * 12 * 20]
    pool_flat = tf.reshape(pool, [-1, 12 * 12 * 20])
    
    #Fully Connected Layer = dense layer
    #
    #Input Tensor Shape: [batch_size, 12 * 12 * 20]
    #Output Tensor Shape: [batch_size, 100]
    fully_cnted = tf.layers.dense(
            inputs = pool_flat,
            units = 100,
            activation = tf.nn.sigmoid)
    
    #Output Layer
    #Softmax
    #Input Tensor Shape: [batch_size, 100]
    #Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(
            inputs = fully_cnted,
            units = 10,
            activation = tf.nn.softmax)
    
    predictions = {
        #Generate predictions (for PREDICT and EVAL mode)
        "classes":tf.argmax(input = logits, axis = 1),
        #Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        #`logging_hook´.
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
        }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    #Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    #Configura the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, 
                                          loss = loss, 
                                          train_op = train_op)
    
    #Add evaluations metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels = labels, predictions = predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    