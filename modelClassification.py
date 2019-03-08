#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
#

import tensorflow as tf
import numpy as np
import math
import os

batch_size = 100
n_input = 499
save_steps = 1000

class modelclassification():
    def __init__(self):

        self.X = tf.placeholder('float',[None,n_input])

        self.y = tf.placeholder('float',[None,2])

        self.hidden_variable = tf.placeholder('float',[None,int(math.sqrt(n_input))])

    # 构建网络模型
    def build_model(self,n_input):

        hidden1_units = int(math.sqrt(n_input))
        hidden2_units = int(math.sqrt(n_input))
        # print('hidden1_units is ', hidden1_units)
        # print('hidden2_units is ', hidden2_units)

        # hidden1层
        with tf.name_scope('hidden1'):
            # weights[784,28]
            weights = tf.Variable(
                tf.truncated_normal([n_input, hidden1_units],
                                    stddev=1.0 / math.sqrt(float(n_input))),
                name='weights')  # 权重是标准方差为输入尺寸开根号分之一的正态分布
            # biases[28]
            biases = tf.Variable(tf.zeros([hidden1_units]),
                                 name='biases')
            # hidden1[none,28]
            hidden1 = tf.nn.relu(tf.matmul(self.X, weights) + biases)
        # hidden2层
        with tf.name_scope('hidden2'):
            # weights[28,28]
            weights = tf.Variable(
                tf.truncated_normal([hidden1_units, hidden2_units],
                                    stddev=1.0 / math.sqrt(float(hidden1_units))),
                name='weights')
            # # biases[28]
            biases = tf.Variable(tf.zeros([hidden2_units]),
                                 name='biases')
            # hidden2[none,28]
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
            self.hidden_variable =hidden2
            # print('hidden2 is ',hidden2)
            # Linear
        with tf.name_scope('full_connect'):
            weights = tf.Variable(
                tf.truncated_normal([hidden2_units, 2],
                                    stddev=1.0 / math.sqrt(float(hidden1_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([2]),
                                 name='biases')

            self.pred = tf.nn.softmax(tf.matmul(hidden2, weights) + biases)
            # print('predition is ',self.pred)

    def build_loss(self):

        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.loss = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def build_training(self, learning_rate, ):

        tf.summary.scalar('loss', self.loss)
        loss_temp = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.pred)
        cross_entropy_loss = tf.reduce_mean(loss_temp)
        self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                            epsilon=1e-08).minimize(cross_entropy_loss)

    def out_model(self):

        return self.hidden_variable

    def train_model(self,X,Y):

        datasize = X.shape[0]

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True

        # with tf.Session(config=config) as sess:
        with tf.Session(config=tf.ConfigProto(device_count={'cpu':0})) as sess:
            sess.run(tf.global_variables_initializer())
            # ckpt = tf.train.latest_checkpoint()
            STEPS = 55000

            for i in range(STEPS):
                start = (i * batch_size) % datasize
                end = min(start + batch_size, datasize)   #每次按照end - start个数据进行训练（选择100）
                # tf.convert_to_tensor()
                feed = {self.X: X[start:end], self.y: Y[start:end]}#训练数据集{X:x_data,Y:y_data}
                sess.run(self.train_step, feed_dict=feed)
                if i % 1000 == 1:
                    batchloss = sess.run(self.loss, feed_dict=feed)
                    # correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
                    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                if i % save_steps == 1 and batchloss < 100:
                    saver.save(sess, os.path.join('checkpoint_1', 'latent_model'),
                               global_step=i)

    def infer(self, X):

        saver = tf.train.Saver()
        out = self.out_model()

        print(out)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver.restore(sess,'./checkpoint_1/latent_model-52001')
            result = sess.run(out, feed_dict={self.X: X})
            # print result
        return result







