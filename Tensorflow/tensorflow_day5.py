#-*- coding:utf-8 -*-
"""
@author:zjk
@file: tensorflow_day5.py
@time: 2019/09/04
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data', one_hot= True)
batch_size = 100
n_batch = mnist.train.num_examples


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    """
    strides:   stride[0] = stride[3]=1,stride[1],stride[2]分别代表x，y的步长
    padding: string from :‘SAME’,'VAILD'
    :param x: Tensor of shape [batch, height, width, channles]
    :param w:
    :return:
    """
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
     ksize : [1,x,y,1]
    :param x:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#  数据x: None * 784
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 参数：[batch, height, width, channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])

#  LAYER1
# 5*5的采样窗口， 32卷积核， 1个平面抽取特征
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])  #32个卷积核

#  卷积操作
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) #  max_pooling

#  LAYER2， 64个卷积核
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])      #产生64个特征平面
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#  全连接
#  28*28 第一次卷积后还是28*28, 池化后为7*7；第二次卷积为14*14，池化后为7*7
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# 池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)

#  dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#  第二个全连接层,softmax层
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

with tf.name_scope("cross_entory"):
    cross_entory = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar("cross_entory", cross_entory)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entory)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
with tf.name_scope("acc"):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("acc", accuracy)

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter('cnnlogs/train/', sess.graph)
    test_writer = tf.summary.FileWriter('cnnlogs/test/', sess.graph)
    for epoch in range(100):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
          #  if i % 100 == 0:
            train,train_acc, _ = sess.run([merged, accuracy,train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob:0.7})
            train_writer.add_summary(train, epoch)
            test, test_acc = sess.run([merged, accuracy], {x: mnist.test.images[:10000], y: mnist.test.labels[:10000], keep_prob: 1})
            test_writer.add_summary(test, epoch)
            print("Iter  %d batch, train_accuracy is %g, test_accuracy is %g" % (batch,train_acc, test_acc))



