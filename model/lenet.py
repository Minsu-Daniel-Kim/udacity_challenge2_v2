import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist

def lenet(images, scope='LeNet', is_training=True):
    tf.image_summary("image", images)
    net = slim.layers.conv2d(images, 20, [5,5], padding='SAME', scope='conv1')
    net = slim.layers.max_pool2d(net, [2,2], scope='pool1')
    net = slim.layers.conv2d(net, 50, [5,5], padding='SAME', scope='conv2')
    net = slim.layers.max_pool2d(net, [2,2], scope='pool2')
    net = slim.layers.flatten(net, scope='flatten3')
    net = slim.layers.fully_connected(net, 500, scope='fully_connected4')
    net = slim.layers.fully_connected(net, 100, scope='fully_connected5')
    net = slim.layers.fully_connected(net, 1, scope='fully_connected6')
    return net
