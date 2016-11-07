from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def drivenet(inputs,
             NUM_CLASS=1,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
             scope='drivenet'):

    with tf.variable_scope(scope, 'driveNet', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],activation_fn=tf.nn.relu,
                            outputs_collections=[end_points_collection]):
            tf.image_summary("img", inputs, max_images= 20)
            net = slim.conv2d(inputs, 24, [1, 1], 5, padding='VALID',
                              scope='conv1')

            net = slim.conv2d(net, 36, [1, 1], 5, scope='conv2')
            net = slim.conv2d(net, 48, [1, 1], 5, scope='conv3')
            net = slim.conv2d(net, 64, [1, 1], 5, scope='conv4')
            net = slim.conv2d(net, 64, [1, 1], 5, scope = 'conv5')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1164, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout7')

            net = slim.fully_connected(net, 100, scope='fc8')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout9')
            net = slim.fully_connected(net, 50, scope='fc10')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout11')
            net = slim.fully_connected(net, 10, scope='fc12')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout13')
            net = slim.fully_connected(net, NUM_CLASS, scope='fc14', activation_fn=None)

            return net, None
