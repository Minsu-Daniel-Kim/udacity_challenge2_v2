import tensorflow as tf
import tensorflow.contrib.slim as slim
from util.input import inputs
from model.lenet import lenet
from model.vgg16 import vgg16
from model.cifarnet import cifarnet
from model.alexnet import alexnet_v2
from model.resnet_v1 import resnet_v1_50
import pandas as pd
import argparse
import numpy as np


flags = tf.app.flags
flags.DEFINE_string('model', None, "Model name")
flags.DEFINE_integer('label', -1, 'label')
FLAGS = flags.FLAGS

models = {

    'lenet': lenet,
    'alexnet': alexnet_v2,
    'resnet': resnet_v1_50,
    'cifarnet': cifarnet,
    'vgg': vgg16
}

def export_result(frame_id_lst, steering_angle_lst, model):
    dict = {'frame_id': frame_id_lst, 'steering_angle': steering_angle_lst}
    pd.DataFrame(dict).to_csv("output/%s_output.csv" % model,index=False)

def convert_data_to_tensors(x, y):
    inputs = tf.constant(x)
    inputs.set_shape([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])

    outputs = tf.constant(y)
    outputs.set_shape([y.shape[0], 1])
    return inputs, outputs


train_dir = 'data'
logdir = 'log_regression/cifarnet_%s' % FLAGS.label
labels = pd.read_csv("output/labels2_output.csv")

with tf.Graph().as_default():
    # pass

    num_data = 6601
    image, image_name = inputs(train_dir, "submission", num_data, None, one_hot_labels=False)

    target_image_names = labels['frame_id'][labels['steering_angle'] == FLAGS.label]
    tmp_images = []
    tmp_image_names = []
    for i in range(len(image)):

        if int(image_name[i]) in list(target_image_names):
            tmp_images.append(image[i])
            tmp_image_names.append(image_name[i])
    tmp_images = np.array(tmp_images)
    tmp_image_names = np.array(tmp_image_names)

    images, image_names = convert_data_to_tensors(tmp_images, tmp_image_names)
    predictions = models[FLAGS.model](images, NUM_CLASS=1, is_training=False)
    ckpt_dir = logdir
    sv = tf.train.Supervisor(logdir=ckpt_dir)
    with sv.managed_session() as sess:

        preds, image_names = sess.run([predictions, image_names])
    export_result([str(img_name) for img_name in list(np.concatenate(image_names))],
                  list(np.concatenate(preds[0])), "regression_%s" % FLAGS.label)
