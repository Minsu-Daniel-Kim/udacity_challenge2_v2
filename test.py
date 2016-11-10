import tensorflow as tf

import tensorflow.contrib.slim as slim
from util.input import inputs
from model.lenet import lenet
from model.vgg16 import vgg16
from model.cifarnet import cifarnet
from model.alexnet import alexnet_v2
from model.resnet_v1 import resnet_v1_50

import matplotlib.pyplot as plt


import pandas as pd
import argparse
import numpy as np
# from input import inputs, submission_read_and_decode

flags = tf.app.flags
flags.DEFINE_string('train_dir', None, 'Directory with the training data.')
flags.DEFINE_string('prediction_type', None, 'Regression / Classification')
flags.DEFINE_string('model', None, "Model name")
flags.DEFINE_integer('batch_size', 300, 'Batch size.')
flags.DEFINE_integer('label', -1, 'label')
flags.DEFINE_integer('num_batches', None, 'Num of batches to train (epochs).')
flags.DEFINE_float('learning_rate', None, 'Specify learning rate')
# flags.DEFINE_float('momentum', None, 'Specify momentum')
FLAGS = flags.FLAGS
NUM_CLASS=5

if FLAGS.label == -1:

    log_dir = "./log_%s/%s/train" % (FLAGS.prediction_type, FLAGS.model)
else:
    log_dir = "./log_%s/%s_%s/train" % (FLAGS.prediction_type, FLAGS.model, FLAGS.label)

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

def get_mse(pred, label):

    length = label.shape[0]

    # mse = ((A - B) ** 2).mean(axis=ax)
    mse = ((pred - label) ** 2).mean()

    return mse
def main(train_dir, batch_size, num_batches, logdir, prediction_type, label):
    with tf.Graph().as_default():
        # pass
        num_data = 6601
        image, image_name = inputs(train_dir, "submission", num_data, None, one_hot_labels=False)
        images, image_names = convert_data_to_tensors(image, image_name)

        predictions = models[FLAGS.model](images, NUM_CLASS=NUM_CLASS, is_training=False)
        ckpt_dir = logdir
        sv = tf.train.Supervisor(logdir=ckpt_dir)
        with sv.managed_session() as sess:

            preds, image_names = sess.run([predictions, image_names])
            # preds = tf.argmax(preds, 1)
            predictions = np.array(preds)
            # img_names = np.array(image_names)

            # print(predictions[1])
            print(predictions[0])
            print(np.argmax(predictions[0], 1))
            predictions = np.argmax(predictions[0], 1)
            print(predictions)
            # print(img_names)

        export_result([str(img_name) for img_name in list(np.concatenate(image_names))],
                      list(predictions), "labels2")



if __name__=='__main__':
    main(FLAGS.train_dir, FLAGS.batch_size, FLAGS.num_batches, log_dir, FLAGS.prediction_type, FLAGS.label)
