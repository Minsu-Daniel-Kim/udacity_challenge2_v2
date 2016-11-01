import tensorflow as tf

import tensorflow.contrib.slim as slim
from input import inputs
from model.lenet import lenet
from model.vgg16 import vgg16
from model.cifarnet import cifarnet
from model.alexnet import alexnet_v2
from model.resnet_v1 import resnet_v1_50
import pandas as pd
import argparse
import numpy as np
from input import inputs, submission_read_and_decode


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--model')

args = parser.parse_args()
print(args.model)

models = {

    'lenet': lenet,
    'alexnet': alexnet_v2,
    'resnet': resnet_v1_50,
    'cifarnet': cifarnet,
    'vgg': vgg16
}

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
def main():
    with tf.Graph().as_default():
        # pass
        image, label = inputs("./data", "test", 10000, None, one_hot_labels=False)
        images, labels = convert_data_to_tensors(image, label)

        predictions = models[args.model](images)
        ckpt_dir = 'log/%s/train' % args.model
        sv = tf.train.Supervisor(logdir=ckpt_dir)
        with sv.managed_session() as sess:

            preds, labels = sess.run([predictions, labels])

            print(get_mse(preds, labels))




if __name__=='__main__':
    main()
