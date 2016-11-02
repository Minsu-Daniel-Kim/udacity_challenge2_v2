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

flags = tf.app.flags


parser = argparse.ArgumentParser(description='Process some integers.')


parser.add_argument('--model')
parser.add_argument('--train_dir')

args = parser.parse_args()
print(args.model)

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
def main():
    with tf.Graph().as_default():
        # pass
        num_data = 5000
        image, label = inputs(args.train_dir, "contrast", num_data, None, one_hot_labels=False)
        images, labels = convert_data_to_tensors(image, label)

        ## contrast version
        # c_image, c_label = inputs(args.train_dir, "contrast", num_data, None, one_hot_labels=False)
        # print(c_image)
        # c_images, c_labels = convert_data_to_tensors(c_image, c_label)

        ###

        predictions = models[args.model](images)

        ckpt_dir = 'log/%s/train' % args.model
        sv = tf.train.Supervisor(logdir=ckpt_dir)
        with sv.managed_session() as sess:

            preds, labels = sess.run([predictions, labels])




        # c_predictions = models[args.model](c_images)
        # with sv.managed_session() as sess:
        #     c_preds, c_labels = sess.run([c_predictions, c_labels])
        #
        # print("augmented data pure mse: ", get_mse(preds, labels))
        # print("augmented data avg mse: ", get_mse((preds + c_labels) / 2, labels))
        #
            predictions = np.array(preds)
            img_name = np.array(labels)

        export_result([str(img_name) for img_name in list(np.concatenate(img_name))],
                      list(np.concatenate(predictions)), "original_contrast")



if __name__=='__main__':
    main()
