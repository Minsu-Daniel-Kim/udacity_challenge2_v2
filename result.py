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
from input import inputs


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--model')

args = parser.parse_args()
print(args.model)

def export_result(frame_id_lst, steering_angle_lst):
    dict = {'frame_id': frame_id_lst, 'steering_angle': steering_angle_lst}
    pd.DataFrame(dict).to_csv("result/%s_output.csv" % args.model,index=False)


def main():
    with tf.Graph().as_default():
        images, labels = inputs("./data", True, 1000, 100, one_hot_labels=False)
        predictions = lenet(images)
        # Create the model structure. (Parameters will be loaded below.)
        #     predictions, end_points = regression_model(inputs, is_training=False)
        ckpt_dir = 'log/lenet/train'
        # Make a session which restores the old parameters from a checkpoint.
        sv = tf.train.Supervisor(logdir=ckpt_dir)
        with sv.managed_session() as sess:
            predictions = sess.run([predictions])
            print(predictions)


if __name__=='__main__':
    main()