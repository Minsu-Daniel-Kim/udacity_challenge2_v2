import tensorflow as tf
import tensorflow.contrib.slim as slim
from util.input import inputs
from model.lenet import lenet
from model.driveNet import drivenet
from model.vgg16 import vgg16
from model.cifarnet import cifarnet
from model.alexnet import alexnet_v2
from model.resnet_v1 import resnet_v1_50

flags = tf.app.flags
flags.DEFINE_string('train_dir', None, 'Directory with the training data.')
flags.DEFINE_string('prediction_type', None, 'Regression / Classification')
flags.DEFINE_string('model', None, "Model name")
flags.DEFINE_integer('batch_size', 300, 'Batch size.')
flags.DEFINE_integer('label', -1, 'label')
flags.DEFINE_integer('num_batches', None, 'Num of batches to train (epochs).')
flags.DEFINE_float('learning_rate', None, 'Specify learning rate')
flags.DEFINE_float('gpu', 1.0, 'Specify gpu')
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
    'drivenet': drivenet,
    'vgg': vgg16
}


def main(train_dir, batch_size, num_batches, logdir, prediction_type, label, gpu):

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu
    session = tf.Session(config=config)

    if prediction_type == "regression":
        print('regression is called')




        images, angles = inputs(train_dir, True, batch_size, num_batches, label, one_hot_labels=False)


        predictions, end_points = models[FLAGS.model](images, NUM_CLASS=1, is_training=True)
        slim.losses.mean_squared_error(predictions, angles)

        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('loss_mse', total_loss)

        # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

        slim.learning.train(train_op, logdir, number_of_steps=1000000, save_summaries_secs=60)
    else:

        print('classification is called')
        images, labels = inputs(train_dir, True, batch_size, num_batches, label, one_hot_labels=True)
        predictions, end_points = models[FLAGS.model](images, NUM_CLASS=NUM_CLASS, is_training=True)
        slim.losses.softmax_cross_entropy(predictions, labels)
        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('loss_cross_entropy', total_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

        slim.learning.train(train_op, logdir, number_of_steps=1000000, save_summaries_secs=60)


if __name__ == '__main__':
    main(FLAGS.train_dir, FLAGS.batch_size, FLAGS.num_batches, log_dir, FLAGS.prediction_type, FLAGS.label, FLAGS.gpu)
