import tensorflow as tf

import tensorflow.contrib.slim as slim
from util.input import inputs
from model.lenet import lenet
from model.vgg16 import vgg16
from model.driveNet import drivenet
from model.cifarnet import cifarnet
from model.alexnet import alexnet_v2
from model.resnet_v1 import resnet_v1_50


# slim = tf.contrib.slim
NUM_CLASS=5


flags = tf.app.flags
flags.DEFINE_string('train_dir', None, 'Directory with the training data.')
flags.DEFINE_string('prediction_type', None, 'Regression / Classification')
flags.DEFINE_string('model', None, "Model name")
flags.DEFINE_integer('batch_size', 300, 'Batch size.')
flags.DEFINE_integer('num_batches', None, 'Num of batches to train (epochs).')
flags.DEFINE_float('learning_rate', None, 'Specify learning rate')
# flags.DEFINE_string('log_dir', './log/eval',
#                     'Directory where to log data.')
# flags.DEFINE_string('checkpoint_dir', './log/train',
#                     'Directory with the model checkpoint data.')
FLAGS = flags.FLAGS

log_dir = "./log_%s/%s/eval" % (FLAGS.prediction_type, FLAGS.model)
checkpoint_dir = "./log_%s/%s/train" % (FLAGS.prediction_type, FLAGS.model)

models = {

    'lenet': lenet,
    'alexnet': alexnet_v2,
    'resnet': resnet_v1_50,
    'cifarnet': cifarnet,
    'drivenet': drivenet,
    'vgg': vgg16
}

def main(train_dir, batch_size, num_batches, logdir, prediction_type, checkpoint_dir):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = tf.Session(config=config)


    if prediction_type == "regression":
        print('regression is called')
        images, angles = inputs(train_dir,
                                False,
                                batch_size,
                                num_batches,
                                one_hot_labels=False)


        predictions, end_points = models[FLAGS.model](images, NUM_CLASS=1, is_training=False)
        metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
            "mse": slim.metrics.streaming_mean_squared_error(predictions, angles),

        })

        tf.scalar_summary('loss_mse', metrics_to_values['mse'])


        slim.evaluation.evaluation_loop(
            '',
            checkpoint_dir,
            logdir,
            num_evals=num_batches,
            eval_op=metrics_to_updates['mse'],
            summary_op=tf.merge_all_summaries(),
            eval_interval_secs=30)
    else:
        print('classification is called')
        images, labels = inputs(train_dir,
                                False,
                                batch_size,
                                num_batches,
                                one_hot_labels=True)

        predictions, end_points = models[FLAGS.model](images, NUM_CLASS=NUM_CLASS, is_training=False)
        predictions = tf.argmax(predictions, 1)
        predictions = tf.one_hot(predictions, NUM_CLASS, dtype=tf.int32)
        predictions = tf.cast(predictions, tf.int32)

        # with tf.Session() as sess:
        #     init_op = tf.initialize_all_variables()

        metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
            "accuracy": slim.metrics.streaming_accuracy(predictions, labels),
            "precision": slim.metrics.streaming_precision(predictions, labels),
        })

        tf.scalar_summary('accuracy', metrics_to_values['accuracy'])
        tf.scalar_summary('precision', metrics_to_values['precision'])

        # Evaluate every 30 seconds
        slim.evaluation.evaluation_loop(
            '',
            checkpoint_dir,
            logdir,
            num_evals=num_batches,
            eval_op=[metrics_to_updates['accuracy'], metrics_to_updates['precision']],
            summary_op=tf.merge_all_summaries(),
            eval_interval_secs=30)


if __name__=='__main__':
    main(FLAGS.train_dir,
         FLAGS.batch_size,
         FLAGS.num_batches,
         log_dir,
         FLAGS.prediction_type,
         checkpoint_dir)
