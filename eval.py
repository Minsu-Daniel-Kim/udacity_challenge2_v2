import tensorflow as tf

import tensorflow.contrib.slim as slim
from input import inputs
from model.lenet import lenet
from model.vgg16 import vgg16
from model.cifarnet import cifarnet
from model.alexnet import alexnet_v2
from model.resnet_v1 import resnet_v1_50





flags = tf.app.flags
flags.DEFINE_string('train_dir', './data',
                    'Directory with the training data.')
flags.DEFINE_integer('batch_size', 200, 'Batch size.')
flags.DEFINE_integer('num_batches', 100, 'Num of batches to evaluate.')
flags.DEFINE_string('model', None, "Model name")
# flags.DEFINE_string('log_dir', './log/eval',
#                     'Directory where to log data.')
# flags.DEFINE_string('checkpoint_dir', './log/train',
#                     'Directory with the model checkpoint data.')
FLAGS = flags.FLAGS

log_dir = "./log/%s/eval" % FLAGS.model
checkpoint_dir = "./log/%s/train" % FLAGS.model

models = {

    'lenet': lenet,
    'alexnet': alexnet_v2,
    'resnet': resnet_v1_50,
    'cifarnet_v1': cifarnet,
    'vgg': vgg16
}

def main(train_dir, batch_size, num_batches, log_dir, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = log_dir

    images, labels = inputs(train_dir, False, batch_size, num_batches,one_hot_labels=False)
    predictions = models[FLAGS.model](images)
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        print(predictions)



    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
        "streaming_mse": slim.metrics.streaming_mean_squared_error(predictions, labels),
    })

    tf.scalar_summary('mse', metrics_to_values['streaming_mse'])

    # Evaluate every 30 seconds
    slim.evaluation.evaluation_loop(
        '',
        checkpoint_dir,
        log_dir,
        num_evals=num_batches,
        eval_op=metrics_to_updates['streaming_mse'],
        summary_op=tf.merge_all_summaries(),
        eval_interval_secs=30)


if __name__=='__main__':
    main(FLAGS.train_dir,
         FLAGS.batch_size,
         FLAGS.num_batches,
         log_dir,
         checkpoint_dir)
