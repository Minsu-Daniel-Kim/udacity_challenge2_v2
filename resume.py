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
flags.DEFINE_string('model', None, "Model name")
flags.DEFINE_integer('batch_size', 300, 'Batch size.')
flags.DEFINE_integer('num_batches', None, 'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', None, 'Directory with the log data.')
FLAGS = flags.FLAGS

log_dir = "./log/%s/train" % FLAGS.model

models = {

    'lenet': lenet,
    'alexnet': alexnet_v2,
    'resnet': resnet_v1_50,
    'cifarnet': cifarnet,
    'vgg': vgg16
}
def get_init_fn():
    return slim.assign_from_checkpoint_fn(ckpt_dir='log/%s/train' % FLAGS.model)

def main(train_dir, batch_size, num_batches, log_dir):
    images, labels = inputs(train_dir,
                            True,
                            batch_size,
                            num_batches,
                            one_hot_labels=False)
    predictions = models[FLAGS.model](images)
    ###



    ckpt_dir = 'log/%s/train' % FLAGS.model
    sv = tf.train.Supervisor(logdir=ckpt_dir)
    with sv.managed_session() as sess:
        # pass
        slim.losses.mean_squared_error(predictions, labels)
        # total_loss = slim.losses.get_total_loss()
        # tf.scalar_summary('loss', total_loss)
        #
        # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
        # train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)
        #
        slim.learning.train(train_op,
                            log_dir,
                            init_fn=et_init_fn(),
                            save_summaries_secs=20,
                            number_of_steps=100000)


        slim.learning.train(
            train_op,
            logdir=train_dir,
            init_fn=get_init_fn(),
            save_summaries_secs=20,
            number_of_steps=100000)


if __name__ == '__main__':
    main(FLAGS.train_dir, FLAGS.batch_size, FLAGS.num_batches, log_dir)