import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
TEST_FILE = 'augmented_test.tfrecords'
TEST_CONTRAST_FILE = 'augmented_test_contrast.tfrecords'
SUBMISSION_FILE = 'submission_test.tfrecords'
CHANNEL = 3
HEIGHT = 60
WEIGHT = 80

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.float32),
            'img_name': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    img_name = tf.decode_raw(features['img_name'], tf.uint8)

    image.set_shape([HEIGHT * WEIGHT * CHANNEL])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.reshape(image, [HEIGHT, WEIGHT, CHANNEL])

    # preprocessing
    tf.image.per_image_whitening(image)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.float32)
    label = tf.reshape(label, [1])

    return image, label, img_name

def inputs(train_dir, train, batch_size, num_epochs, one_hot_labels=False):
    """Reads input data num_epochs times.
    Args:
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
        train forever.
    Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
        * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
        Note that an tf.train.QueueRunner is added to the graph, which
        must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None

    if train is "submission":
        filename = os.path.join(train_dir, SUBMISSION_FILE)

        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(
                [filename], num_epochs=num_epochs)

            # Even when reading in multiple threads, share the filename
            # queue.

            image, label, img_name = read_and_decode(filename_queue)

            images = []
            img_names = []
            with tf.Session() as sess:
                # Start populating the filename queue.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                for i in range(batch_size):
                    # Retrieve a single instance:
                    a, b = sess.run([image, img_name])
                    images.append(a)
                    img_names.append(b)

                coord.request_stop()
                coord.join(threads)
                print('done!!')

        return np.array(images), np.array(img_names)

    elif train is "test" or "contrast":

        filename = os.path.join(train_dir, TEST_FILE if train is 'test' else TEST_CONTRAST_FILE)

        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(
                [filename], num_epochs=num_epochs)

            image, label, img_name = read_and_decode(filename_queue)

            images = []
            labels = []

            with tf.Session() as sess:
                # Start populating the filename queue.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                for i in range(batch_size):
                    # Retrieve a single instance:
                    a, b = sess.run([image, label])
                    images.append(a)
                    labels.append(b)

                coord.request_stop()
                coord.join(threads)
        return np.array(images), np.array(labels)

    else:
        filename = os.path.join(train_dir, TRAIN_FILE if train else VALIDATION_FILE)

        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(
                [filename], num_epochs=num_epochs)

            image, label, img_name = read_and_decode(filename_queue)
            # if one_hot_labels:
            #     label = tf.one_hot(label, mnist.NUM_CLASSES, dtype=tf.int32)

            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            images, sparse_labels = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size, num_threads=4,
                capacity=1000 + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)

        return images, sparse_labels
