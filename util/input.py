import os
import numpy as np
import tensorflow as tf
import itertools
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist

# TRAIN_FILE = 'train.tfrecords'
# VALIDATION_FILE = 'validation.tfrecords'
# TEST_FILE = 'augmented_test.tfrecords'
# TEST_CONTRAST_FILE = 'augmented_test_contrast.tfrecords'
SUBMISSION_FILE = 'submission_test.tfrecords'
CHANNEL = 3
HEIGHT = 60
WEIGHT = 80
TRAIN_DIR = "data"
NUM_CLASSES = 5

def label0():
    return tf.constant(0)
def label1():
    return tf.constant(1)
def label2():
    return tf.constant(2)
def label3():
    return tf.constant(3)
def label4():
    return tf.constant(4)
def default():
    return tf.constant(-1)


def convert_data_to_tensors(x, y):
    inputs = tf.constant(x)
    inputs.set_shape([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])

    outputs = tf.constant(y)
    outputs.set_shape([y.shape[0], 1])
    return inputs, outputs

def read_and_decode_submission(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            # 'angle': tf.FixedLenFeature([], tf.float32),
            # 'label': tf.FixedLenFeature([], tf.int64),
            # 'label': tf.FixedLenFeature([], tf.int64),
            'img_name': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # img_name = tf.decode_raw(features['img_name'], tf.uint8)
    # img_name = features['img_name']
    img_name = tf.decode_raw(features['img_name'], tf.int64)
    # img_name = tf.cast(features['img_name'], tf.uint8)

    image.set_shape([HEIGHT * WEIGHT * CHANNEL])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.reshape(image, [HEIGHT, WEIGHT, CHANNEL])
    # image = tf.image.resize_images(image, tf.pack(tf.constant(60, dtype=tf.int32), tf.constant(80, dtype=tf.int32)))

    # preprocessing
    # image = tf.image.rgb_to_grayscale(image)
    # image = tf.image.per_image_standardization(image)
    # image = tf.image.per_image_whitening(image)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    # angle = tf.cast(features['angle'], tf.float32)
    # label = tf.cast(features['label'], tf.uint8)
    # label = features['label']
    # label = tf.cond(tf.less(angle, tf.constant(-0.03966)), label0,
    #         lambda: tf.cond(tf.less(angle, tf.constant(-0.00698)), label1,
    #                         lambda: tf.cond(tf.less(angle, tf.constant(0.01266)), label2,
    #                                         lambda: tf.cond(tf.less(angle, tf.constant(0.04189)), label3, label4))))

    # angle = tf.reshape(angle, [1])


    # label = None
    # label = tf.cond(tf.equal(label, 0), lambda: tf.convert_to_tensor(0), lambda: tf.convert_to_tensor(1))
    return image, img_name
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'angle': tf.FixedLenFeature([], tf.float32),
            # 'label': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            # 'img_name': tf.FixedLenFeature([], tf.int64)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # img_name = tf.decode_raw(features['img_name'], tf.uint8)
    img_name = None
    # img_name = tf.cast(features['img_name'], tf.uint8)
    image.set_shape([HEIGHT * WEIGHT * CHANNEL])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.reshape(image, [HEIGHT, WEIGHT, CHANNEL])
    # image = tf.image.resize_images(image, tf.pack(tf.constant(60, dtype=tf.int32), tf.constant(80, dtype=tf.int32)))

    # preprocessing
    # image = tf.image.rgb_to_grayscale(image)
    # image = tf.image.per_image_standardization(image)
    # image = tf.image.per_image_whitening(image)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    angle = tf.cast(features['angle'], tf.float32)
    # label = tf.cast(features['label'], tf.uint8)
    label = features['label']
    # label = tf.cond(tf.less(angle, tf.constant(-0.03966)), label0,
    #         lambda: tf.cond(tf.less(angle, tf.constant(-0.00698)), label1,
    #                         lambda: tf.cond(tf.less(angle, tf.constant(0.01266)), label2,
    #                                         lambda: tf.cond(tf.less(angle, tf.constant(0.04189)), label3, label4))))

    angle = tf.reshape(angle, [1])


    # label = None
    # label = tf.cond(tf.equal(label, 0), lambda: tf.convert_to_tensor(0), lambda: tf.convert_to_tensor(1))
    return image, angle, label, img_name

def aggregate_dataset(direction, dataset, label, dataset_subset='all'):
    """
        direction = ['left', 'center', 'right']
        dataset = ['original', 'flip', 'contrast', 'flip_contrast']
        label = [0]
        dataset_subset = 'all' or ['train','val']

        return {}
    """
    train_lst = []
    test_lst = []
    validation_lst = []

    def get_subset_lst(subset):
        subste_list = []
        for a, b in list(itertools.product(direction, dataset)):
            folders = [folder for folder in os.listdir(TRAIN_DIR) if folder.startswith("udacity")]
            for folder in folders:
                # print(folder)
                try:
                    if label is -1:
                        print("ho")

                        dir = TRAIN_DIR + "/" + folder + "/%s/%s" % (a, b)
                        labels = [file for file in os.listdir(dir) if file.startswith("label")]
                        # print(labels)
                        train = []
                        for item in labels:

                            train += [dir + "/" + item +'/'+ file2 for file2 in os.listdir(dir + '/' + item) if subset in file2]
                    else:

                        dir = TRAIN_DIR + "/" + folder + "/%s/%s/label%s" % (a, b, label)
                        train = [dir + "/" + file for file in os.listdir(dir) if subset in file]
                    subste_list += train

                except:

                    pass
        return [subset, subste_list]

    train_lst = np.array(train_lst)
    test_lst = np.array(test_lst)
    validation_lst = np.array(validation_lst)

    if dataset_subset is 'all':
        dataset_subset = ['train', 'test', 'validation']

    total_dict = {}
    for subset in dataset_subset:
        total_dict[get_subset_lst(subset)[0]] = get_subset_lst(subset)[1]

    return total_dict

def inputs(train_dir, train, batch_size, num_epochs, label=None, one_hot_labels=False):

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
            print('works here')
            image, img_name = read_and_decode_submission(filename_queue)
            # image, angle, label, img_name = read_and_decode(filename_queue)
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

    elif train is ("test" or "contrast"):


        # test_filenames = aggregate_dataset(['center'], ['original'], ['test'])['test']

        test_filenames = "data/submission_test.tfrecords"

        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(
                test_filenames, num_epochs=num_epochs)

            image, angle, label, img_name = read_and_decode(filename_queue)

            images = []
            angles = []

            with tf.Session() as sess:
                # Start populating the filename queue.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                for i in range(batch_size):
                    # Retrieve a single instance:
                    a, b = sess.run([image, angle])
                    images.append(a)
                    angles.append(b)

                coord.request_stop()
                coord.join(threads)

        return np.array(images), np.array(angles)

    else:
        print('label : ', label)
        if train:

            filenames = aggregate_dataset(['center'], ['original'], label, ['train'])['train']

        else:
            filenames = aggregate_dataset(['center'], ['original'], label, ['validation'])['validation']

        print('input data: ', filenames)
        with tf.name_scope('input'):
            filename_queue = tf.train.string_input_producer(
                filenames, num_epochs=num_epochs)

            image, angle, label, img_name = read_and_decode(filename_queue)
            if one_hot_labels:
                print("one_hot_labels")
                label = tf.one_hot(label, NUM_CLASSES, dtype=tf.int32)
                images, sparse_labels = tf.train.shuffle_batch(
                    [image, label], batch_size=batch_size, num_threads=4,
                    capacity=1000 + 3 * batch_size,
                    # Ensures a minimum amount of shuffling of examples.
                    min_after_dequeue=1000)
                return images, sparse_labels
            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            else:

                images, sparse_angles = tf.train.shuffle_batch(
                    [image, angle], batch_size=batch_size, num_threads=4,
                    capacity=1000 + 3 * batch_size,
                    # Ensures a minimum amount of shuffling of examples.
                    min_after_dequeue=1000)
                return images, sparse_angles