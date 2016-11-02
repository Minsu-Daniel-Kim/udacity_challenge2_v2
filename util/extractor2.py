import os
import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def make_pickle(imgs, img_names, type):
    img_label = {'images': imgs, 'labels': img_names}
    pickle.dump(img_label, open("data/%s.pkl" % type, "wb"))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def make_tfrecords(images, labels, type):

    num_examples = labels.shape[0]

    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    directory = "."

    filename = os.path.join(directory, type + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(len(labels)):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _float_feature(float(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())


def extract_image_label(folder, type):
    print("extracting from " + folder)
    csv_file = pd.read_csv("/Volumes/DANIEL/Challenge-2/" + type + "/%d/interpolated.csv" % int(folder),
                           usecols=['filename', 'angle', 'frame_id'])
    csv_file = csv_file[csv_file['frame_id'] == "center_camera"]

    dat_dir = "/Volumes/DANIEL/Challenge-2/" + type + "/%d/center/" % int(folder)

    images_list = [dat_dir + file for file in os.listdir(dat_dir) if not file.startswith("._")]
    images_name_list = [file for file in os.listdir(dat_dir) if not file.startswith("._")]

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        tmp_image_lst = []
        tmp_label_lst = []
        tmp_img_name_lst = []

        for image_name in images_name_list:
            image = cv2.imread(dat_dir + "/" + image_name)
            image = cv2.resize(image, (160, 120))

            # store original image
            tmp_image_lst.append(image)
            angle = csv_file['angle'][csv_file['filename'] == "center/" + image_name]
            tmp_label_lst.append(float(angle))
            tmp_img_name_lst.append(image_name)

            # store contract image
            contrast_image = image.copy()
            contrast_image = adjust_gamma(contrast_image, gamma=2)
            tmp_image_lst.append(contrast_image)
            tmp_label_lst.append(float(angle))
            tmp_img_name_lst.append("contrast_" + image_name)

            # store flipped image
            flipped_image = image.copy()
            flipped_image = cv2.flip(flipped_image, 1)
            tmp_image_lst.append(flipped_image)
            tmp_label_lst.append(float(angle) * -1)
            tmp_img_name_lst.append("flipped_" + image_name)

            # store flipped contract image
            flipped_contrast_image = flipped_image.copy()
            flipped_contrast_image = adjust_gamma(flipped_contrast_image, gamma=2)
            tmp_image_lst.append(flipped_contrast_image)
            tmp_label_lst.append(float(angle) * -1)
            tmp_img_name_lst.append("flipped_contrast_" + image_name)

    images = np.array(tmp_image_lst)
    labels = np.array(tmp_label_lst)
    img_names = np.array(tmp_img_name_lst)
    return images, labels, img_names


# type = 'Train'

# folders = [file for file in os.listdir("/Volumes/DANIEL/Challenge-2/" + type) if not file.startswith("._")]
# images = []
# labels = []
# names = []

# for folder in folders:
#     i, l, img_names = extract_image_label(folder, type)
#     images.append(i)
#     labels.append(l)
#     names.append(img_names)
# images = np.array(np.concatenate(images))
# labels = np.array(np.concatenate(labels))
# names = np.array(np.concatenate(names))

images, labels, img_names = extract_image_label()


X_train_val, X_test, y_train_val, y_test = train_test_split(images, labels, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

print("making test set....")
make_tfrecords(X_test, y_test, 'data/augmented_test')

print("making train set....")
make_tfrecords(X_train, y_train, 'data/augmented_train')

print("making validation set....")
make_tfrecords(X_val, y_val, 'data/augmented_validation')

# print("pickling....")
# make_pickle(X_test, y_test, 'augmented_test')
# make_pickle(X_val, y_val, 'augmented_validation')
# make_pickle(X_train, y_train, 'augmented_train')