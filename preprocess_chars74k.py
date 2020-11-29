import numpy as np
import os
import h5py
import cv2
import re


def extract_data(input_dir, output_loc):
    sz = (32, 32)

    filters = ['Bmp']
    filenames = []
    negative_images = []
    negative_images_gray = []
    negative_labels = []
    positive_images = []
    positive_images_gray = []
    positive_labels = []

    for path, dirs, files in os.walk(input_dir):
        # filtering out unwanted paths
        if sum(map(lambda f: f in path, filters)) == len(filters):
            filenames += list(map(lambda f: path + '/' + f, files))

    neg_label = 0

    for filename in filenames:
        if filename.endswith('.png'):
            img = cv2.imread(filename)
            img = cv2.resize(img, sz, interpolation=cv2.INTER_AREA)
            if 'Good' in filename:
                positive_images.append(img)
                positive_images_gray.append(np.average(img, axis=2))
                pos_label = re.match("(.*)img0(.*)-(.*).png", filename).groups()[1]
                positive_labels.append(int(pos_label))
            elif 'Bad' in filename:
                negative_images.append(img)
                negative_images_gray.append(np.average(img, axis=2))
                negative_labels.append(neg_label)

    images_gray = negative_images_gray + positive_images_gray
    images = negative_images + positive_images
    labels = negative_labels + positive_labels

    images_gray = np.asarray(images_gray)
    images = np.asarray(images)
    labels = np.asarray(labels)

    output_path = os.path.join(output_loc, 'dataset.hdf5')
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('images', data=images, shape=images.shape, dtype='float32',
                          compression="gzip")
        hf.create_dataset('images_gray', data=images_gray, shape=images_gray.shape, dtype='float32',
                          compression="gzip")
        hf.create_dataset('labels', data=labels, shape=labels.shape, dtype='int32',
                          compression="gzip")


INPUT_DIR = 'train_data'
OUTPUT_DIR = 'dataset'

if __name__ == "__main__":
    extract_data(INPUT_DIR, OUTPUT_DIR)
