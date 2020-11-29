import numpy as np


def preprocess_gray(patches):
    """
    Parameters:
        patches (ndarray of shape (N, n_rows, n_cols, ch))
    Returns:
        patches (ndarray of shape (N, n_rows, n_cols, 1))
    """
    n_images, n_rows, n_cols, _ = patches.shape
    patches = np.mean(patches, axis=3, dtype='float')
    patches = patches[:, np.newaxis].transpose(0, 2, 3, 1)
    patches -= np.mean(patches)
    return patches


def preprocess_train(images_train, labels_train, images_val, labels_val, images_test, labels_test):
    _, n_rows, n_cols, ch = images_train.shape

    # 1. convert to gray images
    x_train = np.mean(images_train, axis=3, dtype='float')
    x_train = x_train[:, np.newaxis].transpose(0, 2, 3, 1)

    x_val = np.mean(images_val, axis=3, dtype='float')
    x_val = x_val[:, np.newaxis].transpose(0, 2, 3, 1)

    x_test = np.mean(images_test, axis=3, dtype='float')
    x_test = x_test[:, np.newaxis].transpose(0, 2, 3, 1)

    # convert class vectors to binary class matrices
    y_train = labels_train.astype('int')
    y_val = labels_val.astype('int')
    y_test = labels_test.astype('int')

    # 2. calc mean value
    mean_value = np.mean(x_train)
    x_train -= mean_value
    x_val -= mean_value
    x_test -= mean_value

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
