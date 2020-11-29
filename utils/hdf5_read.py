import numpy as np
import h5py
from sklearn.model_selection import train_test_split


# load char data from the specified folder
def load_chars_data(path):
    with h5py.File(path + '/dataset.hdf5', 'r') as f:
        X = f['images'][:]
        y = f['labels'][:]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)

    print(y_train.shape, y_val.shape, y_test.shape)
    print(x_train.shape, x_val.shape, x_test.shape)
    print(np.unique(y_train))

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
