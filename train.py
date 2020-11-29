from tensorflow import keras
import utils.hdf5_read as hdf5_read
import utils.preprocess as preprocess
import utils.model_builder as mb
import os
import numpy as np

DIR = 'dataset'
MODEL_DIR = 'model'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# training the model
def train_model(x_train, y_train, x_val, y_val, x_test, y_test, log_dir, classes):
    model = mb.build_model(classes=classes)

    # callback for the training process
    save_model = keras.callbacks.ModelCheckpoint(log_dir + '/weights.hdf5', monitor='val_accuracy', mode='max',
                                                 verbose=0,
                                                 save_best_only=True, save_weights_only=False)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0,
                                                   mode='max')

    # train model
    model.fit(x_train, y_train,
              epochs=50,
              verbose=1,
              batch_size=128,
              validation_data=(x_val, y_val),
              shuffle=True,
              callbacks=[early_stopping, save_model])

    # calculate and store test set performance on the model with best validation error
    print("Calculating performance on test set...")
    model = keras.models.load_model(log_dir + "/weights.hdf5")
    res = model.predict(x_test)
    y_pred = np.argmax(res, axis=1)
    score = sum(np.where(y_pred == y_test, 1, 0)) / len(y_pred)
    print("Test Accuracy:", score)


if __name__ == "__main__":
    pre_trained = False

    (images_train, labels_train), (images_val, labels_val), (images_test, labels_test) = hdf5_read.load_chars_data(DIR)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess.preprocess_train(images_train,
                                                                                       labels_train,
                                                                                       images_val,
                                                                                       labels_val,
                                                                                       images_test,
                                                                                       labels_test)
    print("Train image shape is {}, and Validation image shape is {}".format(x_train.shape,
                                                                             x_val.shape))

    train_model(x_train, y_train, x_val, y_val, x_test, y_test,
                log_dir=MODEL_DIR, classes=63)
