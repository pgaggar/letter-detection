from tensorflow import keras


# custom model
def build_model(filters=32, kernel_size=3, input_shape=(32, 32, 1), classes=10):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters, kernel_size=kernel_size, input_shape=input_shape, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(filters, kernel_size, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(filters, kernel_size))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(filters, kernel_size, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(filters * 4, kernel_size))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(filters * 4, kernel_size, padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(classes, activation='softmax'))

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())
    return model
