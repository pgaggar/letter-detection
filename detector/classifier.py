from tensorflow import keras


class CnnClassifier:

    def __init__(self, model_file, preprocessor, input_shape=(32, 32, 1)):
        self._model = keras.models.load_model(model_file)
        self._preprocessor = preprocessor
        self.input_shape = input_shape

    def predict(self, patches):
        """
        patches (N, 32, 32, 1)
        
        probs (N, n_classes)
        """
        patches_preprocessed = self._preprocessor(patches)
        probs = self._model.predict(patches_preprocessed, verbose=0)
        return probs
