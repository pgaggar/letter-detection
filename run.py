from detect import *
import utils.preprocess as preprocess
import argparse

model = "model/weights.hdf5"
model_input_shape = (32, 32, 1)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image",
                    default="images_input",
                    help="Path to the image folder")

    args = vars(ap.parse_args())

    input_dir = args['image']
    preprocessor = preprocess.preprocess_gray
    detect_images(input_dir, model, preprocessor, model_input_shape)
