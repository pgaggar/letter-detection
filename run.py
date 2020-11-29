import cv2

import detector.classifier as cls
import detector.detect as detector
import detector.region_proposal as rp
import utils.preprocess as preprocess
import os
from pathlib import Path
import argparse

model = "model/weights.hdf5"
model_input_shape = (32, 32, 1)

base_path = Path(__file__).parent
INPUT_DIR = 'images_input'
OUTPUT_DIR = 'images_output'

SAVE_OUTPUT = True
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def detect_images(input_model, preprocessor, input_shape, input_file):
    classifier = cls.CnnClassifier(input_model, preprocessor, input_shape)

    digit_spotter = detector.Detector(classifier, rp.MSER())

    # detect images
    if input_file is None:
        input_dir = INPUT_DIR
        img_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    else:
        img_files = [input_file]

    for i, img_file in enumerate(img_files):
        if not os.path.exists(img_file):
            print("Image not found")
            return
        img = cv2.imread(img_file)
        output_image = digit_spotter.run(img, threshold=0.6, nms_threshold=0.3)

        if SAVE_OUTPUT:
            print('saving output: ', i)
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_file), output_image)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image",
                    default=None,
                    help="Path to the image folder")

    args = vars(ap.parse_args())

    input_file_path = args['image']
    preprocessor = preprocess.preprocess_gray
    detect_images(model, preprocessor, model_input_shape, input_file_path)
