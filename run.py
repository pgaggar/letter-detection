from detect import *
import utils.preprocess as preprocess
import time

if __name__ == "__main__":

    tic = time.process_time()

    model = model_custom
    preprocessor = preprocess.preprocess_gray
    model_input_shape = model_input_shape_custom
    detect_images(model, preprocessor, model_input_shape)

    toc = time.process_time()

    time_diff = toc - tic

    # print("Total prediction time:", time_diff)
    # print("Average prediction time: ", time_diff/9)
