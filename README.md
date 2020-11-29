# Letter Detection
This project demonstrates detection of letters from an image.

# Key Features:
1. Works on different fonts and colors of texts.
2. Works better than pytesseract on multi-colored text. (sample image A-Z_color.jpg)
3. Handles different types of noise.

# Code Overview:
Following are the different files/packages/directories and their uses:
1. detector - This is the key package, and contains the classifier, detector, and the region proposing logic
2. model - This contains the weights of the model obtained after training.
3. utils - This contains utility files for reading and preprocessing the images in the dataset.
4. images_input, images_output - These are the input and output directories, where the images are stored before
and after the model is run respectively.
5. run.py - This is the file that calls the methods to run the text recognition.
6. train.py - This file is for training the model on the dataset(dataset not provided 
as part of this repo due to size)

# DataSet
The dataset used for training the model is [Chars 74K dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/). Due to
the size of the dataset, it is not added into the repository. For training, it can be downloaded and added to `train_data` directory 
in the main `letter-detection` directory. Further train.py can be run to train the model. 
A model has already been trained and weights have been saved in the `model` directory.


# Steps to run:
1. Clone the repository.
2. Create a virtual environment and install dependencies using pipfile or pipfile.lock or requirements.txt
3. Place the target image in the images_input directory and Run the file `run.py`. Alternatively, provide the path to
the image as `--image=path/to/image` if you want the code to take file from another path.
4. The output and the image with bounding boxes is printed on console, and the image is then saved to the `images_output` directory.


