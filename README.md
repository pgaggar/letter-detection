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
in the main `letter-detection` directory. Further `train.py` can be run to train the model. 
A model has already been trained and weights have been saved in the `model` directory.


# Steps to run:
1. Clone the repository.
2. Create a virtual environment and install dependencies using pipfile or pipfile.lock or requirements.txt
3. Place the target image in the images_input directory and Run the file `run.py`. Alternatively, provide the path to
the image as `--image=path/to/image` if you want the code to take file from another path.
4. The output and the image with bounding boxes is printed on console, and the image is then saved to the `images_output` directory.

#Steps to run (using Dockerfile):
1. Clone the repository.
2. Go to the letter-detection directory. Type the following command to build the docker image - `docker build -t letter-detection .`
3. To run one of the given images against the model (images in the `images_input` directory), run the command `docker run -t letter-detection:latest python run.py`.
4. To run any other image, the image must be mounted on the docker image. For that, 
run the command `docker run -v /local/path/to/image.jpg:/opt/program/images_input/image.jpg -t letter-detection:latest python run.py --image=/opt/program/images_input/A.jpg`
5. The text will be printed on the console, and the image with bounding box and label will be stored in `images_output` directory within the container.
6. Now obtain the container id by running `docker container ls -all`. Copy the container id.
6. To move the image from within container to outside the container, run the command 
`docker cp <container_id>:/opt/program/images_output/image.jpg /local/path/to/new_image.jpg`
7. You can now view the image in the path which will contain bounding box surrounding the text, and the text label which was printed.