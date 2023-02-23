# Face Expression Classification
This code is designed to train a neural network model to classify facial expressions using the FER2013 dataset. The dataset contains 48x48 pixel grayscale images of faces, with seven different expressions: angry, disgust, fear, happy, sad, surprise, and neutral.

#By: Maryam Mahmoudi
Date: 26 Nov 2022

The code is divided into two main parts:

# Loading and preprocessing the data: 
This is done using the load_data function, which loads the images, resizes them, normalizes their pixel values, and applies face detection using the MTCNN library. The resulting images and labels are returned as x_train, y_train, x_test, and y_test.
# Defining and training the neural network model: 
This is done using the network function, which defines a simple CNN architecture using Keras, and the fit function, which trains the model using the training data and evaluates it on the test data. The resulting model is saved as face_detection.h5.
The code also includes a data augmentation step using the "tf.keras.preprocessing.image.ImageDataGenerator" class, which applies various transformations to the images to increase the size of the training dataset and reduce overfitting.

Finally, the code includes a function called show_results that plots the accuracy and loss curves for the training and validation datasets during training.

# Requirements
To run the code, you need to have the following libraries installed: 
os, mtcnn, sklearn, tensorflow, cv2, numpy, matplotlib. 
You also need to download the FER2013 dataset and place it in a directory called FER2013 in the same directory as the code. The directory structure should be as follows:
# Dataset: 
FER2013
|-- train
|   |-- angry
|   |-- disgust
|   |-- fear
|   |-- happy
|   |-- neutral
|   |-- sad
|   |-- surprise
|-- test
|   |-- angry
|   |-- disgust
|   |-- fear
|   |-- happy
|   |-- neutral
|   |-- sad
|   |-- surprise
|-- README
|-- fer2013.bib
|-- fer2013.csv

# Note 
the code sets the TensorFlow log level to 3 to suppress warning messages. If you want to see the messages, you can remove the line os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3".
