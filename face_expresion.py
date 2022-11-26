import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import glob
from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import time

begin = time.time()
train_path = r"FER2013\train\\*\\*"
test_path = r"FER2013\test\\*\\*"

def face_detection(img):
    detector = MTCNN()
    try:
        face_elements = detector.detect_faces(img)[0]
        #print(face_elements)
        x, y, w, h = face_elements['box']
        img = img[x:x+w, y:y+h]
    except:
        pass
    return img

def load_data(path):
    data = []
    label = []
    # preprocessing =====> normalization, resize, flat
    for i ,address in enumerate(glob.glob(path)):
        #print(i,address)
        img = cv2.imread(address)
        img =  img/255.0
        #print(img)
        data.append(img)
        label_name = address.split('\\')[-2]
        #print(label)
        label.append(label_name)
        #print('info: {} out of 2600 processed'.format(i))     

    x = np.array(data)
    print(x.shape)
    labels = np.array(label)
    #print(labels)
    le = LabelBinarizer()
    y = le.fit_transform(labels)
    #labels = labels.reshape(1,labels.shape[0])
    #print(label.shape)
    end = time.time()
    print("Total time of face detection for all dataset is {} second which is around {} minute".format(end-begin,(end-begin)/60))
    return x, y

x_train, y_train = load_data(train_path)
x_test, y_test = load_data(test_path)
print("train data shape is {} and label shape is {} and test data shape is {} and test label shape is {} and".format(x_train.shape,y_train.shape,x_test.shape, y_test.shape))
#x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.25)

aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                                        width_shift_range=0.1,
                                                        height_shift_range=0.1,
                                                        shear_range=0.2,
                                                        zoom_range=0.2,
                                                        horizontal_flip=True,
                                                        fill_mode="nearest")

def network():
    net = tf.keras.models.Sequential([ 
                            tf.keras.layers.Conv2D(32,(3,3),  activation='relu', input_shape=(48,48,3)),
                            tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.MaxPool2D(),

                            tf.keras.layers.Conv2D(64,(3,3),  activation='relu'),
                            tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.MaxPool2D(),

                            tf.keras.layers.Flatten(),

                            tf.keras.layers.Dense(32, activation= 'relu'),
                            tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.Dense(7, activation= 'softmax')     ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.01, decay=0.0008)
    net.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])
    return net
def show_results(N):    # Show Output
    plt.plot(N.history['accuracy'], label='train accuracy')
    plt.plot(N.history['val_accuracy'], label='test accuracy')
    plt.plot(N.history['loss'], label='train loss')
    plt.plot(N.history['val_loss'], label='test loss')
    plt.xlabel("epochs")
    plt.ylabel("accuracy/100")
    plt.legend()
    plt.title("Face Expression classification")
    plt.show()
net = network()
print(net.summary())
begin_net = time.time()
N = net.fit(aug.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) // 32, validation_data = (x_test, y_test), epochs = 30)

#N = net.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size=32, epochs = 15)
end_net = time.time()
print("Total time of Network Training for all dataset is {} second which is around {} minute".format(end_net-begin_net,(end_net-begin_net)/60))
show_results(N)
net.save('face_detection.h5')


