import pandas as pd
import os
import random
import seaborn as sns
import numpy as np
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras import optimizers
# import scikitplot as skplt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from skimage import color
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random as rn
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import itertools
from mlxtend.plotting import plot_confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from keras.callbacks import ReduceLROnPlateau
import requests
import itertools


# Maybe apply data regularization methods
# data augmentation has less added value later (32x and 64x are similar)
# need to make sure that the augmentated images still look like the original ex. cat or dog


flower_dict= {0:"phlox",1: "rose", 2:'calendula', 3:'iris',4:'leucanthemum maximum',
              5:'bellflower',6:'viola',7:'rudbeckia laciniata',8:'peony', 9:'aquilegia'}

fish_dict = {0: "Black Sea Sprat", 1: "Gilt-Head Bream", 2: "Hourse Mackerel", 3: "Red Mullet",
            4: "Red Sea Bream", 5: "Shrimp", 6: "Striped Red Mullet", 7: "Trout", 8: "Sea Bass"}

IMG_SIZE=32
LABELS = 9   # should have a different value for fish
FLOWER_LABELS = 10


def read_flowers(input):
    # return images
    labels = []
    file = open(input, "r")
    lines = file.readlines()
    # labels = prevLabels
    for i in range(1, 211):
        strippedLine = lines[i].strip('\n')
        label = strippedLine.split(",")[1]
        labels.append(label)
    return labels

def read_fish_images(directory, prev_list):
    images = prev_list

    root = "Fish_Dataset/Fish_Dataset/"

    for i in range(1, 1001):
        imgName = "{0}/{1}/{2}/{3}.png".format(root, directory, directory, str(i).zfill(5))
        image = cv2.imread(imgName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        #adds each of the images to their correponding arrays
        images.append(image)
    return images

def get_fish_images():
    images = []
    labels = []

    for j in range(0, 9):
        for i in range(1, 1001):
            labels.append(j)

    images = read_fish_images("Black Sea Sprat", images)
    images = read_fish_images("Gilt-Head Bream", images)
    images = read_fish_images("Hourse Mackerel", images)
    images = read_fish_images("Red Mullet", images)
    images = read_fish_images("Red Sea Bream", images)
    images = read_fish_images("Shrimp", images)
    images = read_fish_images("Striped Red Mullet", images)
    images = read_fish_images("Trout", images)
    images = read_fish_images("Sea Bass", images)

    return images, labels


def get_flower_Images():

    images = []
    labels = []

    labels = read_flowers("../flower_images/flower_images/flower_labels.csv")
    root = "../flower_images/flower_images/"
    for i in range(1, 211):
        imgName = "{0}{1}.png".format(root, str(i).zfill(4))
        image = cv2.imread(imgName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        #adds each of the images to their correponding arrays
        images.append(image)


    return images,labels

def augment_fish(images):
    original_len = len(images)
    augmented_images = images

    for i in range(original_len):
        transform = A.Compose([
                        A.OneOf([
                            A.GaussNoise(p=0.2),
                            A.RandomRotate90(0.5),
                            A.OpticalDistortion(p=0.3)
                        ]),
                        A.OneOf([
                            A.MotionBlur(p=0.2),
                            A.MedianBlur(blur_limit=3, p=0.3)
                        ]),
                        A.OneOf([
                            A.HorizontalFlip(p=0.2),
                            A.VerticalFlip(p=0.2)
                        ])
                    ])
        augmented_images[i] = transform(image = images[i])('image')

    return images

def augment_Images(images, labels):

    augmented_images = images
    new_labels = labels

    for j in range(9):
        for i in range (211):
            transform = A.Compose([
            A.RandomRotate90(),
            A.OneOf([
                A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.2),
                A.GaussNoise(p=0.2),
                A.Transpose(p=0.2),
                A.Blur(blur_limit=3)
            ]),
            A.OneOf([
                A.RandomBrightnessContrast(),
                A.OpticalDistortion()
            ], p = 0.3)
            # A.HueSaturationValue(p=0.3),
            ])
            rn.seed(42)

            augmented_image = transform(image=images[i])['image']
            augmented_images.append(augmented_image)
            new_labels.append(labels[i])
    plt.imshow(augmented_images[45])
    plt.show()

    return augmented_images, new_labels


# creates and splits the data set and label arrays
def create_dataset(images, labels, numlabels):

    # splits data into training and test set
    # maybe 60 20 20
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2)

    # converts all to numpy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # reshapes the images
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255

    # one hot encoding
    Y_train = to_categorical(y_train, num_classes = numlabels)
    Y_test = to_categorical(y_test, num_classes = numlabels)

    return X_train, X_test, Y_train, Y_test


def build_model(height,width, depth,classes):
    shape_input=(height,width,depth)

        #creating the input layer

    model=Sequential()
    # 1st conv layer, with 3*3 kernel, and 32 kernels with relu activation, and batch normalization
    model.add(Conv2D(64, (3, 3), padding='same',input_shape=shape_input, activation = "relu"))
    # model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1))

    # 2nd conv layer
    model.add(Conv2D(64, (3, 3), padding='same', activation = "relu"))
    # model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(128,(3,3),padding='same', activation = "relu"))
    # model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(pool_size=(2,2)))


    model.add(Conv2D(128,(3,3),padding='same', activation = "relu"))
    # model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.1))

    model.add(Conv2D(128,(3,3),padding='same', activation = "relu"))
    # model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1))

    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    # model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.1))

    model.add(Dense(classes, activation = "softmax"))
    # model.add(Activation('softmax'))

    model.summary()


    return model

def createModel(shape, numlabels):

    model = Sequential()

    model.add(Conv2D(filters = 25, kernel_size = (5, 5),padding = 'Same',
                 activation ='relu', input_shape = shape))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    # model.add(Dropout(0.3))

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
    model.add(MaxPool2D(pool_size=(1,1), strides=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(16, activation = "relu"))
    model.add(Dropout(0.4))
    model.add(Dense(numlabels, activation = "softmax"))

    model.summary()

    return model

def flowers():

   # gets the image from the root directory
   images, labels = get_flower_Images()

   # images, labels = augment_Images(images, labels)

   # creates the dataset by seperating training and tesitng data
   X_train,X_test, Y_train, Y_test = create_dataset(images, labels, FLOWER_LABELS)

   X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2)

   # creates the architecture
   model = build_model(IMG_SIZE, IMG_SIZE, 1, FLOWER_LABELS)

   # compiles and fits the data
   model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy', 'Recall', 'Precision'])

   # backpropogates and fits model
   history = model.fit(X_train, Y_train, batch_size = 10, epochs = 10, validation_data = (X_val, Y_val))

   filepath = 'saved_nonaug_flowers_rayhaan'
   save_model(model, filepath)

   loss_train = history.history['loss']
   loss_val = history.history['val_loss']
   epochs = range(1,11)
   plt.plot(epochs, loss_train, 'g', label='Training loss')
   plt.plot(epochs, loss_val, 'b', label='validation loss')
   plt.title('Training and Validation loss')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.legend()
   plt.show()

   loss_train = history.history['accuracy']
   loss_val = history.history['val_accuracy']
   epochs = range(1,11)
   plt.plot(epochs, loss_train, 'g', label='Training accuracy')
   plt.plot(epochs, loss_val, 'b', label='validation accuracy')
   plt.title('Training and Validation accuracy')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.show()

   y_pred = model.predict(X_test)
   y_test, y_pred = fromBinaryToCategorical(Y_test, y_pred)
   conf = confusion_matrix(y_test, y_pred)

   fig, ax = plot_confusion_matrix(conf_mat = np.array(conf), colorbar = True, show_absolute = False, show_normed = True)

   plt.show()



def fromBinaryToCategorical(y_test, y_pred):
    y_test_nonbin = []
    y_pred_nonbin = []
    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] == 1:
                y_test_nonbin.append(j)
                break

    for i in range(len(y_test)):
        largest = max(y_pred[i])
        for j in range(len(y_pred[i])):
            if y_pred[i][j] == largest:
                y_pred_nonbin.append(j)
                break

    return y_test_nonbin, y_pred_nonbin



def fish():

   # gets the image from the root directory
   images, labels = get_fish_images()

   # creates the dataset by seperating training and tesitng data
   X_train, X_test,Y_train, Y_test = create_dataset(images, labels, LABELS)

   X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2)

   # creates the architecture
   # model = createModel((IMG_SIZE, IMG_SIZE, 1), LABELS)
   model = createModel((IMG_SIZE, IMG_SIZE, 1), LABELS)

   # compiles and fits the data
   model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy', 'Recall', 'Precision'])

   # backpropogates
   history = model.fit(X_train, Y_train, batch_size = 10, epochs = 100, validation_data = (X_val, Y_val))

   filepath = 'saved_modelBrian_fish'
   save_model(model, filepath)

   loss_train = history.history['loss']
   loss_val = history.history['val_loss']
   epochs = range(1,101)
   plt.plot(epochs, loss_train, 'g', label='Training loss')
   plt.plot(epochs, loss_val, 'b', label='validation loss')
   plt.title('Training and Validation loss')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.legend()
   plt.show()

   loss_train = history.history['accuracy']
   loss_val = history.history['val_accuracy']
   epochs = range(1,101)
   plt.plot(epochs, loss_train, 'g', label='Training accuracy')
   plt.plot(epochs, loss_val, 'b', label='validation accuracy')
   plt.title('Training and Validation accuracy')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.show()

   y_pred = model.predict(X_test)
   y_test, y_pred = fromBinaryToCategorical(Y_test, y_pred)
   conf = confusion_matrix(y_test, y_pred)

   fig, ax = plot_confusion_matrix(conf_mat = np.array(conf), colorbar = True, show_absolute = False, show_normed = True)
  # plot_confusion_matrix(y_true = y_test, y_pred, normalize = True)

   plt.show()

   # Print the confusion matrix using Matplotlib
   # fig, ax = plt.subplots(figsize=(7.5, 7.5))
   # ax.matshow(conf, cmap=plt.cm.Blues, alpha=0.3)
   # for i in range(conf_matrix.shape[0]):
   #     for j in range(conf_matrix.shape[1]):
   #         ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
   #
   # plt.xlabel('Predictions', fontsize=18)
   # plt.ylabel('Actuals', fontsize=18)
   # plt.title('Confusion Matrix', fontsize=18)
   # plt.show()




def flower_Testing(filepath):


    use_samples = []
    file = open("../flower_images/flower_images/flower_labels.csv", "r")
    lines = file.readlines()
    model = load_model(filepath, compile = True)
    for i in range(1, 5):
        new_arr = []
        randImage = random.randint(1, 210)
        strippedLine = lines[randImage].strip('\n')
        label = strippedLine.split(",")[1]
        new_arr.append(randImage)
        new_arr.append(int(label))
        use_samples.append(new_arr)

    samples_to_predict = []

    # Generate plots for samples
    for i in range(len(use_samples)):
  # Generate a plot
        root = "../flower_images/flower_images"

        imgName = "{0}/{1}.png".format(root, str(use_samples[i][0]).zfill(4))
        the_image = cv2.imread(imgName)
        reshaped_image = cv2.cvtColor(the_image, cv2.COLOR_BGR2GRAY)
        reshaped_image = cv2.resize(reshaped_image, (IMG_SIZE, IMG_SIZE))
        reshaped_image = reshaped_image.reshape(32, 32, 1)
        reshaped_image = reshaped_image.astype("float32")
        reshaped_image /= 255
        transform = A.Compose([

        A.OneOf([
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(0.6),
        ], 0.9),
        A.OneOf([            A.GaussNoise(p=0.2),
                    A.Transpose(p=0.2),
                    A.Blur(blur_limit=3, p=0.2)]),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.OpticalDistortion()
        ], p = 0.3)
        # A.HueSaturationValue(p=0.3),
        ])
        rn.seed(42)

        augmented_image = transform(image=reshaped_image)['image']
        plt.imshow(the_image)
        plt.show()
  # Add sample to array for prediction
        samples_to_predict.append(augmented_image)
#
    # Convert into Numpy array
    samples_to_predict = np.array(samples_to_predict)

    # Generate predictions for samples
    predictions = model.predict(samples_to_predict)
    print(predictions)

    # # Generate arg maxes for predictions
    classes = np.argmax(predictions, axis = 1)
    #
    y_test = []
    y_pred = []
    print("\nReal labels: ", end = "")
    for i in range(len(use_samples)):
        y_test.append(flower_dict[use_samples[i][1]])
        print(str(flower_dict[use_samples[i][1]]) + "   ", end = '')
    print('\n')

    for i in range(len(classes)):
        y_pred.append(flower_dict[classes[i]])

    print("Predicted Classification: ", end = "")
    for i in range(len(classes)):
        print(str(flower_dict[classes[i]]) + "   ", end='')
    print("\n")

   # ========== CONFUSION MATRIX ==========
   #  conf = confusion_matrix(y_test,  y_pred)
   #
   #  fig, ax = plot_confusion_matrix(conf_mat = np.array(conf), colorbar = True, show_absolute = False, show_normed = True)
   # # plot_confusion_matrix(y_true = y_test, y_pred, normalize = True)
   #  # plt.xticks(arrange(len(flower_dict.keys()), flower_dict.keys(), rotation=45))
   #  plt.title("Flower Test Confusion Matrix")
   #  plt.show()




def testing(filepath):

    use_samples = []

    model = load_model(filepath, compile = True)

    # UNCOMMENT IF NEED RANDOM *****
    for i in range(5):
        new_arr = []
        randClass = random.randint(0, 8)
        randNum = random.randint(1, 1001)
        fish_class = fish_dict[randClass]
        new_arr.append(randNum)
        new_arr.append(fish_class)
        use_samples.append(new_arr)

    samples_to_predict = []

    # Generate plots for samples
    for i in range(len(use_samples)):
        root = "Fish_Dataset/Fish_Dataset"

        imgName = "{0}/{1}/{2}/{3}.png".format(root, use_samples[i][1], use_samples[i][1],  str(use_samples[i][0]).zfill(5))
        the_image = cv2.imread(imgName)
        reshaped_image = cv2.cvtColor(the_image, cv2.COLOR_BGR2GRAY)
        reshaped_image = cv2.resize(reshaped_image, (IMG_SIZE, IMG_SIZE))
        reshaped_image = reshaped_image.reshape(32, 32, 1)
        reshaped_image = reshaped_image.astype("float32")
        reshaped_image /= 255
        transform = A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(0.6),
        ]),
        A.OneOf([            A.GaussNoise(p=0.2),
                    A.Transpose(p=0.2),
                    A.Blur(blur_limit=3)]),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.OpticalDistortion()
        ], p = 0.3)
        # A.HueSaturationValue(p=0.3),
        ])
        rn.seed(42)

        augmented_image = transform(image=reshaped_image)['image']
        plt.imshow(the_image)
        plt.show()
        plt.imshow(augmented_image)
        plt.show()

  # Add sample to array for prediction
        samples_to_predict.append(augmented_image)
#
    # Convert into Numpy array
    samples_to_predict = np.array(samples_to_predict)

    # Generate predictions for samples
    predictions = model.predict(samples_to_predict)
    print(predictions)

    # # Generate arg maxes for predictions
    classes = np.argmax(predictions, axis = 1)
    #
    y_test = []
    y_pred = []

    print("Real Classification: ", end = "")
    for i in range(len(use_samples)):
       y_test.append(use_samples[i][1])
       print(str(use_samples[i][1]) + "   ", end = '')
    print('\n')

    for i in range(len(classes)):
        y_pred.append(fish_dict[classes[i]])

    print("Predicted Classification: ", end = "")
    for i in range(len(classes)):
        print(str(fish_dict[classes[i]]) + "   ", end='')
    print("\n")

   # ========== CONFUSION MATRIX ==========
    # UNCOMMENT THIS FOR Random TESTING

   #  conf = confusion_matrix(y_test, y_pred)
   #
   #  fig, ax = plot_confusion_matrix(conf_mat = np.array(conf), colorbar = True, show_absolute = False, show_normed = True)
   # # plot_confusion_matrix(y_true = y_test, y_pred, normalize = True)
   #
   #  plt.show()

def main():
    testing("ModelBrianFish")
    # fish()
    # testing("ModelRayFish")
    # flower_Testing("saved_nonaug_flowers_rayhaan")
    # flower_Testing("saved_aug_flowers_rayhaan")


if __name__== "__main__":
    main()
