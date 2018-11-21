from tqdm import tqdm
import cv2                          # working with, mainly resizing, images
import numpy as np
from random import shuffle
import os


# need one hot array value for labels for ML algorithms
# first value = 'catness', second value = 'dogness'
def label_img(img):
    # e.g. dog.93.png hence [-3]
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1, 0]
    elif word_label == 'dog': return [0, 1]


def create_train_data(TRAIN_DIR, IMG_SIZE):
    training_data = []      # create train data with labels
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data(TEST_DIR, IMG_SIZE):
    testing_data = []       # kaggle test data (no labels)
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    np.save('test_data.npy', testing_data)
    return testing_data
