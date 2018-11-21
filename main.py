'''
Rework and improvement of Sentdex's dogs vs cats CNN classifier tutorial to better understand an ML pipeline:
https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/

I have:
 - Improved accuracy through CNN architecture and hyperparameter modification and inclusion of residual blocks
(Ref: https://arxiv.org/pdf/1512.03385.pdf)
 - Data Augmentation
 - Separated to callable functions for easier hyperparameter optimisation, debugging, and more readable code
 - Added custom image input function
 - Added commands while running to eliminate repeated image processing/model training

L2 Regularization + Dropout combined to prevent overfitting and improve model generalisability
(Ref: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

Model attains ~90& accuracy on validation data and a log loss of ~0.32 on Kaggle's test data.
Results analysed with tensorboard.

dependencies = numpy, os, opencv, random, tflearn, tensorflow
'''
import numpy as np         # dealing with arrays
import os                  # dealing with directories
import preprocessing as pp # custom image preprocessing functions
import CNN_model           # custom CNN model functions
from tflearn.data_augmentation import ImageAugmentation

# Directories containing train and test images
TRAIN_DIR = 'C:\\Users\squir\Dropbox\ML Projects\Kaggle\Dogs vs Cats Redux\\train'
TEST_DIR = 'C:\\Users\squir\Dropbox\ML Projects\Kaggle\Dogs vs Cats Redux\\test'
# Model Parameters
IMG_SIZE = 100           # resize image to this height and width
lr = 0.0001              # learning rate
epochs = 25              # number of times model sees full data

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(lr, 'integrated')

# Ask user to load or process
# For first time need to process but subsequently can load data
# UNLESS IMG_SIZE is changed
print('Load pre-existing preprocessed data for training (L) or preprocess data (P)?')
decision1 = input()
if decision1 == 'P':
    train_data = pp.create_train_data(TRAIN_DIR=TRAIN_DIR, IMG_SIZE=IMG_SIZE)
    test_data = pp.process_test_data(TEST_DIR=TEST_DIR, IMG_SIZE=IMG_SIZE)
elif decision1 == 'L':
    if os.path.exists('train_data.npy'):
        train_data = np.load('train_data.npy')
        test_data = np.load('test_data.npy')
    else:
        raise Exception('No preprocessed data exists in path, please preprocess some.')
else:
    raise Exception('Please retry and type L or P')

'''
25000 train images are now:
IMG_SIZE*IMG_SIZE grayscale attached to one hot class label indicating cat [1,0] or dog [0,1] and ordered randomly
'''

# split data into train (24500) and validation (500) data
valid_data = train_data[24500:]
train_data = train_data[:24500]
# derive image and label data from new data sets
train_data_imgs = [item[0] for item in train_data]
train_data_lbls = [item[1] for item in train_data]
valid_data_imgs = [item[0] for item in valid_data]
valid_data_lbls = [item[1] for item in valid_data]
# create arrays for us in models
X_train = np.array(train_data_imgs).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y_train = train_data_lbls
x_valid = np.array(valid_data_imgs).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_valid = valid_data_lbls

# Image Augmentation (flip along vertical axis and random rotations (max 25 degrees)
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

'''             Model execution (training or loading)                '''
# model = CNN_model.cnn(img_size=IMG_SIZE, lr=lr)                                       # basic CNN
# model = CNN_model.resnet(img_size=IMG_SIZE, lr=lr, n=2)                               # simple resnet
model = CNN_model.conv_res_integrated(img_size=IMG_SIZE, lr=lr, n=1, img_aug=img_aug)   # integrated CNN with res layers

# Ask user to load or train new model
# For first time need to process but subsequently can load old model
# UNLESS any parameters are changed
print('Would you like to load pre-existing trained model (L) or train a new one (T)?')
decision2 = input()
if decision2 == 'T':
    model.fit(X_train,
              Y_train,
              n_epoch=epochs,
              validation_set=(x_valid, y_valid),
              snapshot_step=500,
              show_metric=True,
              run_id=MODEL_NAME)
elif decision2 == 'L':
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')
    else:
        raise Exception('No model exists in path, please create and save one.')
else:
    raise Exception('Please retry and type Y or N')

model.save(MODEL_NAME)
