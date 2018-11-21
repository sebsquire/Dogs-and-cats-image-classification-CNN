'''     Create folder with image(s) to be processed and specoify it as CUSTOM_DIR below
        Name the images as a number 1-4 and include maximum of 4 images in folder
        e.g. 3.png
        ONLY use this if a model has already been created                                   '''
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from main import IMG_SIZE, MODEL_NAME, model, lr


CUSTOM_DIR = 'C:\\Users\squir\Dropbox\ML Projects\Kaggle\Dogs vs Cats Redux\CustomImageInput'


def pp_custom(image_dir, img_size):
    custom_testing_data = []       # kaggle test data (no labels)
    for img in os.listdir(image_dir):
        path = os.path.join(image_dir, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        custom_testing_data.append([np.array(img), img_num])
    return custom_testing_data


custom_test_data = pp_custom(image_dir=CUSTOM_DIR, img_size=IMG_SIZE)

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
else:
    raise Exception('No model exists in path, please create and save one.')

fig = plt.figure()

for num, data in enumerate(custom_test_data):
    # cat: [1,0]
    # dog: [0,1]
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(2, 2, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    # model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
