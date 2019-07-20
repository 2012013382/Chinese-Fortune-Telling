import numpy as np
import cv2
from os import walk
from os.path import join
import tensorflow as tf
from scipy.misc import imread, imresize
from PIL import Image
DATA_PATH = 'data/'
TMP_DATA = 'tmp_data/'
IMAGE_PATH = DATA_PATH + 'Images/'
TRAIN_SCORES_FILE_PATH = DATA_PATH + 'train_test_files/split_of_60%training and 40%testing/train.txt'
TEST_SCORES_FILE_PATH = DATA_PATH + 'train_test_files/split_of_60%training and 40%testing/test.txt'

CLASS_NUM = 1
#For resnet50
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3

#Load data(images and scores)
def prepare_data():
    #Obtain image names
    file_names = (walk(IMAGE_PATH)).next()[2]
    file_num = len(file_names)

    #Load train/validation set names and scores
    train_image_names = []
    train_image_scores = []
    
    val_image_names = []
    val_image_scores = []
    with open(TRAIN_SCORES_FILE_PATH, 'r') as f:
        idx = 0
        for line in f:
            try:
                if idx % 8 == 0:
                   row = line.split()
                   val_image_names.append(IMAGE_PATH + row[0])
                   val_image_scores.append(float(row[1]))
                else:
                   row = line.split()
                   train_image_names.append(IMAGE_PATH + row[0])
                   train_image_scores.append(float(row[1]))
                idx += 1
            except Exception as e:
                print('Wrong train set path!')
                pass

    #Load test set names and scores
    test_image_names = []
    test_image_scores = []
    with open(TEST_SCORES_FILE_PATH, 'r') as f:
        for line in f:
            try:
                row = line.split()
                test_image_names.append(IMAGE_PATH + row[0])
                test_image_scores.append(float(row[1]))
            except Exception as e:
                print('Wrong test set path!')
                pass
    
    data = {'train_image_names': train_image_names,
         'train_image_scores': train_image_scores,
         'val_image_names': val_image_names,
         'val_image_scores': val_image_scores,
         'test_image_names': test_image_names,
         'test_image_scores': test_image_scores }
    return data
#Load images for tf queue
def load_data(image_names_list, image_scores_list, epoch_num, preprocess_fn, batch_size, is_training):
    image_name, image_score = tf.train.slice_input_producer([image_names_list, image_scores_list],
                                                            shuffle=True,
                                                            num_epochs=epoch_num)
    img_bytes = tf.read_file(image_name)
    image = tf.image.decode_jpeg(img_bytes, channels=3)

    #is training?
    if is_training:
        processed_image = preprocess_fn(image, IMG_HEIGHT, IMG_WIDTH)
    else:
        processed_image = preprocess_fn(image, IMG_HEIGHT, IMG_WIDTH)

    img, sco = tf.train.batch([processed_image, image_score], batch_size, dynamic_pad=True)
    return img, sco

def read_one_image(file_path, is_batch="False"):
    if is_batch == "True":
       file_names = (walk(file_path)).next()[2]
       file_num = len(file_names)
       images = np.zeros((file_num, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), dtype=np.float32)
       for i in range(len(file_names)):
          img = Image.open(join(file_path, file_names[i])).convert("RGB")
          img = np.array(img.resize((IMG_HEIGHT, IMG_WIDTH)))
          images[i, :, :, :] = img
    else:
       img = Image.open(file_path).convert("RGB")
       img = np.array(img.resize((IMG_HEIGHT, IMG_WIDTH)))
       images = np.reshape(img, [1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
    return images
