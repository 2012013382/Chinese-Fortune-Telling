from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import argparse
from model import predict_model
from data_processing import read_one_image, CLASS_NUM, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL
BEST_MODEL_PATH = './tmp_data/best_model.ckpt'

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default="data/1.jpg")
args = parser.parse_args()
print(args)

def feature_extract():
    with tf.Graph().as_default():
        image = tf.placeholder(tf.float32, [1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
        logits = predict_model(image, is_training=False)
        variables_to_use = slim.get_variables_to_restore()
        variables_restorer = tf.train.Saver(variables_to_use)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            variables_restorer.restore(sess, BEST_MODEL_PATH)
            img = read_one_image(args.image_path)
            score = sess.run(logits,  feed_dict={image: img})
            print(score[0][0])

feature_extract()
