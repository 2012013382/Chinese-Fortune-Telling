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
#Please modify the path for your own image fold.
parser.add_argument('--image_path', type=str, default="data/13_images")
#If batch is "True", it means it will process images in a fold, or it will only process one image.
parser.add_argument('--batch', type=str, default="False")
#If save_features is "True", it means it will save image features as .npy in 'tmp_data', or it will only return it.
parser.add_argument('--save_results', type=str, default="False")
args = parser.parse_args()
print(args)

def feature_extract(image_path):
    with tf.Graph().as_default():
        image = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
        logits, features = predict_model(image, is_training=False)
        variables_to_use = slim.get_variables_to_restore()
        variables_restorer = tf.train.Saver(variables_to_use)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            variables_restorer.restore(sess, BEST_MODEL_PATH)
            img = read_one_image(image_path, args.batch)
            if args.batch == "True":
                score, feature = sess.run([logits, features],  feed_dict={image: img})
                #print(feature.shape)
                if args.save_results == "True":
                    np.save("tmp_data/image_features.npy", feature)
                    np.save("tmp_data/image_scores.npy", score)
                return feature, score
            else:
                score, feature = sess.run([logits, features],  feed_dict={image: img})
                #print(feature.shape)
                return feature, score