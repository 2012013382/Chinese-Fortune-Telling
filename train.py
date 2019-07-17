from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import argparse
from nets import nets_factory
from preprocessing import preprocessing_factory
import time
from data_processing import prepare_data
from data_processing import load_data
from model import predict_model
from data_processing import CLASS_NUM
from validate import validate_model
SAVE_MODEL_PATH = './tmp_data/train_model.ckpt'
BEST_MODEL_PATH = './tmp_data/best_model.ckpt'
RES_v1_50_MODEL_PATH = './tmp_data/resnet_v1_50.ckpt'
TRAIN_LOG_DIR = 'log/'
LR_DECAY_FACTORY = 0.1
EPOCHS_PER_LR_DECAY = 15
MOVING_AV_DECAY = 0.9999

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='resnet_v1_50')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch_num', type=int, default=None)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()
print(args)
#train model
def train_model():
    data = prepare_data()
    #build graph
    with tf.Graph().as_default():
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(args.model_name, is_training=True)
        processed_image, score = load_data(data['train_image_names'],
                                              data['train_image_scores'],
                                              args.epoch_num,
                                              image_preprocessing_fn,
                                              args.batch_size,
                                              True)
        score = tf.reshape(score, [-1, 1])
        print(score.shape)
        logits = predict_model(processed_image, is_training=True)
        print(logits.shape)
        variables_to_restore = slim.get_variables_to_restore(exclude=['resnet_v1_50/logits'])
        variables_restorer = tf.train.Saver(variables_to_restore)
        
        #Loss
        with tf.name_scope('ls'):
            #MSE loss
            loss = tf.sqrt(tf.reduce_mean(tf.square(logits - score)))
            tf.summary.scalar('loss', loss)

        current_epoch = tf.Variable(0, trainable=False)
        decay_step = EPOCHS_PER_LR_DECAY * len(data['train_image_names']) // args.batch_size
        learning_rate = tf.train.exponential_decay(args.lr, current_epoch, decay_step, LR_DECAY_FACTORY, staircase=True)
        
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
        #opt = tf.train.AdamOptimizer(learning_rate)
        optimizer = slim.learning.create_train_op(loss, opt, global_step = current_epoch)
 
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            summary_writer = tf.summary.FileWriter(TRAIN_LOG_DIR, sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            variables_restorer.restore(sess, RES_v1_50_MODEL_PATH)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sum_ls = 0.0
            batch_num = len(data['train_image_scores']) // args.batch_size
            val_step = 0
            best_val_ls = 100.0
            try:
                while not coord.should_stop():
                    _, ls, step, summary = sess.run([optimizer, loss, current_epoch, summary_op])
                    sum_ls += ls

                    if step % 50 == 0:
                        print("Epoch %d, loss %f"%(
                              step / batch_num + 1, ls))
                        summary_writer.add_summary(summary, step)
                    if step % batch_num == 0 and step != 0:
                        print("Epoch %d, mean loss %f"%(step / batch_num + 1, sum_ls / batch_num))
                        sum_ls = 0.0
                        saver.save(sess, SAVE_MODEL_PATH)
                        val_ls = validate_model()
                        if val_ls < best_val_ls:
                           best_val_ls = val_ls
                           saver.save(sess, BEST_MODEL_PATH)
                        print('best val loss %f'%(best_val_ls))
            except tf.errors.OutOfRangeError:
                saver.save(sess, SAVE_MODEL_PATH)
            finally:
                coord.request_stop()
            coord.join(threads)


train_model()


