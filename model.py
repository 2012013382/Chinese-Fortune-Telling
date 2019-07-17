from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from nets import resnet_v1

def predict_model(processed_image, is_training=True):
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=1e-4)):
        _, end_points = resnet_v1.resnet_v1_50(inputs=processed_image, num_classes=1, is_training=is_training)
    logits = end_points['resnet_v1_50/spatial_squeeze']
    return logits
