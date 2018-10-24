# coding=utf-8

from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
from keras.applications import vgg19
from keras import backend as K

parser = argparse.ArgumentParser(description="Neural style transfer with Keras")
parser.add_argument('base_image_path', metavar='base', type=str, help='Path to the image to transform')
parser.add_argument('style_reference_image_path', metavar='ref', type=str, help='Path to the style reference image')
parser.add_argument('result_prefix', metavar='res_prefix', type=str, help='Prefix for the saved results')
parser.add_argument('--iter', type=int, default=10, required=False, help='迭代次数')
parser.add_argument('--content_weight', type=float, default=0.025, required=False, help='内容权重')
parser.add_argument('--style_weight', type=float, default=1.0, required=False, help='风格权重')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False, help='Total Variation weight')

args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix
iterations = args.iter

total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

width, height = load_img(base_image_path).size
img_n_rows = 400
img_n_cols = int(width * img_n_rows / height)


def pre_process_image(image_path):
    img = load_img(image_path, target_size=(img_n_rows, img_n_cols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_n_rows, img_n_cols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_n_rows, img_n_cols, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]


base_image = K.variable(pre_process_image(base_image_path))
style_reference_image = K.variable(pre_process_image(style_reference_image_path))

if K.image_data_format() == 'channel_first':
    combination_image = K.placeholder((1, 3, img_n_rows, img_n_cols))
else:
    combination_image = K.placeholder((1, img_n_rows, img_n_cols))

input_tensor = K.concatenate([base_image, style_reference_image, combination_image], axis=0)
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
print('Model Loaded')

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])


def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_n_rows * img_n_cols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def content_loss(base, combination):
    return K.sum(K.square(combination - base))
