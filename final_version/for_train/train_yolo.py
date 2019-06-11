import time

from construct_network import  yolo_network,custom_loss
from preprocessing import parse_annotation, BatchGenerator
from utils import WeightReader,normalize
from keras.optimizers import Adam
import numpy as np
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from yolov2 import get_model

train_image_folder = '../../keras_yolo2/data/coco/images/train2014/'
train_annot_folder = '../../keras_yolo2/data/coco/pascal_format/train/'
valid_image_folder = '../../keras_yolo2/data/coco/images/val2014/'
valid_annot_folder = '../../keras_yolo2/data/coco/pascal_format/val/'
wt_path = '../../keras_yolo2/data/coco/yolov2.weights'

weight_reader = WeightReader(wt_path)
#model = yolo_network()
#yolo = Yolov2(input_shape = (416,416,3))
model = get_model()
weight_reader.reset()
nb_conv = 23
LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 13 , 13
BOX              = 5
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

BATCH_SIZE       = 2
TRUE_BOX_BUFFER  = 50

for i in range(1, nb_conv + 1):
    conv_layer = model.get_layer('conv_' + str(i))

    if i < nb_conv:
        norm_layer = model.get_layer('norm_' + str(i))

        size = np.prod(norm_layer.get_weights()[0].shape)

        beta = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean = weight_reader.read_bytes(size)
        var = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])

    if len(conv_layer.get_weights()) > 1:
        bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2, 3, 1, 0])
        conv_layer.set_weights([kernel])

#layer   = model.layers[-4] # the last convolutional layer
#weights = layer.get_weights()

#new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
#new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)
#
#layer.set_weights([new_kernel, new_bias])

generator_config = {
    'IMAGE_H'         : IMAGE_H,
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'CLASS'           : len(LABELS),
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : 50,
}


train_imgs, seen_train_labels = parse_annotation(train_annot_folder, train_image_folder, labels=LABELS)
train_batch = BatchGenerator(train_imgs, generator_config, norm=normalize)

valid_imgs, seen_valid_labels = parse_annotation(valid_annot_folder, valid_image_folder, labels=LABELS)
valid_batch = BatchGenerator(valid_imgs, generator_config, norm=normalize, jitter=False)

#early_stop = EarlyStopping(monitor='val_loss',
#                           min_delta=0.001,
#                           patience=3,
#                           mode='min',
#                           verbose=1)

checkpoint = ModelCheckpoint('./data/weights_coco.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=True,
                             mode='min',
                             period=5)

optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss=custom_loss, optimizer=optimizer)

print('len of train batch: ', len(train_batch))
print('len of val batch: ', len(valid_batch))

model.fit_generator(generator        = train_batch,
                    steps_per_epoch  = len(train_batch),
                    epochs           = 6,
                    verbose          = 1,
                    validation_data  = valid_batch,
                    validation_steps = len(valid_batch),
                    callbacks        = [checkpoint],
                    max_queue_size   = 3)

print('done fitting model')
