#!/usr/bin/env python
# coding: utf-8

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Reshape, Input, Lambda
from keras.layers import LeakyReLU
from keras.preprocessing import image as Im
import numpy as np
from utils import *
from config import anchors, class_names
from keras.layers.merge import concatenate
import tensorflow as tf
from matplotlib.pyplot import imread, imshow

from keras import backend as K
K.set_learning_phase(1) #set learning phase

import cv2
from tqdm import tqdm

def load_model(input_shape):
    input = Input(input_shape)

    def space_to_depth_x2(x):
        return tf.space_to_depth(x, block_size=2)

    # Layer 1
    x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
    skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)

    out = Conv2D(5 * (4 + 1 + 80), (1,1), strides=(1,1), padding='same', name='conv_23')(x)

    model = Model(inputs = input, outputs = out)
    model.summary()
    return model

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) 
 
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold=0.5)
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):

    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs


    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=score_threshold)
    
    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes=max_boxes, iou_threshold=iou_threshold)
        
    return scores, boxes, classes

def get_spaced_colors(n):
    max_value = 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(100, max_value, interval)]
    
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def preprocess_test_image(img_path, model_image_size=(416,416)):
    original_image = Im.load_img(img_path)
    test_image = Im.load_img(img_path, target_size = model_image_size)
    test_data = Im.img_to_array(test_image)
    test_data /= 255.0
    test_data = np.expand_dims(test_data, axis = 0)
    return original_image, test_data

model = load_model(input_shape = (416,416,3))
model.load_weights('weights/weights.h5')
anchors = np.array(anchors)

#image_test, image_data = preprocess_test_image("./frame.jpg", model_image_size = (416, 416))
#
#image_shape = image_test.size
#image_shape = tuple(float(i) for i in reversed(image_test.size))
#
#yolo_outputs = yolo_head(model.output, anchors, len(class_names))
#scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
#
#sess = K.get_session()
#out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={
#    model.input: image_data
#})
#
#print('Found {} boxes for {}'.format(len(out_boxes), "test_person.jpg"))
#
#colors = get_spaced_colors(len(class_names))
#draw_boxes(image_test, out_scores, out_boxes, out_classes, class_names, colors)
#image_test.save(os.path.join("output_images", "test_person.jpg"), quality=90)
#output_image = imread(os.path.join("output_images", "test_person.jpg"))
#
#imshow(output_image)

video_inp = './conference.mp4'
video_out = './conference_bbox.avi'

video_reader = cv2.VideoCapture(video_inp)

nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

#video_writer = cv2.VideoWriter(video_out,
#                               cv2.VideoWriter_fourcc(*'XVID'), 
#                               50.0, 
#                               (frame_w, frame_h))
video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               10.0, 
                               (frame_w, frame_h))

for i in tqdm(range(nb_frames)):
    ret, image = video_reader.read()
    
    #input_image = cv2.resize(image, (416, 416))
    #input_image = input_image / 255.
    #input_image = input_image[:,:,::-1]
    #input_image = np.expand_dims(input_image, 0)

    #netout = model.predict([input_image, dummy_array])

    #boxes = decode_netout(netout[0], 
    #                      obj_threshold=0.3,
    #                      nms_threshold=NMS_THRESHOLD,
    #                      anchors=ANCHORS, 
    #                      nb_class=CLASS)
    #image = draw_boxes(image, boxes, labels=LABELS)

    cv2.imwrite("./frame.jpg", image)

    image_test, image_data = preprocess_test_image('./frame.jpg', model_image_size = (416, 416))
    
    image_shape = image_test.size
    image_shape = tuple(float(i) for i in reversed(image_test.size))
    
    yolo_outputs = yolo_head(model.output, anchors, len(class_names))
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
    
    sess = K.get_session()
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={
        model.input: image_data
    })
    
    print('Found {} boxes for {}'.format(len(out_boxes), "test_person.jpg"))
    
    colors = get_spaced_colors(len(class_names))
    draw_boxes(image_test, out_scores, out_boxes, out_classes, class_names, colors)
    image_test.save(os.path.join("output_images", "test_person.jpg"), quality=90)
    output_image = imread(os.path.join("output_images", "test_person.jpg"))

    im2 = output_image.copy()
    im2[:, :, 0] = output_image[:, :, 2]
    im2[:, :, 2] = output_image[:, :, 0]

    #video_writer.write(np.uint8(output_image))
    video_writer.write(im2)
    
video_reader.release()
video_writer.release()
