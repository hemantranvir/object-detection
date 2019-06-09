import time

from keras.preprocessing import image as Im
import numpy as np
from utils import *
from config import anchors, class_names
from matplotlib.pyplot import imread, imshow

from keras import backend as K
K.set_learning_phase(1) #set learning phase

import cv2
from tqdm import tqdm

from yolov2 import Yolov2

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

def main():
    yolo = Yolov2(input_shape = (416,416,3))
    model = yolo.get_model()
    model.load_weights('weights/yolov2.h5')
    anchor = np.array(anchors)

    video_reader = cv2.VideoCapture(0)

    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    f = open('./no_of_people.txt', mode='w+')
    frame_no = 0
    
    video_inp = './conference.mp4'
    video_out = './conference_bbox.avi'
    
    video_reader = cv2.VideoCapture(video_inp)
    fps = video_reader.get(cv2.CAP_PROP_FPS)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    video_writer = cv2.VideoWriter(video_out,
                                   cv2.VideoWriter_fourcc(*'MPEG'), 
                                   fps, 
                                   (frame_w, frame_h))

    f = open('./no_of_people.txt', mode='w+')
    
    for i in tqdm(range(nb_frames)):
        ret, image = video_reader.read()
        
        cv2.imwrite("./frame.jpg", image)
    
        image_test, image_data = preprocess_test_image('./frame.jpg', model_image_size = (416, 416))
        
        image_shape = image_test.size
        image_shape = tuple(float(i) for i in reversed(image_test.size))
        
        yolo_outputs = yolo_head(model.output, anchor, len(class_names))
        scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
        
        sess = K.get_session()

        out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={
            model.input: image_data
        })
        
        print('Found {} boxes for {}'.format(len(out_boxes), "test_person.jpg"))
        
        colors = get_spaced_colors(len(class_names))
        count = draw_boxes(image_test, out_scores, out_boxes, out_classes, class_names, colors)
        image_test.save(os.path.join("output_images", "test_person.jpg"), quality=90)
        output_image = imread(os.path.join("output_images", "test_person.jpg"))
    
        im2 = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
        video_writer.write(im2)

        if i%25 == 0:
            info = '{}, {}\n'.format(int(i/25), count)
            f.write(info)
        
    video_reader.release()
    video_writer.release()
    f.close()

if __name__ == '__main__':
    main()
