import time

import numpy as np
from utils import *
from config import anchors, class_names
from matplotlib.pyplot import imread, imshow
import matplotlib.pyplot as plt

from keras import backend as K
K.set_learning_phase(1) #set learning phase

import cv2
from tqdm import tqdm

from yolov2 import Yolov2
from tinyyolov2 import TinyYolov2

def main():
    #yolo = Yolov2(input_shape = (416,416,3))
    #model = yolo.get_model()
    #model.load_weights('weights/yolov2.h5')

    yolo = TinyYolov2(input_shape = (416,416,3))
    model = yolo.get_model()
    model.load_weights('weights/tinyyolov2.h5')

    anchor = np.array(anchors)

    #video_reader = cv2.VideoCapture(0)

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

    sess = K.get_session()

    for i in tqdm(range(nb_frames)):
        ret, image = video_reader.read()

        cv2.imwrite("./frame.jpg", image)

        image_test, image_data = preprocess_test_image('./frame.jpg', model_image_size = (416, 416))

        image_shape = image_test.size
        image_shape = tuple(float(i) for i in reversed(image_test.size))

        yolo_outputs = yolo_head(model.output, anchor, len(class_names))
        scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

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
