#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys

import keras
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import models


import numpy as np
import cv2
import sys
import dlib
from skimage import io
import numpy
import matplotlib.pyplot as plt
import math
import time

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: {0} <model_name>'.format(sys.argv[0]))
        exit(-1)
    else:
        model_name = sys.argv[1]

    # load predictor
    predictor_path = "D:\\CSLT\\Retinanet_face\\keras-retinanet\\model\\dlib\\shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(model_name)

    cap = cv2.VideoCapture("F:\\cslt\\spring1\\郭子凡\\interview\\因为一段访问被圈粉——郭子凡、D.O.、周震南 - 1.因为一段访问被圈粉-郭子凡、D.O.周震南(Av48807671,P1).mp4")
    while True:
        ret, raw_image = cap.read()

        raw_image = cv2.resize(raw_image, (642, 360))
        image = raw_image.copy()
        # image = image / 127.5
        # image = image - 1.

        # cv2.imshow('image', image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        # boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))[:3]
        boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        # boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > 0.05)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:100]

        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]

        image_scores = scores[scores_sort]
        image_labels = labels[0, indices[scores_sort]]

        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        selection = np.where(image_scores > 0.5)[0]


        ##add position


        # You should download this file manually
        # predictor_path = "shape_predictor_68_face_landmarks.dat"
        #
        # detector = dlib.get_frontal_face_detector()
        # predictor = dlib.shape_predictor(predictor_path)

        # feel free to use any photo you want

        # plt.imshow(image)

        # array of faces
        # dets = detector(image, 1)
        # for k, d in enumerate(dets):
        #     shape = predictor(image, d)
        #     np_shape = []
        #     for i in shape.parts():
        #         np_shape.append([i.x, i.y])
        #     np_shape = numpy.array(np_shape)
        #
        #     plt.scatter(np_shape[:, 0], np_shape[:, 1], c='w', s=8)
        #     # plt.plot(shape_graphed_np[:, 0], shape_graphed_np[:, 1], c='w')
        #
        # plt.show()
        imgsplit= raw_image.copy()
        for i in selection:
            b = np.array(image_boxes[i, :]).astype(int)
            print("b: ", b)
            print(raw_image.shape)

            if(b[0] < 0  or b[2]>raw_image.shape[1] or b[1] <0 or b[3]>raw_image.shape[0]):
                continue
            facepicture = imgsplit[b[1]:b[3], b[0]:b[2], :]
            print(facepicture.shape)
            cv2.imshow("faceimg",facepicture)
            time.sleep(5)
            cv2.rectangle(raw_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2, cv2.LINE_AA)
            ## predict points using dlib
            print(type(raw_image))
            pre_b = dlib.rectangle(int(b[0]), int(b[1]), int(b[2]), int(b[3]))

            print("pre_b: ", pre_b)
            shape = predictor(raw_image, pre_b)
            np_shape = []
            for key_point in shape.parts():
                # print("i: ", key_point)
                np_shape.append([key_point.x, key_point.y])
            np_shape = numpy.array(np_shape)
            index_color = 0
            for point in np_shape:
                pos = (point[0], point[1])
                cv2.circle(raw_image, pos, 1, (255, 255, 255/68*index_color), -1)
                index_color = index_color + 1
            ## end
            # draw labels
            caption = str(image_labels[i]) + " : " + str(image_scores[i])

            cv2.putText(raw_image, str(caption), (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
            cv2.putText(raw_image, str(caption), (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        cv2.imshow('raw_image', raw_image)
        if cv2.waitKey(10) == 27:
            break
