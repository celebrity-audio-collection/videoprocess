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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
from facenet_code import facenet
import facenet_code.align.detect_face

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models


import numpy as np
import cv2
import sys
import dlib
from skimage import io
import numpy
import matplotlib.pyplot as plt
import math
import time


def compare(imgin):

    image_size = 160
    margin = 44
    gpu_memory_fraction = 0.7
    model ="D:\\CSLT\\Retinanet_face\\keras-retinanet\\facenet_code\\20180402-114759"

    print("main is called")
    image_files=[imgin,"D:\\CSLT\Retinanet_face\\keras-retinanet\\images\\乔杉.jpg"]
    images = load_and_align_data(image_files, image_size, margin, gpu_memory_fraction)
    print("image loaded")
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    with tf.Graph().as_default():

        print("load graph")
        with tf.Session() as sess:

            # Load the model
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)

            nrof_images = len(image_files)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, image_files[i]))
            print('')

            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    if i!=j:
                        result = dist
                    print('  %1.4f  ' % dist, end='')
                print('')
    return result

def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = facenet_code.align.detect_face.create_mtcnn(sess, None)

    tmp_image_paths = copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = facenet_code.align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            image_paths.remove(image)
            print("can't detect face, remove ", image)
            continue
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images


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

    cap = cv2.VideoCapture("E:\\CSLT1000\\vedios\\乔杉\\interview\\《来电狂响》霍思燕乔杉讲述电影故事 - 1.《来电狂响》霍思燕乔杉讲述电影故事(Av39758217,P1).mp4")
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
            length = int(max(b[3]-b[1],b[2]-b[0])/2)
            center = [int((b[1]+b[3])/2),int((b[0]+b[2])/2)]
            facepicture = imgsplit[center[0]-length:center[0]+length, center[1]-length:center[1]+length, :]
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
