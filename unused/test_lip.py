# https://github.com/sachinsdate/lip-movement-net
import datetime
import argparse

from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import GRU, SimpleRNN
from keras.layers import Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from keras import metrics

from sklearn.preprocessing import MinMaxScaler
import time

from face_detection import FaceDetection
from common import config

import os
import numpy as np
from scipy.misc import imresize

import cv2
import dlib
import math
import csv

np.random.seed(int(time.time()))

face_detection_model = FaceDetection()

# Add keys to this hash for supporting other action classes. e.g. CLASS_HASH = {'other': 0, 'speech': 1, 'chew': 2,
# 'laugh': 3}
CLASS_HASH = {
    'silent': 0,
    'speaking': 1
}

MOUTH_WIDTH = 100
MOUTH_HEIGHT = 50
HORIZONTAL_PAD = 0.10

IMG_NUM_CHANNELS = 1
FRAME_SEQ_LEN = 25
BATCH_SIZE = 16
NUM_EPOCHS = 500
NUM_CLASSES = len(CLASS_HASH.keys())
NUM_FEATURES = 1
NUM_GRID_COMBINATIONS = 1

shape_predictor = None

num_grid_combos_completed = 0

start_time = 0

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []


class LipMovementNet(object):
    def __init__(self, num_rnn_layers=1, num_neurons_in_rnn_layer=100, is_bidirectional=False, use_gru=False,
                 dropout=0.0,
                 num_output_dense_layers=1,
                 num_neurons_in_output_dense_layers=2, activation_output_dense_layers='relu', optimizer='rmsprop',
                 lr=0.0001,
                 frames_n=FRAME_SEQ_LEN, num_features=NUM_FEATURES, num_classes=NUM_CLASSES):
        self.frames_n = frames_n
        self.num_features = num_features
        self.num_classes = num_classes

        self.num_rnn_layers = num_rnn_layers
        self.num_neurons_in_rnn_layer = num_neurons_in_rnn_layer
        self.is_bidirectional = is_bidirectional
        self.use_gru = use_gru
        self.dropout = dropout

        self.num_output_dense_layers = num_output_dense_layers
        self.num_neurons_in_output_dense_layers = num_neurons_in_output_dense_layers
        self.activation_output_dense_layers = activation_output_dense_layers
        self.optimizer = optimizer
        self.lr = lr

    def build(self):

        input_shape = (self.frames_n, self.num_features)

        self.model = Sequential()

        for i in range(self.num_rnn_layers):
            return_sequences = True
            if i == self.num_rnn_layers - 1:
                return_sequences = False

            specify_input_shape = False
            if i == 0:
                specify_input_shape = True

            name = 'rnn-' + str(i)

            if self.use_gru:
                if self.is_bidirectional:
                    if specify_input_shape:
                        self.model.add(
                            Bidirectional(GRU(self.num_neurons_in_rnn_layer, return_sequences=return_sequences,
                                              name=name), input_shape=input_shape, merge_mode='concat'))
                    else:
                        self.model.add(
                            Bidirectional(GRU(self.num_neurons_in_rnn_layer, return_sequences=return_sequences,
                                              name=name), merge_mode='concat'))
                else:
                    if specify_input_shape:
                        self.model.add(GRU(self.num_neurons_in_rnn_layer, return_sequences=return_sequences, name=name,
                                           input_shape=input_shape))
                    else:
                        self.model.add(GRU(self.num_neurons_in_rnn_layer, return_sequences=return_sequences, name=name))
            else:
                if self.is_bidirectional:
                    if specify_input_shape:
                        self.model.add(
                            Bidirectional(SimpleRNN(self.num_neurons_in_rnn_layer, return_sequences=return_sequences,
                                                    name=name), input_shape=input_shape, merge_mode='concat'))
                    else:
                        self.model.add(
                            Bidirectional(SimpleRNN(self.num_neurons_in_rnn_layer, return_sequences=return_sequences,
                                                    name=name), merge_mode='concat'))
                else:
                    if specify_input_shape:
                        self.model.add(
                            SimpleRNN(self.num_neurons_in_rnn_layer, return_sequences=return_sequences, name=name,
                                      input_shape=input_shape))
                    else:
                        self.model.add(
                            SimpleRNN(self.num_neurons_in_rnn_layer, return_sequences=return_sequences, name=name))

        if self.dropout > 0.0:
            self.model.add(Dropout(self.dropout))

        for i in range(self.num_output_dense_layers):
            name = 'dense-' + str(i)
            self.model.add(
                Dense(self.num_neurons_in_output_dense_layers, activation=self.activation_output_dense_layers,
                      name=name))

        self.model.add(Dense(self.num_classes, name='softmax', activation='softmax'))

    def compile(self):
        if self.optimizer == 'adam':
            opt = Adam(lr=self.lr)
        elif self.optimizer == 'rmsprop':
            opt = RMSprop(lr=self.lr)
        else:
            opt = None
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[metrics.categorical_accuracy])

    def summary(self):
        self.model.summary()

    def save(self, model_file):
        self.model.save(model_file)

    def print_params(self):
        print('self.num_rnn_layers = ' + str(self.num_rnn_layers))
        print('self.num_neurons_in_rnn_layer = ' + str(self.num_neurons_in_rnn_layer))
        print('self.is_bidirectional = ' + str(self.is_bidirectional))
        print('self.use_gru = ' + str(self.use_gru))
        print('self.dropout = ' + str(self.dropout))
        print('self.num_output_dense_layers = ' + str(self.num_output_dense_layers))
        print('self.num_neurons_in_output_dense_layers = ' + str(self.num_neurons_in_output_dense_layers))
        print('self.activation_output_dense_layers = ' + str(self.activation_output_dense_layers))
        print('self.optimizer = ' + str(self.optimizer))
        print('self.lr = ' + str(self.lr))


def test_video(video_path, shape_predictor_file, model):
    global shape_predictor
    shape_predictor = dlib.shape_predictor(shape_predictor_file)

    model = load_model(model)

    state = 'Processing'
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_num = 0
    input_sequence = []
    FPS = 0
    last_time = time.clock()

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, img = cap.read()
        if not ret:
            break

        if frame_num % 10 == 0:
            FPS = int(1 / (time.clock() - last_time) * 10)
            last_time = time.clock()

        frame = imresize(img, (720, 1280))
        img = frame.copy()

        cv2.putText(img, 'Frame: ' + str(frame_num) + ' FPS: %d' % FPS, (2, 10), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        (dets, facial_points_vector) = get_facial_landmark_vectors_from_frame(frame)

        if not dets or not facial_points_vector:
            frame_num += 1
            cv2.imshow('Video', img)
            cv2.waitKey(10)
            continue

        # draw a box showing the detected face
        for i, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            # 	i, d.left(), d.top(), d.right(), d.bottom()))

            cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
            # draw the state label below the face
            cv2.rectangle(img, (d.left(), d.bottom()), (d.right(), d.bottom() + 10), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, state, (d.left() + 2, d.bottom() + 10 - 3), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # add the facial points vector to the current input sequence vector for the RNN
        input_sequence.append(facial_points_vector)

        if len(input_sequence) >= FRAME_SEQ_LEN:
            # get the most recent N sequences where N=FRAME_SEQ_LEN
            seq = input_sequence[-1 * FRAME_SEQ_LEN:]
            f = []
            for coords in seq:
                part_61 = (int(coords[2 * 61]), int(coords[2 * 61 + 1]))
                part_67 = (int(coords[2 * 67]), int(coords[2 * 67 + 1]))
                part_62 = (int(coords[2 * 62]), int(coords[2 * 62 + 1]))
                part_66 = (int(coords[2 * 66]), int(coords[2 * 66 + 1]))
                part_63 = (int(coords[2 * 63]), int(coords[2 * 63 + 1]))
                part_65 = (int(coords[2 * 65]), int(coords[2 * 65 + 1]))

                A = dist(part_61, part_67)
                B = dist(part_62, part_66)
                C = dist(part_63, part_65)

                avg_gap = (A + B + C) / 3.0

                f.append([avg_gap])

            scaler = MinMaxScaler()
            arr = scaler.fit_transform(f)

            X_data = np.array([arr])

            # y_pred is already categorized
            y_pred = model.predict_on_batch(X_data)

            # print('y_pred=' + str(y_pred) + ' shape=' + str(y_pred.shape))

            # convert y_pred from categorized continuous to single label
            y_pred_max = y_pred[0].argmax()
            print('y_pred=' + str(y_pred) + ' y_pred_max=' + str(y_pred_max))

            for k in CLASS_HASH:
                if y_pred_max == CLASS_HASH[k]:
                    state = k;
                    break

            # redraw the label
            for i, d in enumerate(dets):
                # draw the state label below the face
                cv2.rectangle(img, (d.left(), d.bottom()), (d.right(), d.bottom() + 10), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, state, (d.left() + 2, d.bottom() + 10 - 3), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_num += 1


def get_facial_landmark_vectors_from_frame(frame):
    # print('Fetching face detections and landmarks...')

    bboxes, landmarks = face_detection_model.update(frame)

    dets = []

    if bboxes is None:
        print('no detections')
        return (None, None)
    # assume only 1 face per frame
    facial_points = []
    for k, bbox in enumerate(bboxes):
        pre_b = dlib.rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        dets.append(pre_b)
        shape = shape_predictor(frame, pre_b)

        if shape is None:
            continue

        for i in np.arange(0, 68):
            part = shape.part(i)
            # mouth_points.append((part2.x, part2.y))
            facial_points.append(part.x)
            facial_points.append(part.y)

        if len(facial_points) > 0:
            break

    # print('Returning (' + str(len(dets)) + ', ' + str(len(facial_points)) + ')')
    return dets, facial_points


def dist(p1, p2):
    p1_x = p1[0]
    p2_x = p2[0]
    p1_y = p1[1]
    p2_y = p2[1]
    dist = math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
    return dist


if __name__ == '__main__':

    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video_file", required=False, default='videos/蔡明/蔡明-1.mp4',
                    help="path to video file")
    ap.add_argument("-p", "--shape_predictor", required=False,
                    default=config.landmark_predictor,
                    help="shape predictor file")
    ap.add_argument("-m", "--model", required=False,
                    default='lip-movement-net/models/1_32_False_True_0.25_lip_motion_net_model.h5',
                    help="shape model file")

    args = vars(ap.parse_args())

    if args['shape_predictor'] and args['video_file'] and args['model']:
        test_video(args['video_file'], args['shape_predictor'], args['model'])
        exit(0)
