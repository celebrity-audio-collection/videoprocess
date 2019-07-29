# The MIT License (MIT)
# Copyright (c) 2018 Sachin S Date
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import datetime
import argparse

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import to_categorical
from keras.layers.core import Dense
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import GRU, SimpleRNN
from keras.layers import Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from keras import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import time

import progressbar
from progressbar import ETA, Percentage, RotatingMarker

import os
import numpy as np
from scipy.misc import imresize

import cv2
import dlib
import math
import csv

np.random.seed(int(time.time()))

# Add keys to this hash for supporting other action classes. e.g. CLASS_HASH = {'other': 0, 'speech': 1, 'chew': 2, 'laugh': 3}
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

detector = dlib.get_frontal_face_detector()
shape_predictor = None

num_grid_combos_completed = 0

start_time = 0

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []


def load_sequences_into_memory(dataset_top_dir, type_name):
    X_data = []
    y_data = []

    num_seq_dirs = 0
    class_wise_totals = {}

    data_set_type_dir = os.path.join(dataset_top_dir, type_name)
    # print('Processing ' + data_set_type_dir)
    class_names = os.listdir(data_set_type_dir)
    # print('Found class_names ' + str(class_names))
    for class_name in class_names:
        num_sequences_for_class = 0
        class_dir = os.path.join(data_set_type_dir, class_name)
        # print('Processing ' + class_dir)
        data_set_names = os.listdir(class_dir)
        # print('Found datasets ' + str(data_set_names))
        for data_set_name in data_set_names:
            data_set_dir = os.path.join(class_dir, data_set_name)
            # print('Processing data_set_dir=' + data_set_dir)
            person_dir_names = os.listdir(data_set_dir)
            # print('Found person dirs=' + str(person_dir_names))
            for person_dir_name in person_dir_names:
                person_dir = os.path.join(data_set_dir, person_dir_name)
                # print('\nProcessing person_dir_name=' + person_dir_name)

                sequence_dir_names = os.listdir(person_dir)
                n = len(sequence_dir_names)
                num_seq_dirs += n
                num_sequences_for_class += n

        class_wise_totals[class_name] = num_sequences_for_class

    print('Loading ' + str(num_seq_dirs) + ' sequences into memory for  ' + data_set_type_dir)
    print('Class-wise totals:' + str(class_wise_totals))

    widgets = [ETA(), progressbar.Bar('>', '[', ']'), Percentage(), RotatingMarker()]
    bar = progressbar.ProgressBar(maxval=num_seq_dirs, widgets=widgets)

    bar.start()
    i = 0
    bar.update(i)

    data_set_type_dir = os.path.join(dataset_top_dir, type_name)
    class_names = os.listdir(data_set_type_dir)
    for class_name in class_names:
        class_dir = os.path.join(data_set_type_dir, class_name)
        data_set_names = os.listdir(class_dir)
        for data_set_name in data_set_names:
            data_set_dir = os.path.join(class_dir, data_set_name)
            person_dir_names = os.listdir(data_set_dir)
            for person_dir_name in person_dir_names:
                person_dir = os.path.join(data_set_dir, person_dir_name)

                sequence_dir_names = os.listdir(person_dir)

                for sequence_dir_name in sequence_dir_names:
                    sequence_dir = os.path.join(person_dir, sequence_dir_name)
                    facial_landmark_file_names = sorted(os.listdir(sequence_dir))
                    facial_landmark_file_names = facial_landmark_file_names[25:50]
                    # this should not happen if the data preparation has happened correctly
                    if len(facial_landmark_file_names) != FRAME_SEQ_LEN:
                        print('WARNING: Ignoring sequence dir ' + sequence_dir + ' with sequence len ' + str(
                            len(facial_landmark_file_names)))
                        continue

                    lip_separation_sequence = []
                    for facial_landmark_file_name in facial_landmark_file_names:
                        facial_landmark_file_path = os.path.join(sequence_dir, facial_landmark_file_name)
                        with open(facial_landmark_file_path, 'r') as f_obj:
                            reader = csv.reader(f_obj)
                            for coords in reader:
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

                                break

                            # note that [avg_gap] is a feature vector of length 1. hence the square brackets
                            lip_separation_sequence.append([avg_gap])

                    scaler = MinMaxScaler()
                    arr = scaler.fit_transform(lip_separation_sequence)
                    X_data.append(arr)
                    y_data.append(CLASS_HASH[class_name])

                    i += 1
                    bar.update(i)

    bar.finish()

    X_data = np.array(X_data)
    y_data = np.array(y_data)
    print('\nData loading completed. X_data.shape=' + str(X_data.shape) + ' y_data.shape=' + str(y_data.shape))

    return (X_data, y_data)


def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


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


def build_and_compile(num_rnn_layers=1, num_neurons_in_rnn_layer=32, is_bidirectional=True, use_gru=True, dropout=0.25,
                      num_output_dense_layers=0, num_neurons_in_output_dense_layers=0,
                      activation_output_dense_layers='relu', optimizer='adam', lr=0.0001,
                      frames_n=FRAME_SEQ_LEN, num_features=NUM_FEATURES, num_classes=NUM_CLASSES):
    lmn = LipMovementNet(num_rnn_layers, num_neurons_in_rnn_layer, is_bidirectional, use_gru, dropout,
                         num_output_dense_layers, num_neurons_in_output_dense_layers, activation_output_dense_layers,
                         optimizer, lr,
                         frames_n, num_features, num_classes)
    lmn.build()
    lmn.compile()
    lmn.print_params()
    lmn.summary()

    global num_grid_combos_completed
    if num_grid_combos_completed > 0:
        print_progress()

    num_grid_combos_completed += 1

    return lmn.model


def print_progress():
    global start_time, num_grid_combos_completed, NUM_GRID_COMBINATIONS
    end_time = time.time()
    minutes_elapsed = int((end_time - start_time) / 60)
    print('=================================================================')
    print('COMPLETED ' + str(num_grid_combos_completed) + ' out of ' + str(NUM_GRID_COMBINATIONS) + ' in ' + str(
        minutes_elapsed) + ' minutes')
    if num_grid_combos_completed > 0:
        print('Minutes per combination=' + str(int(minutes_elapsed / num_grid_combos_completed)))
    print('=================================================================')


def train(dataset_path, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, num_rnn_layers=1, num_neurons_in_rnn_layer=32,
          is_bidirectional=True, use_gru=True, dropout=0.25,
          num_output_dense_layers=0, num_neurons_in_output_dense_layers=0, activation_output_dense_layers='relu',
          optimizer='adam', lr=0.0001,
          frames_n=FRAME_SEQ_LEN, num_features=NUM_FEATURES, num_classes=NUM_CLASSES):
    model = build_and_compile(num_rnn_layers, num_neurons_in_rnn_layer, is_bidirectional, use_gru, dropout,
                              num_output_dense_layers, num_neurons_in_output_dense_layers,
                              activation_output_dense_layers, optimizer, lr,
                              frames_n, num_features, num_classes)

    tensorboard_dir = os.path.join(dataset_path, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    models_dir = os.path.join(dataset_path, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    # define callbacks
    callbacks = [ModelCheckpoint(
        os.path.join(models_dir,
                     'lip_movement_net_model_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{'
                     'val_loss:.4f}_val_categorical_accuracy-{val_categorical_accuracy:.4f}.h5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        period=1),
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.0005,
            patience=10),
        TensorBoard(log_dir=tensorboard_dir,
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True)
    ]

    global X_train, y_train, X_val, y_val
    if len(X_train) == 0:
        (X_train, y_train) = load_sequences_into_memory(dataset_path, 'train')
        # convert the labels from integers to vectors
        y_train = to_categorical(y_train, num_classes=num_classes)
    if len(X_val) == 0:
        (X_val, y_val) = load_sequences_into_memory(dataset_path, 'val')
        # convert the labels from integers to vectors
        y_val = to_categorical(y_val, num_classes=num_classes)

    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size

    print('Steps_per_epoch=' + str(steps_per_epoch))
    print('Validation_steps=' + str(validation_steps))
    print('Starting the training...')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              callbacks=callbacks, validation_data=(X_val, y_val), shuffle=True)
    print('Training completed.')

    model_file_path = os.path.join(models_dir, str(num_rnn_layers) + '_' + str(num_neurons_in_rnn_layer) + '_' + str(
        is_bidirectional) + '_' + str(use_gru) + '_' + str(dropout) + '_lip_movement_net_model.h5')
    model.save(model_file_path)
    print('Model file saved to path: ' + model_file_path)


def test(dataset_path, num_rnn_layers=1, num_neurons_in_rnn_layer=32, is_bidirectional=True, use_gru=True,
         dropout=0.25):
    print('Testing...')

    models_dir = os.path.join(dataset_path, 'models')

    model_file_path = os.path.join(models_dir, str(num_rnn_layers) + '_' + str(num_neurons_in_rnn_layer) + '_' + str(
        is_bidirectional) + '_' + str(use_gru) + '_' + str(dropout) + '_lip_movement_net_model.h5')

    print('Using model file: ' + model_file_path)
    model = load_model(model_file_path)

    global X_test, y_test
    if len(X_test) == 0:
        (X_test, y_test) = load_sequences_into_memory(dataset_path, 'test')

    # y_pred is already categorized
    y_pred = model.predict_on_batch(X_test)

    # convert y_pred from categorized continuous to single label
    y_pred_single_label = []
    for y_pred_i in y_pred:
        y_pred_single_label.append(y_pred_i.argmax())

    y_pred_single_label = np.array(y_pred_single_label)

    precision = precision_score(y_test, y_pred_single_label)
    recall = recall_score(y_test, y_pred_single_label)
    f1 = f1_score(y_test, y_pred_single_label)
    roc_auc = roc_auc_score(y_test, y_pred_single_label)

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred_single_label))
    print('Precision=', str(precision))
    print('Recall=', str(recall))
    print('F1=', str(f1))
    print('ROC AUC=', str(roc_auc))

    return [precision, recall, f1, roc_auc]


def test_video(video_path, shape_predictor_file, model):
    global shape_predictor
    shape_predictor = dlib.shape_predictor(shape_predictor_file)

    model = load_model(model)

    frames = []
    # if video_path is a directory full of frames, read all the frames in from that directory
    if os.path.isdir(video_path):
        frame_names = sorted(os.listdir(video_path))
        for frame_name in frame_names:
            img = cv2.imread(os.path.join(video_path, frame_name))
            img = imresize(img, (256, 320))
            frames.append(img)
    else:
        pass
        # cap = cv2.VideoCapture(video_path)
        '''
        while True:
            ret, img = cap.read()
            if not ret:
                break
            img = imresize(img, (360, 640))
            frames.append(img)
        '''

    print('Fetched ' + str(len(frames)) + ' frames from the video.')
    state = 'Processing'
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_num = 0
    input_sequence = []

    cap = cv2.VideoCapture(video_path)
    while True:

        ret, img = cap.read()
        if not ret:
            break
        frame = imresize(img, (360, 640))
        img = frame.copy()

        cv2.putText(img, str(frame_num), (2, 10), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        (dets, facial_points_vector) = get_facial_landmark_vectors_from_frame(frame)

        if not dets or not facial_points_vector:
            frame_num += 1
            # loop back to the beginning of the frame set
            if frame_num == len(frames):
                frame_num = 0
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
        # loop back to the beginning of the frame set
        # if frame_num == len(frames):
        #    frame_num = 0


def get_facial_landmark_vectors_from_frame(frame):
    print('Fetching face detections and landmarks...')
    dets = detector(frame, 1)
    if dets is None:
        print('no detections')
        return (None, None)
    # assume only 1 face per frame
    facial_points = []
    for k, d in enumerate(dets):
        shape = shape_predictor(frame, d)
        if shape is None:
            continue

        for i in np.arange(0, 68):
            part = shape.part(i)
            # mouth_points.append((part2.x, part2.y))
            facial_points.append(part.x)
            facial_points.append(part.y)

        if len(facial_points) > 0:
            break

    print('Returning (' + str(len(dets)) + ', ' + str(len(facial_points)) + ')')
    return (dets, facial_points)


def dist(p1, p2):
    p1_x = p1[0]
    p2_x = p2[0]
    p1_y = p1[1]
    p2_y = p2[1]
    dist = math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
    return dist


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def generate_grid_data(path_to_grid_options_csv):
    fp_obj = open(path_to_grid_options_csv, 'w', newline='')
    file_writer = csv.writer(fp_obj, delimiter=',')

    i = 0
    for num_rnn_layers in [1, 2]:
        for num_neurons_in_rnn_layer in [32, 64, 128]:
            for is_bidirectional in [False, True]:
                for use_gru in [False, True]:
                    for dropout in [0.0, 0.25, 0.5]:
                        file_writer.writerow([num_rnn_layers,
                                              num_neurons_in_rnn_layer,
                                              is_bidirectional,
                                              use_gru,
                                              dropout])
                        i += 1

    print('Wrote ' + str(i) + ' grid combinations to ' + path_to_grid_options_csv)
    fp_obj.flush()
    fp_obj.close()


def train_in_grid_search_mode(path_to_grid_options_csv, path_to_grid_results_csv, path_to_dataset_dir):
    results = {}

    # note down the combinations that have been explored already so that we don't train on those again.
    # useful if grid search is resumed after a pre-maturely terminated run.
    if os.path.exists(path_to_grid_results_csv):
        global num_grid_combos_completed
        num_grid_combos_completed = 0
        fp_obj0 = open(path_to_grid_results_csv, 'r')
        reader = csv.reader(fp_obj0)
        for results_row in reader:
            num_rnn_layers = results_row[0]
            num_neurons_in_rnn_layer = results_row[1]
            is_bidirectional = results_row[2]
            use_gru = results_row[3]
            dropout = results_row[4]

            key = num_rnn_layers + '_' + num_neurons_in_rnn_layer + '_' + is_bidirectional + '_' + use_gru + '_' + dropout
            results[key] = True

            num_grid_combos_completed += 1

        fp_obj0.close()

    global NUM_GRID_COMBINATIONS
    NUM_GRID_COMBINATIONS = 0
    fp_obj1 = open(path_to_grid_options_csv, 'r')
    reader = csv.reader(fp_obj1)
    for grid_options in reader:
        NUM_GRID_COMBINATIONS += 1
    fp_obj1.close()

    fp_obj1 = open(path_to_grid_options_csv, 'r')
    reader = csv.reader(fp_obj1)

    exists_file = os.path.exists(path_to_grid_results_csv)
    fp_obj2 = open(path_to_grid_results_csv, 'a+', newline='')
    file_writer = csv.writer(fp_obj2, delimiter=',')
    if not exists_file:
        file_writer.writerow(['num_rnn_layers', 'num_neurons_in_rnn_layer', 'is_bidirectional', 'use_gru', 'dropout',
                              'precision', 'recall', 'f1', 'roc_auc'])

    global start_time
    start_time = time.time()

    for grid_options in reader:
        num_rnn_layers = int(grid_options[0])
        num_neurons_in_rnn_layer = int(grid_options[1])
        is_bidirectional = str2bool(grid_options[2])
        use_gru = str2bool(grid_options[3])
        dropout = float(grid_options[4])

        d = grid_options[0] + '_' + grid_options[1] + '_' + grid_options[2] + '_' + grid_options[3] + '_' + \
            grid_options[4]

        if d in results:
            print('SKIPPING already trained combination: ' + str(grid_options))
            continue

        train(dataset_path=path_to_dataset_dir, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
              num_rnn_layers=num_rnn_layers, num_neurons_in_rnn_layer=num_neurons_in_rnn_layer,
              is_bidirectional=is_bidirectional, use_gru=use_gru, dropout=dropout,
              num_output_dense_layers=0,
              num_neurons_in_output_dense_layers=0, activation_output_dense_layers='relu',
              optimizer='adam', lr=0.0001,
              frames_n=FRAME_SEQ_LEN, num_features=NUM_FEATURES, num_classes=NUM_CLASSES)

        results = test(path_to_dataset_dir, num_rnn_layers, num_neurons_in_rnn_layer, is_bidirectional, use_gru,
                       dropout)

        row = grid_options + results
        file_writer.writerow(row)
        fp_obj2.flush()

    fp_obj1.close()

    fp_obj2.flush()
    fp_obj2.close()


if __name__ == '__main__':

    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')

    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--dataset", required=False,
                    help="path to training data directory")
    ap.add_argument("-gg", "--gen_grid_options", required=False,
                    help="generate the csv file containing the grid search combinations")
    ap.add_argument("-go", "--grid_options_csv", required=False,
                    help="path to file containing grid search combinations")
    ap.add_argument("-gr", "--grid_results_csv", required=False,
                    help="path to output file containing grid search results")
    ap.add_argument("-v", "--video_file", required=False, default='../videos/interview-1.mp4',
                    help="path to video file")
    ap.add_argument("-p", "--shape_predictor", required=False, default='../models/shape_predictor_68_face_landmarks.dat',
                    help="shape predictor file")
    ap.add_argument("-m", "--model", required=False, default='../models/2_64_False_True_0.5_lip_motion_net_model.h5',
                    help="shape model file")

    args = vars(ap.parse_args())

    if args['gen_grid_options'] == 'True':
        generate_grid_data(args['grid_options_csv'])
        exit(0)

    if args['grid_options_csv']:
        train_in_grid_search_mode(args['grid_options_csv'], args['grid_results_csv'], args['dataset'])
        exit(0)

    if args['dataset']:
        start_time = time.time()

        train(args['dataset'], epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, num_rnn_layers=1, num_neurons_in_rnn_layer=32,
              is_bidirectional=True, use_gru=True, dropout=0.25,
              num_output_dense_layers=0, num_neurons_in_output_dense_layers=0, activation_output_dense_layers='relu',
              optimizer='adam', lr=0.0001,
              frames_n=FRAME_SEQ_LEN, num_features=NUM_FEATURES, num_classes=NUM_CLASSES)

        test(args['dataset'])
        exit(0)

    if args['shape_predictor'] and args['video_file'] and args['model']:
        test_video(args['video_file'], args['shape_predictor'], args['model'])
        exit(0)
