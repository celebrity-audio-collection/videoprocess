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

from argparse import ArgumentParser
import cv2
import argparse
import os
import dlib
import numpy as np
import progressbar
from progressbar import ETA, Percentage, RotatingMarker
import csv
import random
import time

# Which frame number in the video file do you want to start reading the frames from for each dataset.
# The dataset key name must match the name of the dataset folder in the source daatset directory structure
# If a dataset is not mentioned in this dictionary, the start offset is assumed to be zero for that dataset.
# This is useful to specify if you know videos from this dataset (e.g. GRID) do not contain the action unit of interest (e.g. speech)
# before the start offset.
VIDEO_START_OFFSETS = {
    'GRID': 25,
    'HMDB': 0
}
# Which frame number in the video file do you want to stop reading the frames at for each dataset. The frame grabber
# will read frames from the video file in the closed-open interval [VIDEO_START_OFFSET, VIDEO_END_OFFSET)
# The dataset key name must match the name of the dataset folder in the source daatset directory structure
# If a dataset is not mentioned in this dictionary, the end offset is assumed to be max int32
# This is useful to specify if you know videos from this dataset (e.g. GRID) do not contain the action unit of interest
# (e.g. speech) after the end offset.
VIDEO_END_OFFSETS = {
    'GRID': 50,
    'HMDB': 2147483647
}
# concatenate all the sequences generated from all the video files for a person, then chop the concatenated
# list into equal size sequences. This is useful if you have a lot of tiny sized video sequences of the same person.
# If a dataset is not mentioned in this dictionary, cross_file_boundaries defaults to False for that dataset.
CROSS_FILE_BOUNDARIES = {
    'GRID': False,
    'BBC': True
}

detector = dlib.get_frontal_face_detector()
shape_predictor = None


# Before running this script, organize your source dataset as follows
# dataset/
# 	train/
# 		speech/
# 			dataset_1_name/
# 				person_1/
# 					person_1_video_file_name_1.mp4
# 					person_1_video_file_name_2.mp4
# 					person_1_video_file_name_3.mp4
# 					person_1_video_file_name_4.mp4
# 					person_1_video_file_name_5.mp4
# 					...
# 					...
# 					...
# 				person_2/
# 					person_2_video_file_name_1.mp4
# 					person_2_video_file_name_2.mp4
# 					person_2_video_file_name_3.mp4
# 					person_2_video_file_name_4.mp4
# 					person_2_video_file_name_5.mp4
# 					...
# 					...
# 					...
# 				...
# 				...
# 				...
# 			dataset_2_name/
# 				person_3/
# 					person_3_video_file_name_1.mp4
# 					person_3_video_file_name_2.mp4
# 					person_3_video_file_name_3.mp4
# 					person_3_video_file_name_4.mp4
# 					person_3_video_file_name_5.mp4
# 					...
# 					...
# 					...
# 				person_4/
# 					person_4_video_file_name_1.mp4
# 					person_4_video_file_name_2.mp4
# 					person_4_video_file_name_3.mp4
# 					person_4_video_file_name_4.mp4
# 					person_4_video_file_name_5.mp4
# 					...
# 					...
# 					...
# 				...
# 				...
# 				...
# 			...
# 			...
# 			...
# 		silence/
# 			dataset_1_name/
# 			dataset_2_name/
# 				person_5/
# 					person_5_video_file_name_1.mp4
# 					person_5_video_file_name_2.mp4
# 					person_5_video_file_name_3.mp4
# 					person_5_video_file_name_4.mp4
# 					person_5_video_file_name_5.mp4
# 				person_6/
# 					person_6_video_file_name_1.mp4
# 					person_5_video_file_name_2.mp4
# 					person_5_video_file_name_3.mp4
# 					person_5_video_file_name_4.mp4
# 					person_5_video_file_name_5.mp4
# 				...
# 				...
# 				...
# 			dataset_3_name/
# 			...
# 			...
# 			...
# 	val/
# 		speech/
# 			...
# 		silence/
# 			...
# 	test/
# 		speech/
# 			...
# 		silence/
# 			...
# see assets/sample_source_dataset for a sample directory structure
def prepare():
    ap: ArgumentParser = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True,
                    help='path to directory containing videos or image sequences')
    ap.add_argument('-o', '--output', required=True,
                    help='path to output sequences dataset directory')
    ap.add_argument('-s', '--sequence_length', required=False,
                    help='length of each sequence')
    ap.add_argument('-p', '--shape_predictor', required=True,
                    help='shape shape_predictor file')
    ap.add_argument('-npf', '--num_person_files', required=False,
                    help='choose this many number of random video files per person from a speaker/individual\'s directory')

    args = vars(ap.parse_args())

    print('input=' + args['input'])
    print('output=' + args['output'])

    in_dir = args['input']
    out_dir = args['output']

    sequence_length = 25
    if args['sequence_length']:
        sequence_length = int(args['sequence_length'])

    global shape_predictor
    shape_predictor = dlib.shape_predictor(args['shape_predictor'])

    num_person_files = 0
    if args['num_person_files']:
        num_person_files = int(args['num_person_files'])

    # create the output directory structure
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_train_dir = os.path.join(out_dir, 'train')
    if not os.path.exists(out_train_dir):
        os.makedirs(out_train_dir)

    out_val_dir = os.path.join(out_dir, 'val')
    if not os.path.exists(out_val_dir):
        os.makedirs(out_val_dir)

    out_test_dir = os.path.join(out_dir, 'test')
    if not os.path.exists(out_test_dir):
        os.makedirs(out_test_dir)

    out_models_dir = os.path.join(out_dir, 'models')
    if not os.path.exists(out_models_dir):
        os.makedirs(out_models_dir)

    out_tensorboard_dir = os.path.join(out_dir, 'tensorboard')
    if not os.path.exists(out_tensorboard_dir):
        os.makedirs(out_tensorboard_dir)

    # get the list of all class names from the input dataset's directory structure
    in_train_dir = os.path.join(in_dir, 'train')
    class_names = os.listdir(in_train_dir)
    print('Found the following detection classes: ' + str(class_names))

    # create the output class directories
    for class_name in class_names:
        for d in [out_train_dir, out_val_dir, out_test_dir]:
            dir_name = os.path.join(d, class_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

    print('Finished creating output dataset directory structure.')

    # calculate the number of video files to process so that we can display progress bar
    total_num_video_files = 0
    for type_name in ['train', 'val', 'test']:
        data_set_type_dir = os.path.join(in_dir, type_name)
        class_names = os.listdir(data_set_type_dir)
        for class_name in class_names:
            class_dir = os.path.join(data_set_type_dir, class_name)
            data_set_names = os.listdir(class_dir)
            for data_set_name in data_set_names:
                data_set_dir = os.path.join(class_dir, data_set_name)
                person_dir_names = os.listdir(data_set_dir)
                for person_dir_name in person_dir_names:
                    person_dir = os.path.join(data_set_dir, person_dir_name)
                    video_file_names = os.listdir(person_dir)
                    total_num_video_files += len(video_file_names)

    print('Found ' + str(total_num_video_files) + ' video files.')

    # initialize the progress bar
    widgets = [ETA(), progressbar.Bar('>', '[', ']'), Percentage(), RotatingMarker()]
    bar = progressbar.ProgressBar(maxval=total_num_video_files, widgets=widgets)
    bar.start()
    i = 0
    bar.update(i)

    # process each video in each person dir, create the train/val/tet data sequences and write them out to the
    # corresponding output dir
    for type_name in ['train', 'val', 'test']:
        data_set_type_dir = os.path.join(in_dir, type_name)
        print('Processing ' + data_set_type_dir)
        class_names = os.listdir(data_set_type_dir)
        print('Found class_names ' + str(class_names))
        for class_name in class_names:
            class_dir = os.path.join(data_set_type_dir, class_name)
            print('Processing ' + class_dir)
            data_set_names = os.listdir(class_dir)
            print('Found datasets ' + str(data_set_names))
            for data_set_name in data_set_names:
                data_set_in_dir = os.path.join(class_dir, data_set_name)
                data_set_out_dir = os.path.join(os.path.join(os.path.join(out_dir, type_name), class_name),
                                                data_set_name)
                # print('data_set_in_dir=' + data_set_in_dir)
                # print('data_set_out_dir=' + data_set_out_dir)
                if not os.path.exists(data_set_out_dir):
                    os.makedirs(data_set_out_dir)
                person_dir_names = os.listdir(data_set_in_dir)
                # print('Found person dirs=' + str(person_dir_names))
                for person_dir_name in person_dir_names:
                    person_in_dir = os.path.join(data_set_in_dir, person_dir_name)
                    person_out_dir = os.path.join(data_set_out_dir, person_dir_name)
                    print('\nProcessing person_dir_name=' + person_dir_name)
                    # print('person_out_dir=' + person_out_dir)
                    if not os.path.exists(person_out_dir):
                        os.makedirs(person_out_dir)
                    elif len(os.listdir(person_out_dir)) > 0:
                        print('SKIPPING person directory ' + person_out_dir)
                        i += 1
                        bar.update(i)
                        continue

                    video_start_offset = 0
                    if data_set_name in VIDEO_START_OFFSETS:
                        video_start_offset = VIDEO_START_OFFSETS[data_set_name]
                    video_end_offset = 2147483647
                    if data_set_name in VIDEO_END_OFFSETS:
                        video_end_offset = VIDEO_END_OFFSETS[data_set_name]
                    cross_file_boundaries = False
                    if data_set_name in CROSS_FILE_BOUNDARIES:
                        cross_file_boundaries = CROSS_FILE_BOUNDARIES[data_set_name]

                    extract_facial_landmarks_sequences_for_person(person_in_dir, person_out_dir, num_person_files,
                                                                  video_start_offset, video_end_offset, class_name,
                                                                  sequence_length, cross_file_boundaries)

                    i += 1
                    bar.update(i)

    bar.finish()


def extract_facial_landmarks_sequences_for_person(person_in_dir, person_out_dir, num_person_files, video_start_offset,
                                                  video_end_offset, class_name, sequence_length, cross_file_boundaries):
    video_file_names = os.listdir(person_in_dir)
    # print('extracting facial landmarks for person=' + person_in_dir)

    if num_person_files > 0:
        random.seed(int(time.time()))
        file_names = random.sample(list(enumerate(video_file_names)), num_person_files)
        indexes = []
        files = []
        for (index, file_name) in file_names:
            indexes.append(index)
            files.append(file_name)

        video_file_names = sorted(files)

    # print('Processing video files=' + str(video_file_names))

    all_fixed_len_sequences_for_person = []
    sequences_for_person = []
    for video_file_name in video_file_names:
        video_file_path = os.path.join(person_in_dir, video_file_name)
        # fetch all the sequences from one video file or frames folder
        sequences__per_video_file = extract_facial_landmarks_sequences_from_video_file(video_file_path,
                                                                                       video_start_offset,
                                                                                       video_end_offset)
        # print('Found ' + str(len(sequences__per_video_file)) + ' sequences')
        if cross_file_boundaries:
            sequences_for_person += sequences__per_video_file
        else:
            # chop up all the sequences for this video file into groups of length=sequence_length
            for j in range(0, len(sequences__per_video_file), sequence_length):
                fixed_len_sequence = sequences__per_video_file[j:j + sequence_length]
                if len(fixed_len_sequence) < sequence_length:
                    continue
                all_fixed_len_sequences_for_person.append(fixed_len_sequence)

    if cross_file_boundaries:
        # print('Accumulated sequences_for_person of length=' + str(len(sequences_for_person)))
        for j in range(0, len(sequences_for_person), sequence_length):
            fixed_len_sequence = sequences_for_person[j:j + sequence_length]
            if len(fixed_len_sequence) < sequence_length:
                continue
            all_fixed_len_sequences_for_person.append(fixed_len_sequence)

    # print('all_fixed_len_sequences_for_person='+str(len(all_fixed_len_sequences_for_person)))

    write_sequences_to_disk(all_fixed_len_sequences_for_person, person_out_dir, class_name)


def extract_facial_landmarks_sequences_from_video_file(video_file_path, video_start_offset, video_end_offset):
    # print('extracting facial landmark sequences from file=' + video_file_path)
    frames = []
    # check if the video_file_path is actually a directory containing video frames
    if os.path.isfile(video_file_path) and \
            (video_file_path.lower().endswith('.mp4') or video_file_path.lower().endswith('.mpeg')
             or video_file_path.lower().endswith('.mpg') or video_file_path.lower().endswith('.avi')):
        frames = convert_video_file_to_frames(video_file_path, video_start_offset, video_end_offset)
    elif os.path.isdir(video_file_path):
        frames = read_frames_from_dir(video_file_path, video_start_offset, video_end_offset)
    else:
        # print(video_file_path + ' is neither a video format I can read, nor a directory. Ignoring it.')
        return []

    sequences = []
    for frame in frames:
        # for now get the full 68 point landmarks vector. we'll pick only the mouth points from this vector later
        # during training
        landmarks_vector = get_facial_landmark_vectors_from_frame(frame)
        # found a valid value?
        if len(landmarks_vector) >= 0:
            sequences.append(landmarks_vector)

    return sequences


# all_fixed_len_sequences_for_person has the following structure:
# [ [[], [], [], ...sequence_length times], [[], [], [], ...sequence_length times], [[], [], [], ...sequence_length times],...]
def write_sequences_to_disk(all_fixed_len_sequences_for_person, person_out_dir, class_name):
    # print('writing ' + str(len(all_fixed_len_sequences_for_person)) + ' sequences to disk for ' + person_out_dir + ' class=' + class_name)
    seq_num = 0
    for sequence in all_fixed_len_sequences_for_person:
        out_seq_dir_path = os.path.join(person_out_dir, stringify_seq(seq_num) + '_' + class_name)
        if not os.path.exists(out_seq_dir_path):
            os.makedirs(out_seq_dir_path)
        counter = 0
        for landmarks_vector_row in sequence:
            fp_obj = open(os.path.join(out_seq_dir_path, stringify_seq(counter) + '.csv'), 'w', newline='')
            file_writer = csv.writer(fp_obj, delimiter=',')
            file_writer.writerow(landmarks_vector_row)
            fp_obj.flush()
            fp_obj.close()
            counter += 1

        seq_num += 1


def convert_video_file_to_frames(video_file_path, video_start_offset, video_end_offset):
    frames = []
    cap = cv2.VideoCapture(video_file_path)
    frame_num = 0
    # print('video_start_offset='+str(video_start_offset) + ' video_end_offset='+str(video_end_offset))
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.shape is None or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            break

        if frame_num >= video_start_offset and frame_num < video_end_offset:
            frames.append(frame)

        frame_num += 1

    # print('Returning ' + str(len(frames)) + ' frames')
    return frames


def read_frames_from_dir(path, video_start_offset, video_end_offset):
    frames = []
    file_names = sorted(os.listdir(path))
    frame_num = 0

    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        frame = None
        ignore = False
        # try to load the image
        try:
            frame = cv2.imread(file_path)
            if frame is None or frame.shape is None or frame.shape[0] <= 0 or frame.shape[1] <= 0:
                ignore = True

        # if OpenCV cannot load the image then the image is likely
        # corrupt so we should ignore it
        except:
            # print('Exception')
            ignore = True

        # check to see if the image should be deleted
        if ignore:
            # print('[INFO] Ignoring {}'.format(file_path))
            # os.remove(file_path)
            pass
        elif frame is not None:
            if frame_num >= video_start_offset and frame_num < video_end_offset:
                frames.append(frame)

        frame_num += 1

    return frames


def get_facial_landmark_vectors_from_frame(frame):
    dets = detector(frame, 1)
    if dets is None:
        return []
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

    return facial_points


def stringify_seq(num):
    s = str(num)
    l = len(s)
    num_zeros = 6 - l
    for i in range(num_zeros):
        s = '0' + s

    return s


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    prepare()
