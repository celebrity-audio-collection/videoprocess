# -*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common import *
from cv_tracker import CV_Tracker
from face_detection import FaceDetection
from face_validation import FaceValidation
from speaker_validation import SpeakerValidation
from evaluate import evaluate_result

import cv2
import subprocess
import gc
import numpy as np
from scipy.io import wavfile
import time
import argparse

if config.debug:
    from audio_player import AudioPlayer

if config.use_facenet:
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=gpuconfig)
    KTF.set_session(sess)


def isTracking(box_center, tracker_list):
    for tracker in tracker_list:
        if tracker.is_tracking(box_center) is True:
            return True
    return False


def load_models():
    print("loading face detection model")
    face_detection_model = FaceDetection()

    # load face validation model
    print("loading face validation model")
    face_validation_model = FaceValidation()

    # SyncNet
    print("loading speaker validation model")
    speaker_validation = SpeakerValidation()
    return face_detection_model, face_validation_model, speaker_validation


def process_single_video(video_dir, output_dir, face_detection_model, face_validation_model, speaker_validation,
                         output_video_dir=None):
    audio_tmp = './temp/audio.wav'
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s > %s 2>&1" % (
        video_dir, audio_tmp, os.path.join(config.log_dir, "ffmpeg.log")))
    output = subprocess.call(command, shell=True, stdout=None)
    sample_rate, audio = wavfile.read(audio_tmp)
    print(audio.shape)

    # result
    predict_results = open(output_dir, "w")
    # predict_results = open(os.path.join(os.getcwd(), 'result', POI, POI + '-' + str(config.video_num) + '.txt'), "w")

    tracker_list = []
    candidates = []
    series_id = 0

    cap = cv2.VideoCapture(video_dir)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if config.enable_syncnet:
        assert video_fps == 25
    print("Video FPS:", video_fps)

    if config.write_video:
        videoWriter = cv2.VideoWriter(os.path.join(output_video_dir, 'song.avi'),
                                      cv2.VideoWriter_fourcc(*'XVID'), video_fps, (1280, 720))

    if cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 1280:
        need_to_resize = True
    else:
        need_to_resize = False

    # start_frame = 0
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # shot_count = start_frame - 1

    shot_count = 0

    print("start process")
    start_time = time.time()
    while True:
        if need_to_resize:
            success, raw_image = cap.read()
            if not success:
                break
            raw_image = cv2.resize(raw_image, (1280, 720))
        else:
            success, raw_image = cap.read()
        if not success:
            break
        image = raw_image.copy()
        bboxes, landmarks = face_detection_model.update(raw_image)
        new_tracker_list = []

        # track
        for tracker in tracker_list:
            tracked, bbox = tracker.update(raw_image, shot_count)
            # if target lost, start SyncNet process
            if tracked is False:
                if config.debug:
                    print("tracking failed")
                if config.enable_syncnet:
                    if config.debug:
                        print(16000 * tracker.start_shot // video_fps, 16000 * (tracker.end_shot) // video_fps)
                    part_audio = audio[
                                 int(16000 * tracker.start_shot // video_fps):int(
                                     16000 * (tracker.end_shot) // video_fps)]
                    if len(part_audio) != len(tracker.sync_seq) * 16000 // video_fps:
                        print("fatal: video and audio does not match")
                        print("startshot", tracker.start_shot)
                        print("endshot", tracker.end_shot)
                        print(tracker.series_name)
                        print(len(tracker.sync_seq))
                        print(len(part_audio))
                        # exit(-1)]
                    if config.debug:
                        wavfile.write('temp/segment.wav', 16000, part_audio)
                        player = AudioPlayer('temp/segment.wav')

                    offset, confidence, dists_npy = speaker_validation.evaluate(video_fps, tracker.sync_seq, part_audio)
                    silent_audio = np.zeros(part_audio.shape, dtype=audio.dtype)
                    __, conf_silent, __ = speaker_validation.evaluate(video_fps, tracker.sync_seq, silent_audio)
                    # print(conf_silent)
                    confidence[conf_silent > 3] = 0
                    # confidence = conf_silent

                    if config.debug:
                        print("Sequence length:", len(tracker.sync_seq))
                        debug_cap = cv2.VideoCapture(video_dir)
                        debug_cap.set(1, tracker.start_shot)
                        player.play()
                        for i in range(len(tracker.sync_seq)):
                            if i < 6:
                                if need_to_resize:
                                    __, img = debug_cap.read()
                                    img = cv2.resize(img, (1280, 720))
                                else:
                                    __, img = debug_cap.read()
                                cv2.imshow('Speaking', img)
                                cv2.waitKey(40)
                            else:
                                if need_to_resize:
                                    __, img = debug_cap.read()
                                    img = cv2.resize(img, (1280, 720))
                                else:
                                    __, img = debug_cap.read()
                                box = tracker.bbox_seq[i]
                                lip_box = tracker.lip_box_seq[i]
                                try:
                                    confidence_caption = 'Conf: %.3f' % (confidence[i - 6])
                                    clr = int(max(min(confidence[i - 6] * 30, 255), 0))
                                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, clr, 255 - clr), 2,
                                                  cv2.LINE_AA)
                                    cv2.rectangle(img, (lip_box[2], lip_box[0]), (lip_box[3], lip_box[1]), (255, 0, 0),
                                                  2, cv2.LINE_AA)
                                except:
                                    confidence_caption = 'Conf: exceeded'
                                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2,
                                                  cv2.LINE_AA)
                                cv2.putText(img, confidence_caption, (int(box[0]), int(box[1]) + 20),
                                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                                cv2.putText(img, confidence_caption, (int(box[0]), int(box[1]) + 20),
                                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                                cv2.imshow('Speaking', img)
                                cv2.waitKey(40)
                        cv2.waitKey(0)
                    prelabels = speaker_validation.verification(confidence, tracker.start_shot, predict_results)
                    candidates = candidates + prelabels
            else:
                new_tracker_list.append(tracker)
        tracker_list = new_tracker_list

        # for each face detected
        for boundary, landmark in zip(bboxes, landmarks):
            boundary = boundary.astype(np.int)
            center = [int((boundary[1] + boundary[3]) / 2), int((boundary[0] + boundary[2]) / 2)]
            validation = face_validation_model.confirm_validity(raw_image, boundary=boundary, landmark=landmark)

            if validation:
                caption = "Yes"
                tracking = isTracking((center[1], center[0]), tracker_list)
                lip_center = np.mean(landmark[3:], axis=0)
                # new target
                if not tracking:
                    series_id += 1
                    new_tracker = CV_Tracker(raw_image, boundary, series_id, lip_center, shot_count)
                    tracker_list.append(new_tracker)
                else:
                    for tracker in tracker_list:
                        if tracker.valid is True:
                            continue
                        if tracker.is_valid(center):
                            # build lip picture sequence
                            tracker.update_lip_seq(raw_image, boundary, lip_center)
            else:
                caption = "No"

            if config.showimg:
                cv2.rectangle(image, (boundary[0], boundary[1]), (boundary[2], boundary[3]), (0, 255, 0), 2,
                              cv2.LINE_AA)
                index_color = 0
                for point in landmark:
                    pos = (point[0], point[1])
                    cv2.circle(image, pos, 1, (255, 255, 255 / 68 * index_color), -1)
                    index_color = index_color + 1
                # lip center
                lip_center = np.mean(landmark[3:], axis=0)
                cv2.circle(image, (lip_center[0], lip_center[1]), 1, (0, 0, 0), -1)
                for tracker in tracker_list:
                    if tracker.tracked is True:
                        bbox = tracker.bbox
                        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                                      (int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])), (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, str(tracker.series_name), (int(bbox[0]), int(bbox[1]) - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                        cv2.putText(image, str(tracker.series_name), (int(bbox[0]), int(bbox[1]) - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                    else:
                        print("Warning a invalid tracker was not removed")
                cv2.putText(image, str(caption), (boundary[0], boundary[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 0, 0), 2)
                cv2.putText(image, str(caption), (boundary[0], boundary[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                            (255, 255, 255), 1)

        new_tracker_list = []
        for tracker in tracker_list:
            if tracker.valid is False:
                tracker.drop_count += 1
                tracker.update_lip_seq(raw_image, None, None)
            if tracker.drop():
                tracker.set_end_shot(shot_count)
                if config.debug:
                    print("tracker missed the target")
                if config.enable_syncnet:
                    part_audio = audio[int(16000 // video_fps * tracker.start_shot): int(
                        16000 // video_fps * (tracker.end_shot - config.patience + 1))]
                    if len(part_audio) != len(tracker.sync_seq[:-config.patience]) * 16000 // video_fps:
                        print("fatal: video and audio does not match")
                        print("startshot", tracker.start_shot)
                        print("endshot", tracker.end_shot)
                        print(len(tracker.sync_seq))
                        print(len(part_audio))
                        # exit(-2)
                    if config.debug:
                        wavfile.write('temp/segment.wav', 16000, part_audio)
                        player = AudioPlayer('temp/segment.wav')

                    offset, confidence, dists_npy = speaker_validation.evaluate(video_fps,
                                                                                tracker.sync_seq[:-config.patience],
                                                                                part_audio)
                    silent_audio = np.zeros(part_audio.shape, dtype=audio.dtype)
                    __, conf_silent, __ = speaker_validation.evaluate(video_fps, tracker.sync_seq[:-config.patience],
                                                                      silent_audio)
                    # print(conf_silent)
                    confidence[conf_silent > 3] = 0
                    # confidence = conf_silent

                    if config.debug:
                        print("Sequence length:", len(tracker.sync_seq[:-config.patience]))

                        debug_cap = cv2.VideoCapture(video_dir)
                        debug_cap.set(1, tracker.start_shot)
                        player.play()
                        for i in range(len(tracker.sync_seq) - config.patience):
                            if i < 6:
                                if need_to_resize:
                                    __, img = debug_cap.read()
                                    img = cv2.resize(img, (1280, 720))
                                else:
                                    __, img = debug_cap.read()
                                cv2.imshow('Speaking', img)
                                cv2.waitKey(40)
                            else:
                                if need_to_resize:
                                    __, img = debug_cap.read()
                                    img = cv2.resize(img, (1280, 720))
                                else:
                                    __, img = debug_cap.read()
                                box = tracker.bbox_seq[i]
                                lip_box = tracker.lip_box_seq[i]
                                try:
                                    confidence_caption = 'Conf: %.3f' % (confidence[i - 6])
                                    clr = int(max(min(confidence[i - 6] * 30, 255), 0))
                                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, clr, 255 - clr), 2,
                                                  cv2.LINE_AA)
                                    cv2.rectangle(img, (lip_box[2], lip_box[0]), (lip_box[3], lip_box[1]), (255, 0, 0),
                                                  2, cv2.LINE_AA)
                                except:
                                    confidence_caption = 'Conf: exceeded'
                                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2,
                                                  cv2.LINE_AA)
                                cv2.putText(img, confidence_caption, (int(box[0]), int(box[1]) + 20),
                                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                                cv2.putText(img, confidence_caption, (int(box[0]), int(box[1]) + 20),
                                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                                cv2.imshow('Speaking', img)
                                cv2.waitKey(40)
                        cv2.waitKey(0)
                    prelabels = speaker_validation.verification(confidence, tracker.start_shot, predict_results)
                    candidates = candidates + prelabels
            else:
                new_tracker_list.append(tracker)
        tracker_list = new_tracker_list

        if shot_count % 1000 == 0 and shot_count != 0:
            print('Shot {:d}, FPS {:.2f} '.format(shot_count, 1000 / (time.time() - start_time)), end='\n')
            start_time = time.time()
        if config.showimg:
            cv2.imshow('Video', image)
        if config.write_video:
            videoWriter.write(image)
        shot_count += 1

        if cv2.waitKey(10) == 27:
            break

    predict_results.close()
    # evaluate
    if config.enable_evaluation:
        index = video_dir.rfind('.')
        evaluate_result(video_dir[:index] + ".csv", output_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='basic run command')

    parser.add_argument('--POI', default='', help='the POI to start with')

    args = parser.parse_args()
    starting_POI = args.POI
    print(starting_POI)

    # global init
    face_detection_model, face_validation_model, speaker_validation = load_models()
    print("all model loaded")
    #
    if not os.path.exists(config.video_base_dir):
        print("fatal: invalid base video path")
    if not os.path.exists(config.image_base_dir):
        print("fatal: invalid base video path")
    if not os.path.exists(config.temp_dir):
        os.makedirs(config.temp_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    POIS = os.listdir(config.video_base_dir)
    if starting_POI != '':
        while POIS[0] != starting_POI:
            print("skipping {}".format(POIS[0]))
            POIS.pop(0)

    for POI in POIS:
        print("current POI: {}".format(POI))
        if not os.path.exists((os.path.join(config.image_base_dir, POI))):
            print("image of {} is not exist".format(POI))
            continue
        POI_imgs = [os.path.join(config.image_base_dir, POI, pic) for pic in
                    os.listdir(os.path.join(config.image_base_dir, POI))]
        POI_categories = os.listdir(os.path.join(config.video_base_dir, POI))

        face_validation_model.update_POI(POI_imgs)
        print("POI images updated")

        # 遍历文件下所有文件
        for category in POI_categories:
            category_video = os.path.join(config.video_base_dir, POI, category)
            for root, dirs, files in os.walk(category_video):
                for file in files:
                    if file.find('.csv') > 0 or file.find('.txt') > 0:
                        continue
                    index = file.rfind('.')
                    category_output = root.replace(config.video_base_dir, config.output_dir)
                    if not os.path.exists(category_output):
                        os.makedirs(category_output)
                    single_video_dir = os.path.join(root, file)
                    single_output_dir = os.path.join(category_output, file[:index] + '.txt')
                    print("Processing video: {}".format(single_video_dir))
                    try:
                        process_single_video(single_video_dir, single_output_dir, face_detection_model,
                                             face_validation_model, speaker_validation)
                    except AssertionError:
                        print("FPS of {} is not 25.".format(single_video_dir))
                    gc.collect()
