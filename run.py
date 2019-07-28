# -*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common import config
from cv_tracker import CV_Tracker
from face_detection import FaceDetection
from face_validation import FaceValidation
from speaker_validation import SpeakerValidation
from evaluate import evaluate_result

import cv2
import subprocess
import numpy as np
from scipy.io import wavfile
import time

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


if __name__ == '__main__':

    # load detection model
    print("loading face detection model")
    face_detection_model = FaceDetection()

    # load face validation model
    print("loading face validation model")
    facenet_model = FaceValidation()
    facenet_model.update_POI(config.image_files)

    # SyncNet
    print("loading speaker validation model")
    speaker_validation = SpeakerValidation()

    # get wave file
    audio_tmp = './temp/audio.wav'
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (config.video_dir, audio_tmp))
    output = subprocess.call(command, shell=True, stdout=None)
    sample_rate, audio = wavfile.read(audio_tmp)

    print("all model loaded")

    # result
    predict_results = open("./testans.txt", "w")

    tracker_list = []
    candidates = []
    first_shot = True
    series_id = 0
    shot_count = 0

    cap = cv2.VideoCapture(config.video_dir)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Video FPS:", video_fps)

    for i in range(0):
        cap.read()

    while True:
        start_time = time.time()
        success, raw_image = cap.read()
        if not success:
            break

        raw_image = cv2.resize(raw_image, (1280, 720))
        bboxes, landmarks = face_detection_model.update(raw_image)
        new_tracker_list = []

        # track
        for tracker in tracker_list:
            tracked, bbox = tracker.update(raw_image, shot_count)
            # if target lost, start SyncNet process
            if tracked is False:
                if config.enable_syncnet:
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
                        # exit(-1)
                    offset, confidence, dists_npy = speaker_validation.evaluate(video_fps, tracker.sync_seq, part_audio)
                    if config.debug:
                        print("Sequence length:", len(tracker.sync_seq[:-config.patience]))
                        for i in range(len(tracker.raw_seq[:-config.patience])):
                            img = tracker.raw_seq[i].copy()
                            box = tracker.bbox_seq[i]
                            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2, cv2.LINE_AA)
                            try:
                                confidence_caption = 'Conf: %.3f' % (confidence[i])
                            except:
                                confidence_caption = 'Conf: exceeded'
                            cv2.putText(img, confidence_caption, (int(box[0]), int(box[1]) - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                            cv2.putText(img, confidence_caption, (int(box[0]), int(box[1]) - 10),
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
            validation = facenet_model.confirm_validity(raw_image, boundary=boundary, landmark=landmark)

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
                cv2.rectangle(raw_image, (boundary[0], boundary[1]), (boundary[2], boundary[3]), (0, 255, 0), 2,
                              cv2.LINE_AA)
                index_color = 0
                for point in landmark:
                    pos = (point[0], point[1])
                    cv2.circle(raw_image, pos, 1, (255, 255, 255 / 68 * index_color), -1)
                    index_color = index_color + 1
                # lip center
                lip_center = np.mean(landmark[3:], axis=0)
                cv2.circle(raw_image, (lip_center[0], lip_center[1]), 1, (0, 0, 0), -1)
                for tracker in tracker_list:
                    if tracker.tracked is True:
                        bbox = tracker.bbox
                        cv2.rectangle(raw_image, (int(bbox[0]), int(bbox[1])),
                                      (int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])), (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(raw_image, str(tracker.series_name), (int(bbox[0]), int(bbox[1]) - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                        cv2.putText(raw_image, str(tracker.series_name), (int(bbox[0]), int(bbox[1]) - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                    else:
                        print("Warning a invalid tracker was not removed")
                cv2.putText(raw_image, str(caption), (boundary[0], boundary[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 0, 0), 2)
                cv2.putText(raw_image, str(caption), (boundary[0], boundary[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                            (255, 255, 255), 1)

        new_tracker_list = []
        for tracker in tracker_list:
            if tracker.valid is False:
                tracker.drop_count += 1
                # tracker.update_lip_seq(raw_image, boundary, None)
            if tracker.drop():
                tracker.set_end_shot(shot_count)
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
                    offset, confidence, dists_npy = speaker_validation.evaluate(video_fps,
                                                                                tracker.sync_seq[:-config.patience],
                                                                                part_audio)
                    if config.debug:
                        print("Sequence length:", len(tracker.sync_seq[:-config.patience]))
                        for i in range(len(tracker.raw_seq[:-config.patience])):
                            img = tracker.raw_seq[i].copy()
                            box = tracker.bbox_seq[i]
                            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2, cv2.LINE_AA)
                            try:
                                confidence_caption = 'Conf: %.3f' % (confidence[i])
                            except:
                                confidence_caption = 'Conf: exceeded'
                            cv2.putText(img, confidence_caption, (int(box[0]), int(box[1]) - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                            cv2.putText(img, confidence_caption, (int(box[0]), int(box[1]) - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                            cv2.imshow('Speaking', img)
                            cv2.waitKey(40)
                        cv2.waitKey(0)
                    prelabels = speaker_validation.verification(confidence, tracker.start_shot, predict_results)
                    candidates = candidates + prelabels
            else:
                new_tracker_list.append(tracker)
        tracker_list = new_tracker_list

        print('Shot {:d}, FPS {:.2f} '.format(shot_count, 1 / (time.time() - start_time)), end=' ')
        if config.showimg:
            cv2.imshow('Video', raw_image)
        shot_count += 1

        first_shot = False
        if cv2.waitKey(10) == 27:
            break

    # evaluate
    if config.enable_evaluation:
        evaluate_result()
