# -*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common import *
from cv_tracker import CV_Tracker
from face_detection import FaceDetection
from face_validation import FaceValidation
from speaker_validation import SpeakerValidation
from evaluate import *

import cv2
import subprocess
import gc
import numpy as np
from scipy.io import wavfile
import time
import argparse

# log
fpr = {"entertain": 0, "interview": 0, "song": 0, "act": 0, "live": 0, "recite": 0, "speech": 0, "vlog": 0, "tvs": 0,
       "movie": 0}
recall = {"entertain": 0, "interview": 0, "song": 0, "act": 0, "live": 0, "recite": 0, "speech": 0, "vlog": 0, "tvs": 0,
          "movie": 0}
num = {"entertain": 0, "interview": 0, "song": 0, "act": 0, "live": 0, "recite": 0, "speech": 0, "vlog": 0, "tvs": 0,
       "movie": 0}
names = ["entertain", "interview", "song", "act", "live", "recite", "speech", "vlog", "tvs", "movie"]

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


'''
    @requires video_dir != None, output_dir != None, face_detection_model !=  None, face_validation_model != None, speaker_validation != None,
    @effects  处理单个视频，输出切分标记到output_dir
'''


def process_single_video(video_dir, output_dir, face_detection_model, face_validation_model, speaker_validation,
                         output_video_dir=None):
    # 将视频音轨导出到临时文件夹中，采样率为16000
    audio_tmp = os.path.join(config.temp_dir, 'audio.wav')
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s > %s 2>&1" % (
        video_dir, audio_tmp, os.path.join(config.log_dir, "ffmpeg.log")))
    output = subprocess.call(command, shell=True, stdout=None)
    sample_rate, audio = wavfile.read(audio_tmp)
    # print(audio.shape)

    # 打开标签输出文件
    predict_results = open(output_dir, "w")
    # predict_results = open(os.path.join(os.getcwd(), 'result', POI, POI + '-' + str(config.video_num) + '.txt'), "w")

    # 初始化临时变量
    tracker_list = []
    candidates = []
    series_id = 0

    # 验证视频帧数
    cap = cv2.VideoCapture(video_dir)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if config.enable_syncnet:
        assert video_fps == 25
    print("Video FPS:", video_fps)

    # 是否输出视频，若需要，则需要传入额外参数output_video_dir
    if config.write_video:
        videoWriter = cv2.VideoWriter(os.path.join(output_video_dir, 'song.avi'),
                                      cv2.VideoWriter_fourcc(*'XVID'), video_fps, (1280, 720))

    # 视频宽度大于1280时，缩放至 1280 * 720
    if cap.get(cv2.CAP_PROP_FRAME_WIDTH) > 1280:
        need_to_resize = True
    else:
        need_to_resize = False

    # 跳读n帧 debug 过程中使用，实际运行中不可跳读
    # start_frame = 0
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    # shot_count = start_frame - 1

    shot_count = 0

    print("\033[94mstart process\033[0m")
    video_type = video_dir.split("/")[-2]
    if video_type == "interview" or video_type == "speech":
        config.starting_confidence = config.easy_starting_confidence
        config.patient_confidence = config.easy_patient_confidence
    elif video_type == "entertain":
        config.starting_confidence = config.hard_starting_confidence
        config.patient_confidence = config.hard_patient_confidence
    else:
        config.starting_confidence = config.normal_starting_confidence
        config.patient_confidence = config.normal_patient_confidence
    print("\033[94mthreshold:  %s & %s\033[0m" % (str(config.starting_confidence), str(config.patient_confidence)))
    start_time = time.time()
    while True:
        # resize
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

        # track
        new_tracker_list = []
        for tracker in tracker_list:
            tracked, bbox = tracker.update(raw_image, shot_count)
            # if target lost, start SyncNet process
            if tracked is False:
                if config.debug:
                    print("tracking failed")
                if config.enable_syncnet:
                    if config.debug:
                        print(16000 * tracker.start_shot // video_fps, 16000 * (tracker.end_shot) // video_fps)

                    # 默认视频帧速率为25时，截取相应音频并验证长度是否合规
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

                    # 分别使用原音轨和空音轨调用 syncnet，对于空音轨中置信度高于3 的部分，将原音轨中计算出的相应置信度置零
                    offset, confidence, dists_npy = speaker_validation.evaluate(video_fps, tracker.sync_seq, part_audio)
                    silent_audio = np.zeros(part_audio.shape, dtype=audio.dtype)
                    __, conf_silent, __ = speaker_validation.evaluate(video_fps, tracker.sync_seq, silent_audio)
                    # print(conf_silent)
                    confidence[conf_silent > 2.5] = 0
                    # confidence = conf_silent

                    # debug 模式下输出额外信息
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
                # 验证该人脸是否已经被某个追踪器追踪
                tracking = isTracking((center[1], center[0]), tracker_list)
                lip_center = np.mean(landmark[3:], axis=0)
                # new target
                if not tracking:
                    series_id += 1
                    new_tracker = CV_Tracker(raw_image, boundary, series_id, lip_center, shot_count)
                    tracker_list.append(new_tracker)
                else:
                    # 验证追踪器是否追踪到该人脸
                    for tracker in tracker_list:
                        if tracker.valid is True:
                            continue
                        if tracker.is_valid(center):
                            # build lip picture sequence
                            tracker.update_lip_seq(raw_image, boundary, lip_center)
            else:
                caption = "No"

            # showimg 模式下输出人脸检测，识别，追踪信息
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

        # 对于追踪区域中没有人脸的tracker，判断是否需要关闭tracker
        new_tracker_list = []
        for tracker in tracker_list:
            if tracker.valid is False:
                tracker.drop_count += 1
                tracker.update_lip_seq(raw_image, None, None)
            if tracker.drop():
                tracker.set_end_shot(shot_count)
                if config.debug:
                    print("tracker missed the target")
                # 关闭 tracker 前，处理tracker保存的视频序列
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
                    confidence[conf_silent > 2.5] = 0
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

        # 跳出循环
        if cv2.waitKey(10) == 27:
            break

    predict_results.close()

    dataclean(output_dir)

    # evaluate
    if config.enable_evaluation:
        index = video_dir.rfind('.')
        FPR, Recall = evaluate_result(video_dir[:index] + ".csv", output_dir)
        try:
            fpr[video_type] += FPR
            recall[video_type] += Recall
            num[video_type] += 1
        except:
            print("\033[91man uncommen type: %s\033[0m" % (video_type))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='basic run command')
    parser.add_argument('--POI', default='', help='the POI to start with')
    args = parser.parse_args()
    starting_POI = args.POI
    print("start at:", starting_POI)

    # global init
    face_detection_model, face_validation_model, speaker_validation = load_models()
    print("\033[94mall model loaded\033[0m")
    #
    if not os.path.exists(config.video_base_dir):
        print("\033[91mfatal: invalid base video path\033[0m")
    if not os.path.exists(config.image_base_dir):
        print("\033[91mfatal: invalid base video path\033[0m")
    if not os.path.exists(config.temp_dir):
        os.makedirs(config.temp_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # 根据文件路径结构获取视频以及POI照片，开始批处理
    POIS = os.listdir(config.video_base_dir)
    # 跳跃至初始POI，用于越过处理完成的POI
    if starting_POI != '':
        while POIS[0] != starting_POI:
            print("\033[94mskipping {}\033[0m".format(POIS[0]))
            POIS.pop(0)

    for POI in POIS:
        print("\033[96mcurrent POI: {}\033[0m".format(POI))
        if not os.path.exists((os.path.join(config.image_base_dir, POI))):
            print("\033[91mimage of {} is not exist\033[0m".format(POI))
            continue
        POI_imgs = [os.path.join(config.image_base_dir, POI, pic) for pic in
                    os.listdir(os.path.join(config.image_base_dir, POI))]
        POI_categories = os.listdir(os.path.join(config.video_base_dir, POI))

        face_validation_model.update_POI(POI_imgs)
        print("\033[94mPOI images updated\033[94m")

        # 遍历文件下所有文件
        for category in POI_categories:
            category_video = os.path.join(config.video_base_dir, POI, category)
            for root, dirs, files in os.walk(category_video):
                for file in files:
                    if file.find('.csv') > 0 or file.find('.txt') > 0 or file.find(".wav") > 0:
                        continue
                    index = file.rfind('.')
                    category_output = root.replace(config.video_base_dir, config.output_dir)
                    if not os.path.exists(category_output):
                        os.makedirs(category_output)
                    single_video_dir = os.path.join(root, file)
                    single_output_dir = os.path.join(category_output, file[:index] + '.txt')
                    print("\033[93mProcessing video: {}\033[0m".format(single_video_dir))
                    filename, filetype = os.path.splitext(file)
                    try:
                        process_single_video(single_video_dir, single_output_dir, face_detection_model,
                                             face_validation_model, speaker_validation)
                    except AssertionError:
                        print("\033[91mFPS of {} is not 25.\033[0m".format(single_video_dir))
                    gc.collect()

    print("\033[94mcomplete\033[0m")
    for name in names:
        if num[name] == 0:
            continue
        FPR = fpr[name] / num[name]
        Recall = recall[name] / num[name]
        print("\033[92m%s has number: %s\tFPR: %s\tRecall: %s\033[0m" % (name, num[name], FPR, Recall))
