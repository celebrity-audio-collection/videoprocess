#-*-coding:utf-8-*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common import config
from my_tracker import MyTracker
from face_detection import FaceDetection
from face_validation import FaceValidation
from speaker_validation  import FpeakerValidation


import os
import cv2
import dlib
import wave
import sys
import subprocess
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
import pandas as  pd
import time

import keras.backend.tensorflow_backend as KTF

gpuconfig = tf.ConfigProto()
gpuconfig.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=gpuconfig)
KTF.set_session(sess)
# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import models


def isTracking(boxcenter, trackerlist):
    for tracker in trackerlist:
        if tracker.is_tracking(boxcenter) is True:
            return True
    return  False

def processframe(video_fps,stringin):
    strcrop = stringin.split(":")
    frame  = 0
    frame += int(strcrop[0])*video_fps*60*60
    frame += int(strcrop[1])*video_fps*60
    frame += int(strcrop[2])*video_fps
    frame += int(strcrop[3])
    # frame = int(frame * video_fps/30.0)
    return frame


if __name__ == '__main__':

     # load detection model
    print("loading face detection model")
    face_detection_model = FaceDetection(config.face_detection_model)

    # load landmark predictor
    print("loading landmark predictor")
    predictor = dlib.shape_predictor(config.landmark_predictor)

    # load face validation model
    print("loading face validation model")
    facenet_model = FaceValidation()
    facenet_model.update_POI(config.image_files)

    # sync net
    print("loading speaker validation model")
    speakervalidation = FpeakerValidation();

    # lip movement
    # print("loading lip movement detector")
    # lip_movement_modelpath="D:\\CSLT\\Retinanet_face\\keras-retinanet\\model\\lip_movement_model\\2_64_False_True_0.5_lip_motion_net_model.h5"
    # lip_movement_model = keras.models.load_model(lip_movement_modelpath)
    # print("loading lip movement detector finished")
    # serienames = []

    cap = cv2.VideoCapture(config.video_dir)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print("video fps ", video_fps)
    #获取音轨
    audiotmp = './temp/audio.wav'
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (config.video_dir, audiotmp))
    output = subprocess.call(command, shell=True, stdout=None)
    sample_rate, audio = wavfile.read(audiotmp)
    print(audio)



    print("all model loaded")



    # init local variables
    predict_results = open("./testans.txt", "w")

    trackerlist = []
    canditates = []

    firstshot = True

    serieid = 0
    shotcount = 0
    for i in range(570):
        cap.read()
    while True:
        tS = time.time()
        ret, raw_image = cap.read()
        if ret == False:
            break

        raw_image = cv2.resize(raw_image, (960, 540))
        image = raw_image.copy()
        selection = face_detection_model.update(image)
        img_split= raw_image.copy()
        newtrackerlist = []

        ## track
        for tracker in trackerlist:
            tracked, bbox = tracker.update(img_split, shotcount)
            # if(tracker.serie_name=="series2"):
            #     print("series2",len(tracker.syncseq),"  ",shotcount - tracker.startshot,"  ",shotcount )
            #     print("series2 tracked? ",tracked )
            # print("tracked? ", tracked)
            if tracked is False:
                print(trackerlist)
                print(16000 * tracker.startshot//video_fps, 16000 * (tracker.endshot)//video_fps)
                part_audio = audio[int(16000 * tracker.startshot//video_fps):int(16000 * (tracker.endshot)//video_fps)]
                if (len(part_audio) != len(tracker.syncseq) * 16000//video_fps):
                    print("fatal: video and audio does not match")
                    print("startshot", tracker.startshot)
                    print("endshot", tracker.endshot)
                    print(tracker.serie_name)
                    print(len(tracker.syncseq))
                    print(len(part_audio))
                    exit(-1)
                offset, confidence, dists_npy = speakervalidation.evaluate(video_fps, tracker.syncseq,part_audio)
                prelabels = speakervalidation.verification(confidence, tracker.startshot, predict_results)
                canditates = canditates + prelabels
            else:
                newtrackerlist.append(tracker)
        trackerlist = newtrackerlist
        # print(selection)
        for boundary in selection:
            for i in range(len(boundary)):
                boundary[i] = int(boundary[i])
            # print(type(boundary[0]))
            # print(boundary[0])
            if(boundary[0] < 0  or boundary[2]>raw_image.shape[1] or boundary[1] <0 or boundary[3]>raw_image.shape[0]):
                continue

            length = int(max(boundary[3]-boundary[1],boundary[2]-boundary[0])/2)
            center = [int((boundary[1]+boundary[3])/2),int((boundary[0]+boundary[2])/2)]

            facepicture = img_split[max(center[0]-length,0):center[0]+length, max(center[1]-length,0):center[1]+length, :]
            tt = time.time()
            validation = facenet_model.Confirm_validity(facepicture)
            print("face validation cost:",  time.time() - tt)
            ## landmark prediction
            pre_b = dlib.rectangle(int(boundary[0]), int(boundary[1]), int(boundary[2]), int(boundary[3]))
            shape = predictor(raw_image, pre_b)
            np_shape = []
            for key_point in shape.parts():
                np_shape.append([key_point.x, key_point.y])
            np_shape = np.array(np_shape)

            if validation:
                caption = "yes"
                tracking = isTracking((center[1],center[0]), trackerlist)
                if not tracking:
                    serieid += 1
                    lipcenter = np.mean(np_shape[61:67], axis=0)
                    newtracker = MyTracker(img_split,boundary, serieid, lipcenter, shotcount)
                    trackerlist.append(newtracker)
                else:
                    for tracker in trackerlist:
                        if(tracker.valid is True):
                            continue
                        if tracker.is_valid(center):
                            # build lip picture sequence
                            lipcenter = np.mean(np_shape[61:67], axis=0)
                            tracker.update_lip_seq(img_split, boundary, lipcenter)
            else:
                caption = "no"
            if config.showimg:
                cv2.rectangle(raw_image, (boundary[0], boundary[1]), (boundary[2], boundary[3]), (0, 255, 0), 2, cv2.LINE_AA)
                index_color = 0
                for point in np_shape:
                    pos = (point[0], point[1])
                    cv2.circle(raw_image, pos, 1, (255, 255, 255 / 68 * index_color), -1)
                    index_color = index_color + 1
                for tracker in trackerlist:
                    if tracker.tracked is True:
                        bbox = tracker.bbox
                        cv2.rectangle(raw_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])), (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(raw_image, str(tracker.serie_name), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                        cv2.putText(raw_image, str(tracker.serie_name), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                    else:
                        print("Warning a invalid tracker was not removed")
                cv2.putText(raw_image, str(caption), (boundary[0], boundary[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                cv2.putText(raw_image, str(caption), (boundary[0], boundary[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            # cv2.putText(raw_image, str(face_series_name), (b[2], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        newtrackerlist = []
        for tracker in trackerlist:
            if tracker.valid is False:
                tracker.drop_count +=1
                tracker.update_lip_seq(img_split, boundary, None)
            if tracker.drop():
                tracker.set_endshot(shotcount)
                part_audio = audio[int(16000 // video_fps * tracker.startshot) : int(16000 // video_fps * (tracker.endshot -config.patience + 1)) ]
                if (len(part_audio) != len(tracker.syncseq[:-config.patience]) * 16000//video_fps):
                    print("fatal: video and audio does not match")
                    print("startshot", tracker.startshot)
                    print("endshot", tracker.endshot)
                    print(len(tracker.syncseq))
                    print(len(part_audio))
                    exit(-2)
                offset, confidence, dists_npy = speakervalidation.evaluate(video_fps, tracker.syncseq[:-config.patience], part_audio)
                prelabels = speakervalidation.verification(confidence, tracker.startshot, predict_results)
                canditates = canditates + prelabels

            else:
                newtrackerlist.append(tracker)
        trackerlist = newtrackerlist
        # print("shotcount ", shotcount)
        print('shot {:d}: Compute time {:.6f} sec.'.format(shotcount, (time.time() - tS)))
        if config.showimg:
            cv2.imshow('raw_image', raw_image)
        shotcount += 1

        firstshot = False
        if cv2.waitKey(10) == 27:
            break
    ## evaluate
    truelable = pd.read_csv("F:\\白百何\\interview\\interview-1\\interview-1.csv", encoding="utf8")
    truelable = truelable[["入点","出点"]].values
    print(truelable)

    for row_index in range(len(truelable)):
        for col_index in range(len(truelable[row_index])):
            truelable[row_index][col_index] = processframe(video_fps,truelable[row_index][col_index])

    print(truelable)
    missed = 0
    total = 0
    preset = []
    valset = []
    for pair in canditates:
        preset += [i for i in range(pair[0], pair[1])]

    for pair in truelable:
        valset += [i for i in range(pair[0], pair[1])]


    preset = set(preset)
    valset = set(valset)

    FPR = len(preset - valset) / len(preset)
    recall = len(preset & valset) / len(valset)
    print("total valid  frames: ", total)
    print("total missed  frames: ", missed)
    print ("FPR: ", FPR)
    print("Recall: ", recall)

