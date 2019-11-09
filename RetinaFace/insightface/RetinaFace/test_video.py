import cv2
import sys
import numpy as np
import datetime
import time
import os
import glob
sys.path.append('../../insightface')
from RetinaFace.retinaface import RetinaFace

thresh = 0.8
gpuid = 0
scales = [360, 640]
# detector = RetinaFace('./model/retinaface-R50/R50', 0, gpuid, 'net3')
detector = RetinaFace('./model/mnet.25/mnet.25', 0, gpuid, 'net3')


def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    im_shape = frame.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.0
    # if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    while True:
        start_time = time.clock()
        success, frame = cap.read()
        faces, landmarks = detector.detect(frame, thresh, scales=[im_scale], do_flip=False)

        if faces is not None:
            for i in range(faces.shape[0]):
                # print('score', faces[i][4])
                box = faces[i].astype(np.int)
                # color = (255,0,0)
                color = (0, 0, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                if landmarks is not None:
                    landmark5 = landmarks[i].astype(np.int)
                    # print(landmark.shape)
                    for l in range(landmark5.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(frame, (landmark5[l][0], landmark5[l][1]), 1, color, 2)
            end_time = time.clock()
            cv2.imshow('Frame', frame)
            print('Found', faces.shape[0], 'faces', 'FPS:', 1.0 / (end_time - start_time))
        else:
            cv2.imshow('Frame', frame)

        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__':
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    video_path = r'C:\Users\haoli\PycharmProjects\syncnet_python-master-pytorch\interview-1.mp4'
    detect_video(video_path)
