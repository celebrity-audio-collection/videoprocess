import face_model
import argparse
import cv2
import sys
import numpy as np
import time

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/mobilenet/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()



model = face_model.FaceModel(args)
img1 = cv2.imread('Tom_Hanks_2.png')
img1 = model.get_input(img1)
f1 = model.get_feature(img1)
# print(f1[0:10])
# gender, age = model.get_ga(img1)
# print(gender)
# print(age)
# sys.exit(0)

start_time = time.clock()

for i in range(1):
    img2 = cv2.imread('Tom_Hanks.png')
    img2 = cv2.resize(img2, (112, 112))
    img2 = model.get_input(img2)
    f2 = model.get_feature(img2)

end_time = time.clock()
print("Time: ", end_time - start_time)

img3 = cv2.imread('Tom_Cruise.png')
img3 = cv2.resize(img3, (112, 112))
img3 = model.get_input(img3)
f3 = model.get_feature(img3)

dist = np.sum(np.square(f1-f2))
print(dist)
sim = np.dot(f1, f2.T)
print(sim)

dist = np.sum(np.square(f1-f3))
print(dist)
sim = np.dot(f1, f3.T)
print(sim)

# diff = np.subtract(f1, f2)
# dist = np.sum(np.square(diff), 1)
