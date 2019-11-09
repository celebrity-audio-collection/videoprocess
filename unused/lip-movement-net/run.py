from src.lip_movement_net import *
import sys

sys.path.append('./')

test_video("../videos/interview-1.mp4", "../models/shape_predictor_68_face_landmarks.dat",
           "../models/2_64_False_True_0.5_lip_motion_net_model.h5")
