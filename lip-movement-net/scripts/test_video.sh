#!/bin/bash
clear
exec conda activate tf
python src/lip_movement_net.py -v videos/interview-1.mp4 -p models/shape_predictor_68_face_landmarks.dat -m models/1_32_False_True_0.25_lip_motion_net_model.h5
