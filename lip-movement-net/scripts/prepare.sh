#!/bin/bash
clear
python ../src/prepare_training_dataset.py -i ../data/dataset_source -o ../data/dataset_dest -p ../models/shape_predictor_68_face_landmarks.dat
