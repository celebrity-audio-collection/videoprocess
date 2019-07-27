import os

POI = '白百何'

class Config:
    log_dir = './log/log.txt'

    video_num = 1
    video_dir = [os.path.join(os.getcwd(), 'videos', POI, file) for file in
                 os.listdir(os.path.join(os.getcwd(), 'videos', POI))][video_num - 1]
    image_files = [os.path.join(os.getcwd(), 'images', POI, file) for file in
                   os.listdir(os.path.join(os.getcwd(), 'images', POI))]

    face_detection_model = 'model/face_detector.resnet50_retinanet.inference.h5'
    landmark_predictor = "model/dlib/shape_predictor_68_face_landmarks.dat"
    face_validation_path = "model/facenet_model/20180402-114759"

    # lip_movement_modelpath = r"D:\\CSLT\\Retinanet_face\\keras-retinanet\\model\\lip_movement_model\\2_64_False_True_0.5_lip_motion_net_model.h5"
    syncnet_model = r"C:\Users\haoli\PycharmProjects\syncnet_python-master-pytorch\data\syncnet_v2.model"
    exp_name = os.path.basename(log_dir)

    # visual
    # debug = True
    showimg = True

    # Retinaface
    thresh = 0.8
    gpuid = 0

    # Tracker
    tracker_type = 'MOSSE'
    patience = 8

    # face validation
    ## face net
    use_facenet = False
    validation_imagesize = 160
    margin = 44
    threshold = 0.9

    ## insight face
    if use_facenet == False:
        use_insightface = True
    else:
        use_insightface = False
    mobilenet_dir = './model/insightface_model/mobilenet/model,0'
    cosine_threshold = 0.8
    dist_threshold = 1.24

    # speaker validation
    enable_syncnet = True
    starting_confidence = 4
    patient_confidence = 3

    # evaluation
    enable_evaluation = True

    @property
    def input_shape(self):
        return (self.minibatch_size, self.nr_channel) + self.image_shape


config = Config()
