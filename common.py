import os



POI = '白百何'



class Config:
    log_dir = './log/log.txt'

    video_num = 1
    video_dir = [os.path.join(os.getcwd(), 'videos', POI, file) for file in
                 os.listdir(os.path.join(os.getcwd(), 'videos', POI))][video_num - 1]
    image_files = [os.path.join(os.getcwd(), 'images', POI, file) for file in
                   os.listdir(os.path.join(os.getcwd(), 'images', POI))]

    landmark_predictor = "model/dlib/shape_predictor_68_face_landmarks.dat"

    exp_name = os.path.basename(log_dir)

    # visual
    showimg = False
    debug = False

    # RetinaFace
    detect_scale = [360, 640]
    retinaface_model = 'model/retinaface_model/mnet.25/mnet.25'
    thresh = 0.8
    gpuid = 0

    # Tracker
    tracker_type = 'MOSSE'
    patience = 8

    # face validation
    # FaceNet
    use_facenet = False
    face_validation_path = "model/facenet_model/20180402-114759"
    validation_imagesize = 160
    margin = 44
    threshold = 0.9

    # InsightFace
    if not use_facenet:
        use_insightface = True
    else:
        use_insightface = False
    mobilenet_dir = './model/insightface_model/mobilenet/model,0'
    cosine_threshold = 0.8
    dist_threshold = 1.24

    # speaker validation
    # SyncNet
    enable_syncnet = True

    syncnet_model = "./model/syncnet_v2.model"

    starting_confidence = 4
    patient_confidence = 3

    # evaluation
    enable_evaluation = True

    @property
    def input_shape(self):
        return (self.minibatch_size, self.nr_channel) + self.image_shape


config = Config()
