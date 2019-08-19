import os


class Config:
    # log 以及临时文件夹路径
    log_dir = './log'
    temp_dir = './temp'

    # 切分标签输出路径，可以输出到与视频相同的文件夹下，文件路径结构与视频向相同
    output_dir = './result'

    # 视频及文图根路径，文件结构为
    # {video_base_dir}/名人/类别/该类别视频
    # {image_base_dir}/名人/该名人所有照片
    #
    video_base_dir = "/work4/zhangpengyuan/videos-finecut"
    image_base_dir = "/work4/chengsitong/cslt/zzy/poi"
    wav_output_dir = "/work4/zhangpengyuan/wav_results_eps1.5"
    # video_num = 1
    # video_dir = [os.path.join(os.getcwd(), 'videos', POI, file) for file in
    #              os.listdir(os.path.join(os.getcwd(), 'videos', POI))][video_num - 1]
    # image_files = [os.path.join(os.getcwd(), 'images', POI, file) for file in
    #                os.listdir(os.path.join(os.getcwd(), 'images', POI))]

    # dlib landmark predictor
    landmark_predictor = "model/dlib/shape_predictor_68_face_landmarks.dat"

    # switches
    showimg = False
    debug = False
    write_video = False

    # RetinaFace
    detect_scale = [360, 640]
    retinaface_model = 'model/retinaface_model/mnet.25/mnet.25'
    thresh = 0.8
    gpuid = 0

    # CV_Tracker
    # 在tracker内部区域中没有POI的情况下，tracker继续追踪的帧数
    tracker_type = 'MOSSE'
    patience = 5

    # face validation
    # 可选择retinaface 或者 facenet
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
    # cosine 距离阈值
    cosine_threshold = 0.8
    # 欧氏距离阈值
    dist_threshold = 1.24

    # speaker validation
    # SyncNet
    enable_syncnet = True

    syncnet_model = "./model/syncnet_v2.model"

    # 起始点阈值与持续阈值，当syncnet输出的confidence高于starting_confidence时，视为POI开始说话
    # 不低于patient_confidence时判断为连续刷说话，低于patient_confidence时视为说话中断。
    starting_confidence = 4  # default
    patient_confidence = 3  # default
    enable_multi_conf = False   # 是否使用多分类阈值

    # 空音轨中置信度高于多少，原音轨中计算出的相应置信度会被置零
    conf_silent_threshold = 3

    # 不同类型视频的syncnet阈值
    # easy是指简单场景下的阈值，比如interview、speech
    # hard是指困难场景下的阈值，比如entertain
    easy_starting_confidence = 4.3
    easy_patient_confidence = 2.7
    normal_starting_confidence = 4.8
    normal_patient_confidence = 2.8
    hard_starting_confidence = 5.4
    hard_patient_confidence = 3

    # evaluation
    enable_evaluation = True

    # data clean
    # poi_choose_threshold: 声纹中每个人的片段与syncnet重合率要达到多少才可能被认为是POI
    # segment_choose_threshold: 声纹片段与syncnet片段重合率要达到多少才会被选用
    # poi_num: 声纹片段的候选人物中，有几个会被认为可能是POI
    # concat_threshold: 在最后的结果里，两个片段间隔多少会被合并
    # trim_frame: 声纹识别得到的标签会掐头去尾的时间（当该片段时间不低于一个值的时候）
    # proportion: 当获得视频帧数小于原视频总帧数多少时，删除这个txt
    enable_dataclean = False
    poi_choose_threshold = 0.1
    segment_choose_threshold = 0.5
    poi_num = 2
    concat_threshold = 7
    trim_frame = 3
    proportion = 0.0


config = Config()
