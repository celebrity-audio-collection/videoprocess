import copy
from scipy import misc
from scipy.spatial.distance import euclidean
import sklearn
import numpy as np
import os
from common import config
import argparse
from RetinaFaceModel.insightface.deploy.face_model import FaceModel
from RetinaFaceModel.insightface.src.common.face_preprocess import preprocess
import cv2

if config.use_facenet:
    import tensorflow as tf
    import facenet_code.align.detect_face
    from facenet_code import facenet


class FaceValidation:

    def __init__(self, model_path=config.face_validation_path):

        if config.use_insightface:
            parser = argparse.ArgumentParser(description='face model test')
            # InsightFace
            parser.add_argument('--image-size', default='112,112', help='')
            parser.add_argument('--model', default=config.mobilenet_dir, help='path to load model.')
            parser.add_argument('--ga-model', default='', help='path to load model.')
            parser.add_argument('--gpu', default=config.gpuid, type=int, help='gpu id')
            parser.add_argument('--det', default=0, type=int,
                                help='mtcnn option, 1 means using R+O, 0 means detect from begining')
            parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
            parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
            args = parser.parse_args()

            self.valmodel = FaceModel(args)

        elif config.use_facenet:
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph)
            with self.sess.as_default():
                with self.graph.as_default():
                    # Load the model
                    facenet.load_model(model_path)

        self.image_list = []
        self.labelembds = []

    def update_POI(self, imgdir_list):
        self.labelembds = []
        if config.use_insightface:
            tmp_image_paths = copy.copy(imgdir_list)
            features = []
            for image in tmp_image_paths:
                img = misc.imread(os.path.expanduser(image), mode='RGB')
                img1 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                img1 = self.valmodel.get_input(img1)
                features.append(self.valmodel.get_feature(img1).reshape(1, -1))
            self.labelembds = features
        elif config.use_facenet:
            self.image_list = self.load_and_align_data(imgdir_list, config.validation_imagesize, config.margin)
            self.labelembds = self.compute_embedings(self.image_list)

    def load_and_align_data(self, image_paths, image_size, margin):
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor

        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = facenet_code.align.detect_face.create_mtcnn(sess, None)

        tmp_image_paths = copy.copy(image_paths)
        img_list = []
        for image in tmp_image_paths:
            img = misc.imread(os.path.expanduser(image), mode='RGB')
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = facenet_code.align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                                           factor)
            if len(bounding_boxes) < 1:
                image_paths.remove(image)
                print("can't detect face, remove ", image)
                continue

            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
        sess.close()
        # images = np.stack(img_list)
        return img_list

    def process_cutted_image(self, raw_img):
        # process deceted
        aligned = misc.imresize(raw_img, (config.validation_imagesize, config.validation_imagesize), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        return prewhitened

    def compute_embedings(self, img_processed_picture):
        image_dict = np.stack(img_processed_picture)
        with self.graph.as_default():
            with self.sess.graph.as_default():
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: image_dict, phase_train_placeholder: False}
                embd = self.sess.run(embeddings, feed_dict=feed_dict)

        return embd

    def find_cosine_distance(self, vector1, vector2):
        """
        Calculate cosine distance between two vector
        """
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def cal_distance(self, target, source):
        target = sklearn.preprocessing.normalize(target)
        source = sklearn.preprocessing.normalize(source)
        return euclidean(target, source)

    def confirm_validity(self, raw_image, boundary, landmark):

        if config.use_insightface:
            processed_face_pic = preprocess(raw_image, bbox=boundary, landmark=landmark, image_size='112,112')
            processed_face_pic = cv2.cvtColor(processed_face_pic, cv2.COLOR_BGR2RGB)
            processed_face_pic = np.transpose(processed_face_pic, (2, 0, 1))
            embedding = self.valmodel.get_feature(processed_face_pic).reshape(1, -1)
            avg = 0
            dist_list = []
            for i in range(len(self.labelembds)):
                dist = self.cal_distance(embedding, self.labelembds[i])
                dist_list.append(dist)
                avg += dist
            value = avg / (len(self.labelembds))
            # if config.debug:
            # print("dist_avg: {:.3f}".format(value))
            if value < config.dist_threshold:
                return True
            else:
                return False

        elif config.use_facenet:
            if boundary[0] < 0 or boundary[2] > raw_image.shape[1] or \
                    boundary[1] < 0 or boundary[3] > raw_image.shape[0]:
                return False
            length = int(max(boundary[3] - boundary[1], boundary[2] - boundary[0]) / 2)
            center = [int((boundary[1] + boundary[3]) / 2), int((boundary[0] + boundary[2]) / 2)]
            facepicture = raw_image[max(center[0] - length, 0):center[0] + length,
                          max(center[1] - length, 0):center[1] + length, :]
            processed_face_pic = self.process_cutted_image(facepicture)
            picembd = self.compute_embedings([processed_face_pic])
            avg = 0
            # 此处更新计算合法性算法
            for i in range(len(self.labelembds)):
                dist = np.sqrt(np.sum(np.square(np.subtract(self.labelembds[i, :], picembd[0, :]))))
                avg += dist
            value = avg / (len(self.labelembds))

            if value < config.threshold:
                return True
            else:
                return False
