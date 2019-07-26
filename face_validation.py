import tensorflow as tf
import facenet_code.align.detect_face
from facenet_code import facenet
import copy
from scipy import misc
import numpy as np
import os
from common import config
class FaceValidation:


    def __init__(self, model_path = config.face_validation_path):

        self.image_list = []
        g_facenet = tf.Graph()
        sess_fecnet = tf.Session(graph=g_facenet)
        with sess_fecnet.as_default():
            with g_facenet.as_default():
                # Load the model
                facenet.load_model(model_path)
        self.graph = g_facenet
        self.sess = sess_fecnet

    def update_POI(self,imgdir_list):
        self.image_list = self.load_and_align_data(imgdir_list, config.validation_imagesize, config.margin)

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

    def Confirm_validity(self, img_in):
        processed_facepicture = self.process_cutted_image(img_in)
        image_dict = self.image_list.copy()
        image_dict.append(processed_facepicture)
        image_dict = np.stack(image_dict)

        with self.graph.as_default():
            with self.sess.graph.as_default():
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: image_dict, phase_train_placeholder: False}
                emb = self.sess.run(embeddings, feed_dict=feed_dict)

                avg = 0
                #此处更新计算合法性算法
                for i in range(len(emb)-1):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[-1, :]))))
                    # print("face validation：",dist)
                    avg += dist
                value = avg/(len(emb)-1)

        if value < config.threshold:
            return True
        else:
            return False



