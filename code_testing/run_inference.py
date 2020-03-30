"""
Need the Inference graph from trained model folder in order for us to run evaluation for Test images

Command to create "inference_graph" from our trained model(ckpt file)
    Go to Tensorflow Object Detection API (/models/research/object_detection)
    $ python export_inference_graph.py --input_type image_tensor
              --pipeline_config_path training/pipeline.config  (config file from model folder downloaded from Tf model zoo)
              --trained_checkpoint_prefix training/faster_rcnn_inceptionv2/model.ckpt-116268
              --output_directory training/faster_rcnn_inceptionv2/inference_graph
Usage:
  Move this file into Tensorflow Object Detection APi folder (/model/research/object_detection)


Output: csv file consists of Object Detection score + True label (for each class) for every image we evaluate
       This file will be used for visualization.
"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.metrics import average_precision_score
from os import listdir
from os.path import isfile, join
import random
from tqdm import tqdm
import time
# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")
from object_detection.utils import ops as utils_ops
from sklearn.metrics import average_precision_score,f1_score,precision_recall_curve,roc_auc_score,precision_recall_curve

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

def run_inference_for_single_image(image, graph,tensor_dict,sess):
    # image = np.asarray(image)
    # image= np.array(Image.open(image_path))
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    # Run inference
    output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def get_score_per_class(df_label,pair_class_score):
    set_pair_class_score = set(pair_class_score)
    set_pair_class_score = sorted(set_pair_class_score,key=lambda x:x[1],reverse=True)
    final_list = [0,0,0,0,0] # 5 classes
    for pair_class_score in set_pair_class_score:
        score = pair_class_score[1]
        class_ = pair_class_score[0]
        if score == 0: # first score equal to 0, we return negative examples
            return final_list
        if final_list[class_-1] == 0:
            final_list[class_-1] = score
    return final_list
def get_list_of_files(path):
    list_files = [path+"/"+f for f in listdir(path) if isfile(join(path, f))]
    list_img_name = [f for f in listdir(path) if isfile(join(path, f))]
    return list_files, list_img_name
def get_list_class_label(df_label,list_img_name):
    assert all(df_label.index.isin(list_img_name)), "csv file does not contain specific image in image folder."
    print(df_label)
    print(list_img_name[:3])
    df_new = df_label.reindex(list_img_name)
    return df_new.values.tolist() , df_new

def main():
    PATH_TO_FROZEN_GRAPH = 'training/trained_faster_rcnn_inceptionv2/inference_graph/frozen_inference_graph.pb'
    PATH_TO_FROZEN_GRAPH = 'training/trained_ssd_inceptionv2/inference_graph/frozen_inference_graph.pb'
    PATH_TO_FROZEN_GRAPH = 'training/trained_ssd_resnet50/model_ssd_resnet50/inference_graph/frozen_inference_graph.pb'
    PATH_TO_FROZEN_GRAPH = 'training/trained_ssd_mobilenet_fpn/inference_graph/frozen_inference_graph.pb'
    PATH_TO_FROZEN_GRAPH = 'training/trained_rfcn_resnet101/inference_graph/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('training', 'labelmap.pbtxt')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    abs_img_path = 'D:/Coding/SFU_CA/CMPT-733/groupproject/dataset/new_dataset/'

    '''Add image path to the list_img_path for desired img folder'''
    # list_img_path = [abs_img_path+'positive_test', abs_img_path+'negative_test']
    list_img_path = [abs_img_path+'positive_test', abs_img_path+'negative_test',abs_img_path+'negative_train']
    # list_img_path = [abs_img_path+'positive_test', abs_img_path+'negative_test',abs_img_path+'negative_train',abs_img_path+'negative_img11']

    random.seed(time.time())
    list_images = []
    list_img_names = []
    for path in list_img_path:
        list_img,list_img_name = get_list_of_files(path)
        list_images = list_images + list_img
        list_img_names = list_img_names + list_img_name

    temp = list(zip(list_images,list_img_names))
    random.shuffle(temp)
    list_images,list_img_names = zip(*temp)

    TEST_IMAGE_PATHS = list_images

    ''' Path to label file (complete csv file that includes both train and test label)'''
    # df_label = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/dataset/label_test.csv',index_col=0)
    df_label = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/dataset/label_test_train.csv',index_col=0)
    # df_label = pd.read_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/dataset/label_test_train_neg11.csv',index_col=0)

    list_class_label,new_df_label = get_list_class_label(df_label,list_img_names)

    list_avg_precision_score = []
    list_true_class =[]
    list_score =[]
    big_list=[]
    try:
        with detection_graph.as_default():
            with tf.Session() as sess:
                i=0
                for image_path,image_name,true_class_label in tqdm(zip(TEST_IMAGE_PATHS,list_img_names,list_class_label)):
                    image_np = np.array(Image.open(image_path))

                    # Get handles to input and output tensors
                    ops = tf.get_default_graph().get_operations()
                    all_tensor_names = {output.name for op in ops for output in op.outputs}
                    tensor_dict = {}
                    for key in ['num_detections', 'detection_boxes', 'detection_scores',
                                'detection_classes', 'detection_masks']:
                        tensor_name = key + ':0'
                        if tensor_name in all_tensor_names:
                            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_path, axis=0)
                    # Actual detection.
                    output_dict = run_inference_for_single_image(image_np, detection_graph,tensor_dict,sess)

                    pair_class_score = list(zip(output_dict['detection_classes'], output_dict['detection_scores']))
                    score_per_class = get_score_per_class(df_label,pair_class_score)
                    print("true class: ",true_class_label)
                    print("score :",score_per_class)

                    list_true_class.append(true_class_label)
                    list_score.append(score_per_class)
                    big_list.append([image_name]+score_per_class+true_class_label)

                    print("Image name: ",image_name)

                    i += 1
                    if i == 100000:
                        break

                col_score = list(df_label)
                col_true_label = [s+'_label' for s in col_score]
                new_col = ['filename']+col_score+col_true_label
                df_score = pd.DataFrame.from_records(big_list,columns=new_col)
                df_score.to_csv('D:/Coding/SFU_CA/CMPT-733/groupproject/dataset/label_score.csv',index=None)

                precision = dict()
                recall = dict()
                average_precision = dict()
                np_true_class_label = np.asarray(list_true_class)
                np_score = np.asarray(list_score)
                for i in range(5):
                    precision[i], recall[i], _ = precision_recall_curve(np_true_class_label[:, i], np_score[:, i])
                    average_precision[i] = average_precision_score(np_true_class_label[:, i], np_score[:, i])
                print(average_precision)
                #print(average_precision){0: 0.9445920049228803, 1: 0.692395567524862, 2: 0.5938505212904481, 3: 0.7572509541965025, 4: 0.3149322294605498}

    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
