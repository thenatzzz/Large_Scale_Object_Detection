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
# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")
from object_detection.utils import ops as utils_ops
from sklearn.metrics import average_precision_score,f1_score,precision_recall_curve,roc_auc_score,precision_recall_curve

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

import sys
sys.path.insert(0, 'D:/Coding/SFU_CA/CMPT-733/groupproject/models/research/object_detection/')

from utils import label_map_util
from utils import visualization_utils as vis_util

def run_inference_for_single_image(image, graph,tensor_dict,sess):
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


def main(): #p00458,p00118
    folder_model= 'incomplete_trained_ssd_mobilenet_v1' # 22.11,26
    # folder_model= 'trained_ssd_mobilenet_fpn'  # 29.16,29.66
    folder_model= 'trained_ssd_inceptionv2' # 29.12,30.720
    # folder_model= 'trained_faster_rcnn_inceptionv2'  #29.99,30.66
    # folder_model= 'trained_faster_rcnn_resnet50' #37.56,37.688
    # folder_model= 'trained_ssd_resnet50'   #39.76 ,40.575
    # folder_model= 'trained_faster_rcnn_resnet101' #51,48.7
    folder_model= 'trained_rfcn_resnet101'  # 51.36,51.89


    PATH_TO_FROZEN_GRAPH = 'D:/Coding/SFU_CA/CMPT-733/groupproject/report/trained_model/'+folder_model+'/inference_graph/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'labelmap.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    image= 'P00118.jpg'
    image= 'P00458.jpg'
    abs_img_path = 'D:/Coding/SFU_CA/CMPT-733/groupproject/dataset/new_dataset/positive_test/'
    image_path =  abs_img_path + image
    try:
        with detection_graph.as_default():
            with tf.Session() as sess:
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

                '''For Visualization (do not delete)'''
                '''Visualization of the results of a detection.'''
                vis_util.visualize_boxes_and_labels_on_image_array(
                                            image_np,
                                            output_dict['detection_boxes'],
                                            output_dict['detection_classes'],
                                            output_dict['detection_scores'],
                                            category_index,
                                            instance_masks=output_dict.get('detection_masks'),
                                            use_normalized_coordinates=True,
                                            line_thickness=8)
    except Exception as e:
        print(e)
    img = Image.fromarray(image_np, 'RGB')
    ''' To show one image '''
    # img.save('D:/Coding/SFU_CA/CMPT-733/groupproject/report/model_scores/show_image/'+folder_model+'_'+image)
    img.show()

if __name__ == '__main__':
    main()
