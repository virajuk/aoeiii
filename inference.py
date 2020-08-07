import tensorflow as tf
import os
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2

sys.path.append("..")
from object_detection.utils import ops as utils_ops

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = '/home/viraj-uk/Documents/exp01'
# MODEL_NAME = '/home/viraj-uk/Documents/aoe3/models/exp15'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/viraj-uk/Documents/aoe3/models/exp15/damage_label_map.pbtxt'


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)

      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]

  return output_dict

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def main():
    # for image_path in TEST_IMAGE_PATHS:

    # image_path = '/home/viraj-uk/Documents/aoe3/data/test_im/98.jpeg'
    # image_path = '/home/viraj-uk/Pictures/aoe.png'
    image_path = '/home/viraj-uk/Pictures/Untitled.png'
    image = Image.open(image_path)
    # image_np = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    # im_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # image.show()
    #
    # print(image_np.shape)
    # print(type(image))

    # cv2.imshow('hello',image_np)
    # cv2.waitKey()


    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    #
    # # print(len(output_dict['detection_boxes']))
    # # print(output_dict['detection_classes'])
    # # print(output_dict['detection_scores'])
    # # print(output_dict.get('detection_masks'))
    #
    # # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        # min_score_thresh=.9,
        # instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    #
    new_im = Image.fromarray(image_np)
    new_im.show()

    # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # print(image_np.shape)
    # print(len(output_dict['detection_boxes']))
    # print(len(output_dict['detection_classes']))
    # print(output_dict['detection_scores'])

    # cv2.imshow('hello', image_np)
    # cv2.waitKey()

# cap = cv2.VideoCapture("/home/viraj-uk/Documents/aoe3_bck/video/aoe_iii_snow_hills.mp4")
main()