import tensorflow as tf
import numpy as np
from PIL import Image

import cv2

cap = cv2.VideoCapture("video\\aoe_iii_snow_hills.mp4")

from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

PATH_TO_FROZEN_GRAPH = "inference_graph\\frozen_inference_graph.pb"
PATH_TO_LABELS = "data/aoeiii_label_map.pbtxt"

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

image_path = "data\\bbox_train_hills_jpeg\\172.jpeg"
feed_image = Image.open(image_path)

# Memory management code

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.reset_default_graph()
sess = tf.Session(config=config)


_INPUT_NAME = 'image_tensor'
_OUTPUT_NAME = 'detection_boxes'

det_graph = sess.graph
od_graph_def = tf.GraphDef()

# read frozen graph
with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

all_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
print(all_nodes)

while True:
    ret, image_np = cap.read()
    _input = det_graph.get_tensor_by_name(_INPUT_NAME + ":0")

    boxes = det_graph.get_tensor_by_name(_OUTPUT_NAME + ":0")
    scores = det_graph.get_tensor_by_name('detection_scores:0')
    classes = det_graph.get_tensor_by_name('detection_classes:0')
    num_detections = det_graph.get_tensor_by_name('num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={_input: image_np_expanded})

    # print(boxes)
    # print(scores)
    # print(classes)
    print(num_detections)

    vis_util.visualize_boxes_and_labels_on_image_array(
              feed_image,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)

    cv2.imshow('AOE III', image_np)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
