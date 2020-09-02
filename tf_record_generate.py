"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import hashlib
import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
# from object_detection.utils import dataset_util
from util import tf_utils as dataset_util
from collections import namedtuple, OrderedDict



flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'explorer':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    key = hashlib.sha256(encoded_jpg).hexdigest()

    filename = group.filename.encode('utf8')

    # print(filename)

    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    truncated = []
    poses = []
    difficult_obj = []

    for index, row in group.object.iterrows():

        attributes = {}
        attributes = eval(row['region_shape_attributes'])
        label_dic = {}
        label_dic = eval(row['region_attributes'])

        xmins.append(attributes.get('x') / width)
        xmaxs.append((attributes.get('x') + attributes.get('width')) / width)
        ymins.append(attributes.get('y') / height)
        ymaxs.append((attributes.get('y') + attributes.get('height')) / height)
        classes_text.append(label_dic.get('name').encode('utf8'))
        classes.append(class_text_to_int(label_dic.get('name')))

    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def main(_):

    out_put_path = "/home/viraj-uk/Pictures/th_3_british/eval/th_3_british_eval.tfrecord"
    image_dir = "/home/viraj-uk/Pictures/th_3_british/eval"
    csv_input = "/home/viraj-uk/Pictures/th_3_british/eval/th_3_british_eval.csv"

    writer = tf.python_io.TFRecordWriter(out_put_path)
    path = os.path.join(image_dir)
    examples = pd.read_csv(csv_input)

    # grouped = split(examples, 'new_filename')
    grouped = split(examples, 'filename')

    for group in grouped:

        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), out_put_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()