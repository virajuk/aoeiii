import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.contrib.slim import parallel_reader

import sys
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import input_reader_pb2
from util.common_utils import image_with_labels
from util.dataset_info import parts_classes_v2, damage_classes_v1, mturk_forward_dict, parts_classes_v1

# Parameters
BASE_PATH = "/home/viraj-uk/Pictures/guanaco_2/train/guanaco_2_train.tfrecord"
# BASE_PATH = "/home/viraj-uk/Pictures/bighorn_sheep_black/bighorn_sheep_black_eval.tfrecord"
# DATA_PATH = "{}/processed/orig5_parts/*/parts/*_parts-000??-of-00010.tfrecord".format(BASE_PATH)
DATA_PATH = BASE_PATH

AUTO = False
# LABEL_DICT = damage_classes_v1
LABEL_DICT = parts_classes_v1
# LABEL_DICT = mturk_forward_dict

def get_observation():
    with tf.Session() as sess:
        _, string_tensor = parallel_reader.parallel_read([DATA_PATH], reader_class=tf.TFRecordReader,
                                                         num_epochs=1, num_readers=1, shuffle=True,
                                                         dtypes=[tf.string, tf.string], capacity=500,
                                                         min_after_dequeue=200)

        decoder = tf_example_decoder.TfExampleDecoder(load_instance_masks=True,
                                                      instance_mask_type=input_reader_pb2.PNG_MASKS)
        decoded_data = decoder.decode(string_tensor)

        image = decoded_data['image']
        box_list = decoded_data['groundtruth_boxes']
        class_list = decoded_data['groundtruth_classes']
        mask_list = decoded_data['groundtruth_instance_masks']
        file_name = decoded_data['filename']

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while True:
            img, boxes, classes, masks, file_names = sess.run([image, box_list, class_list, mask_list, file_name])

            print("current file {} contains {} masks".format(file_names.decode(), len(boxes)))
            sys.stdout.flush()

            # if masks.shape[0] > 5:
            #     return masks

            image_with_labels(img, boxes, masks, classes, label_dict=LABEL_DICT)
            # image_with_boxes(img, boxes, masks, classes, label_dict=LABEL_DICT)
            plt.tight_layout()

            # if AUTO:
            #     plt.pause(0.5)
            # elif plt.waitforbuttonpress():
            #     break

            plt.waitforbuttonpress()
            plt.close()

        coord.request_stop()
        coord.join(threads)

        return 1


if __name__ == '__main__':
    a = get_observation()
