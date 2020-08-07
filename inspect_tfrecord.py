import tensorflow as tf

i = 0
for example in tf.python_io.tf_record_iterator("/home/viraj-uk/Desktop/train/ssd-tf-record-00001.tfrecord"):
    print(tf.train.Example.FromString(example))
    # print(example)
    # i = i + 1
    # if i >1:
    #     break