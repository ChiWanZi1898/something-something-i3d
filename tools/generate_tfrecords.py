from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time

import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('num_per_file', 256, 'The maximum number of entries in one tfrecords.')
tf.flags.DEFINE_string('output_dir', '../data/selected-10', 'The directory to output tfrecords')

LABEL_FILE = '../data/something-something-v2-labels.json'
TRAIN_DATA_FILE = '../data/something-something-v2-train.json'
VALID_DATA_FILE = '../data/something-something-v2-validation.json'
SELECTED_CLASSES_FILE = '../data/selected-10-classes.json'
SELECTED_SUBCLASSES_FILE = '../data/selected-41-subclasses.json'
MAPPING_FILE = '../data/selected-10-map.json'


def timer(func):
    def time_it(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        tf.logging.info('{}[{:.4f}ms]'.format(func.__name__, (t1 - t0) * 1000.))
        return result

    return time_it


@timer
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        obj = json.load(f, encoding='utf-8')
    return obj


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_example(id, class_index, subclass_index):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'id': _int64_feature(id),
                'class_index': _int64_feature(class_index),
                'subclass_index': _int64_feature(subclass_index)
            }
        )
    )
    return example


def generate_tfrecords(set_type):
    if set_type == 'train':
        data_file = TRAIN_DATA_FILE
    else:
        data_file = VALID_DATA_FILE
    entries = load_json(data_file)
    mapping = load_json(MAPPING_FILE)
    classes = load_json(SELECTED_CLASSES_FILE)
    subclasses = load_json(SELECTED_SUBCLASSES_FILE)

    output_dir = os.path.join(FLAGS.output_dir, set_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    selected_entries = []
    for entry in entries:
        label = entry['template']
        if label in subclasses.keys():
            id = int(entry['id'])
            class_index = int(classes[mapping[label]])
            subclass_index = int(subclasses[label])

            selected_entries.append((id, class_index, subclass_index))

    selected_entry_num = len(selected_entries)

    for tfrecords_index in range((selected_entry_num - 1) // FLAGS.num_per_file + 1):
        tfrecords_path =  os.path.join(output_dir, '{}_{:03d}.tfrecords'.format(set_type, tfrecords_index))
        writer = tf.python_io.TFRecordWriter(tfrecords_path)
        for id, class_index, subclass_index in selected_entries[tfrecords_index * FLAGS.num_per_file:(tfrecords_index + 1) * FLAGS.num_per_file]:
            example = get_example(id, class_index, subclass_index)
            writer.write(example.SerializeToString())
        tf.logging.info('Save to {}'.format(tfrecords_path))
        writer.close()



def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    # generate_tfrecords('train')
    generate_tfrecords('valid')

if __name__ == '__main__':
    main()
