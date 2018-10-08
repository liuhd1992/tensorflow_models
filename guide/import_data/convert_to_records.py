"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import mnist

import input_data

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      image_raw = images[index].tostring()
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'height': _int64_feature(rows),
                  'width': _int64_feature(cols),
                  'depth': _int64_feature(depth),
                  'label': _int64_feature(int(labels[index])),
                  'image_raw': _bytes_feature(image_raw)
              }))
      writer.write(example.SerializeToString())

def convert_tf_dataset_to(sess, data_set, name):
  """Converts a dataset to tfrecords."""
  iterator = data_set.make_one_shot_iterator()
  next_element = iterator.get_next()

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  with tf.python_io.TFRecordWriter(filename) as writer:
    for _ in range(FLAGS.num_examples):
      image, label = sess.run(next_element)
      image = image.reshape((FLAGS.height, FLAGS.width))
      image_raw = image.tostring()

      rows = image.shape[0]
      cols = image.shape[1]

      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'height': _int64_feature(rows),
                  'width': _int64_feature(cols),
                  'depth': _int64_feature(1),
                  'label': _int64_feature(int(label)),
                  'image_raw': _bytes_feature(image_raw)
              }))
      writer.write(example.SerializeToString())

def demo():
  # Get the data.
  data_sets = mnist.read_data_sets(FLAGS.directory,
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=FLAGS.validation_size)

  # Convert to Examples and write the result to TFRecords.
  convert_to(data_sets.train, 'train')
  convert_to(data_sets.validation, 'validation')
  convert_to(data_sets.test, 'test')

def demo1():
  dataset = input_data.train(FLAGS.directory)

  sess = tf.Session()
  convert_tf_dataset_to(sess, dataset, 'train')

# fashion-mnist
def demo2():
  def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape((28 * 28))

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


  def augment(image, label):
    """Placeholder for data augmentation."""
    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.
    return image, label


  def normalize(image, label):
    """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.cast(image, tf.float32) - 0.5
    return image, label


  filename = os.path.join(FLAGS.directory, 'train.tfrecords')
  dataset = tf.data.TFRecordDataset(filename)


  # map takes a python function and applies it to every sample
  dataset = dataset.map(decode)
  dataset = dataset.map(augment)
  dataset = dataset.map(normalize)

  num_epochs = 1
  batch_size = 32
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  batch_data = iterator.get_next()

  sess = tf.Session()

  for _ in range(num_epochs):
    image_batch, label_batch = sess.run(batch_data)

    print(image_batch.shape)
    print(label_batch.shape)

    

def main(unused_argv):
  # demo()
  # demo1()
  demo2()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='../../datasets/fashion-mnist',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--num_examples',
      type=int,
      default=60000,
      help='number of training data'
  )
  parser.add_argument(
      '--height',
      type=int,
      default=28,
      help='image height'
  )
  parser.add_argument(
      '--width',
      type=int,
      default=28,
      help='image width'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the training data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
