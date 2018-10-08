#!/usr/bin/env python

import tensorflow as tf

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2

def demo():
  filename_queue = tf.train.string_input_producer(
        ["../../datasets/iris_training.csv",
        "../../datasets/iris_test.csv"])

  reader = tf.TextLineReader(skip_header_lines=1)
  key, value = reader.read(filename_queue)

  # Default values, in case of empty columns. Also specifies the type of the
  # decoded result.
  record_defaults = [[1.0], [1.0], [1.0], [1.0], [1]]
  col1, col2, col3, col4, col5 = tf.decode_csv(
      value, record_defaults=record_defaults)
  features = tf.stack([col1, col2, col3, col4])

  with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(100):
      # Retrieve a single instance:
      example, label = sess.run([features, col5])
      print(i, example, label)

    coord.request_stop()
    coord.join(threads)

# tiny-imagenet
def demo1():
  filename_queue = tf.train.string_input_producer(
        ["../../datasets/tiny-imagenet-200/train_list.txt",
        "../../datasets/tiny-imagenet-200/val_list.txt"])

  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)

  # Default values, in case of empty columns. Also specifies the type of the
  # decoded result.
  record_defaults = [[""], [1], [1], [1], [1], [1]]
  col1, col2, col3, col4, col5, col6 = tf.decode_csv(
      value, record_defaults=record_defaults, field_delim=' ')
  features = tf.stack([col3, col4, col4, col6])

  with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(300):
      # Retrieve a single instance:
      file_path, class_, bbox = sess.run([col1, col2, features])
      print(file_path, class_, bbox)

    coord.request_stop()
    coord.join(threads)

# tiny-imagenet
def demo2():
  def _parse_function(each_element):
    # each_element = tf.cast(each_element, tf.string)
    each_element = tf.strings.strip(each_element)
    fields = tf.strings.split([each_element], ' ').values
    filename = fields[0]
    label = tf.stack([fields[1], fields[2], fields[3], fields[4], fields[5]])

    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [64, 64])
    image_scaled = image_resized / 255.0
    return image_scaled, label
    
  # image_filename = '../../datasets/tiny-imagenet-200/train_list.txt'
  # image_data = tf.gfile.FastGFile(image_filename, 'r').readlines()
  # for line in image_data:
  #   fields = line.strip().split(' ')

  filenames = ["../../datasets/tiny-imagenet-200/train_list.txt",
               "../../datasets/tiny-imagenet-200/val_list.txt"]
  dataset = tf.data.TextLineDataset(filenames)
  dataset = dataset.map(_parse_function)

  batch_size = tf.placeholder(shape=[], dtype=tf.int64)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_initializable_iterator()
  next_element = iterator.get_next()

  BATCH_SIZE = 32
  sess = tf.Session()
  sess.run(iterator.initializer, feed_dict={batch_size: BATCH_SIZE})

  for _ in range(1):
    image_batch, label_batch = sess.run(next_element)
    print(image_batch.shape)
    print(label_batch)

    for idx in range(BATCH_SIZE):
      image = image_batch[idx]
      clas = int(label_batch[idx][0])
      x1 = int(label_batch[idx][1])
      y1 = int(label_batch[idx][2])
      width = int(label_batch[idx][3])
      height = int(label_batch[idx][4])
      image = cv2.rectangle(image, (x1, y1), (x1 + width, y1 + height),\
                            (255, 0, 0), 2)

      cv2.namedWindow("w")
      cv2.imshow("w", image)
      cv2.waitKey(1000)
    plt.show()

    # for idx in range(BATCH_SIZE):
    #   image = image_batch[idx]
    #   clas = int(label_batch[idx][0])
    #   x1 = float(label_batch[idx][1])
    #   y1 = float(label_batch[idx][2])
    #   width = float(label_batch[idx][3])
    #   height = float(label_batch[idx][4])

    #   rect = patches.Rectangle((x1, y1), width, height)

    #   fig, axis = plt.subplots()
    #   axis.imshow(image_batch[idx])
    #   axis.add_patch(rect)
    #   plt.show()

    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #   plt.subplot(5,5,i+1)
    #   plt.xticks([])
    #   plt.yticks([])
    #   plt.grid(False)
    #   plt.imshow(images[i], cmap=plt.cm.binary)
    #   plt.xlabel(class_names[labels[i]])
    # plt.show()
    

def main():
  # demo()
  # demo1()
  demo2()

if __name__ == "__main__":
  main()