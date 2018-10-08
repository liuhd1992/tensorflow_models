#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import timeit

def demo():
  print(tf.add(1, 1))
  print(tf.add([1, 2], [3, 4]))
  print(tf.square(2) + tf.square(3))

  x = tf.matmul([[2], [4]], [[2, 3]])
  print(x.shape)
  print(x.dtype)

  ndarray = np.ones((3, 3))
  tensor = tf.multiply(ndarray, 10)
  print(tensor)
  print(np.add(tensor, 1))
  print(tensor.numpy())

  x = tf.random_uniform((3, 3))
  print(tf.test.is_gpu_available())
  print(x.device)
  print(x)

  with tf.device("CPU:0"):
      x = tf.random_uniform([1000, 1000])
      print x.device

def demo_datasets():
  ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

  import tempfile
  _, filename = tempfile.mkstemp()

  with open(filename, "w") as f:
    f.write("""Line 1\nLine 2\nLine 3""")
    for x in ds_tensors:
      f.write("\n{}".format(x.numpy()))

  ds_tensors = ds_tensors.map(lambda x: tf.add(x, 10)).shuffle(6).batch(1)
  ds_file = tf.data.TextLineDataset(filename)
  ds_file = ds_file.batch(2)

  # for x in ds_tensors:
    # print x
    
  for x in ds_file:
    print x



def main():
  # demo()
  demo_datasets()
  

if __name__ == "__main__":
    main()
