#!/use/bin/env python

import tensorflow as tf

def demo():
  dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10, 10]))
  print(dataset1.output_types)  # ==> "tf.float32"
  print(dataset1.output_shapes)  # ==> "(10,)"

  dataset2 = tf.data.Dataset.from_tensor_slices(
      (tf.random_uniform([4, 10, 10]),
       tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
  print(dataset2.output_types)
  print(dataset2.output_shapes)

  dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
  print(dataset3.output_types)
  print(dataset3.output_shapes)

  dataset = tf.data.Dataset.from_tensor_slices(
    {"a": tf.random_uniform([4, 100]),
     "b": tf.random_uniform([4, 10, 10], maxval=100, dtype=tf.int32)}
  )
  print(dataset.output_classes)
  print(dataset.output_types)
  print(dataset.output_shapes)

def demo1():
  dataset = tf.data.Dataset.range(100)
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()

  sess = tf.Session()
  for i in range(100):
    print(sess.run(next_element))

def demo2():
  max_value = tf.placeholder(tf.int64, shape=[])
  dataset = tf.data.Dataset.range(max_value)
  iterator = dataset.make_initializable_iterator()
  next_element = iterator.get_next()

  sess = tf.Session()
  
  sess.run(iterator.initializer, feed_dict={max_value: 10})
  for i in range(10):
    print(sess.run(next_element))

  sess.run(iterator.initializer, feed_dict={max_value:100})
  for i in range(100):
    print(sess.run(next_element))

def demo3():
  # Define training and validation datasets with the same structure.
  training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, dtype=tf.int64))
  validation_dataset = tf.data.Dataset.range(50)

  iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                             training_dataset.output_shapes)

  next_element = iterator.get_next()

  training_init_op = iterator.make_initializer(training_dataset)
  validation_init_op = iterator.make_initializer(validation_dataset)

  sess = tf.Session()

  # Run 20 epochs in which the training dataset is traversed, followed by the
  # validation dataset.
  for _ in range(20):
    sess.run(training_init_op)
    for _ in range(100):
      print(sess.run(next_element))

    sess.run(validation_init_op)
    for _ in range(50):
      print(sess.run(next_element))


def main():
  # demo()
  # demo1()
  # demo2()
  demo3()

if __name__ == "__main__":
  main()