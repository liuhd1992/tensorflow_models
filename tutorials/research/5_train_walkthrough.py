#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import, division, print_function

import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

def demo():
  # train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
  
  # train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
  #                                          origin=train_dataset_url)
  # print("Local copy of the dataset file: {}".format(train_dataset_fp))

  train_dataset_fp = "../../datasets/iris_training.csv"

  # column order in CSV file
  column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

  feature_names = column_names[:-1]
  label_name = column_names[-1]

  print("Features: {}".format(feature_names))
  print("Label: {}".format(label_name))

  class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

  batch_size = 32
  train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
  
  # plt.scatter(features['petal_length'],
  #           features['sepal_length'],
  #           c=labels,
  #           cmap='viridis')

  # plt.xlabel("Petal length")
  # plt.ylabel("Sepal length")
  # plt.show()

  def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

  def loss(models, x, y):
    predicts = models(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=predicts)

  def grad(model, inputs, targets):
    with tf.GradientTape() as t:
      current_loss = loss(model, inputs, targets)
    return current_loss, t.gradient(current_loss, model.trainable_variables)
  
  train_dataset = train_dataset.map(pack_features_vector)

  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4, )),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
  ])
  
  learning_rate = 0.01

  # for i in range(2000):
  #   features, labels = next(iter(train_dataset))
  #   curr_loss, dW = grad(model, features, labels)
  #   print("curr loss: {}".format(curr_loss))

  #   # update the model
  #   for index in range(len(model.trainable_variables)):
  #     model.trainable_variables[index].assign_sub(learning_rate * dW[index])
    
  features, labels = next(iter(train_dataset))
  
  optimizer = tf.train.GradientDescentOptimizer(0.01)

  global_step = tf.train.get_or_create_global_step()

  loss_value, grads = grad(model, features, labels)


  print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                          loss_value.numpy()))

  # keep results for plotting
  train_loss_results = []
  train_accuracy_results = []

  num_epochs = 200

  for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    for x, y in train_dataset:
      loss_value, grads = grad(model, x, y)
      optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                global_step=global_step)
      
      # Track progress
      epoch_loss_avg(loss_value)  # add current batch loss
      # compare predicted label to actual label
      epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
    
    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
      print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

  # fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
  # fig.suptitle('Training Metrics')

  # axes[0].set_ylabel("Loss", fontsize=14)
  # axes[0].plot(train_loss_results)

  # axes[1].set_ylabel("Accuracy", fontsize=14)
  # axes[1].set_xlabel("Epoch", fontsize=14)
  # axes[1].plot(train_accuracy_results);

  # plt.show()

  # evaluation
  # test_url = "http://download.tensorflow.org/data/iris_test.csv"

  # test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
  #                                 origin=test_url)

  test_dataset_fp = "../../datasets/iris_test.csv"
  test_dataset = tf.contrib.data.make_csv_dataset(
    test_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1,
    shuffle=False)

  test_dataset = test_dataset.map(pack_features_vector)

  test_accuracy = tfe.metrics.Accuracy()

  for x, y in test_dataset:
    logits = model(x)
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(predictions, y)

  # print(tf.stack([y, predictions], axis=1))

  print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

  predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
  ])

  predictions = model(predict_dataset)
  for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))

def main():
  demo()

if __name__ == "__main__":
  main()