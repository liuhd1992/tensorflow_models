#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

from sklearn.utils import shuffle

DATA_DIR = '../../datasets/fashion-mnist'

def demo():
  def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

  def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
  
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

  # Load training and eval data
  mnist = input_data.read_data_sets(DATA_DIR, one_hot=False, validation_size=0)
  train_images = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  train_images, train_labels = shuffle(train_images, train_labels)
  train_images = train_images.reshape((train_images.shape[0], 28, 28))
  

  eval_images = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  eval_images, eval_labels = shuffle(eval_images, eval_labels)
  eval_images = eval_images.reshape((eval_images.shape[0], 28, 28))

  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


  # plt.figure()
  # plt.imshow(train_images[0])
  # plt.colorbar()
  # plt.grid(False)
  # plt.show()

  # plt.figure(figsize=(10,10))
  # for i in range(25):
  #   plt.subplot(5,5,i+1)
  #   plt.xticks([])
  #   plt.yticks([])
  #   plt.grid(False)
  #   plt.imshow(train_images[i], cmap=plt.cm.binary)
  #   plt.xlabel(class_names[train_labels[i]])
  # plt.show()

  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])

  model.compile(optimizer=tf.train.AdamOptimizer(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(train_images, train_labels, epochs=1)

  test_loss, test_acc = model.evaluate(eval_images, eval_labels)

  print('Test accuracy:', test_acc)

  predictions = model.predict(eval_images)
  print(predictions[0])

  print(class_names[np.argmax(predictions[0])])

  # i = 0
  # plt.figure(figsize=(6,3))
  # plt.subplot(1,2,1)
  # plot_image(i, predictions, eval_labels, eval_images)
  # plt.subplot(1,2,2)
  # plot_value_array(i, predictions,  eval_labels)

  # i = 12
  # plt.figure(figsize=(6,3))
  # plt.subplot(1,2,1)
  # plot_image(i, predictions, eval_labels, eval_images)
  # plt.subplot(1,2,2)
  # plot_value_array(i, predictions,  eval_labels)

  # Plot the first X test images, their predicted label, and the true label
  # Color correct predictions in blue, incorrect predictions in red
  num_rows = 5
  num_cols = 3
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, eval_labels, eval_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, eval_labels)
  plt.show()

def main():
  demo()

if __name__ == "__main__":
  main()