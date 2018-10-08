#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

tfe = tf.contrib.eager

tf.enable_eager_execution()

def demo():
  x = tf.Variable(1.0, dtype=tf.float32)
  tf.assign(x, 3)

  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)

  print(sess.run(x))

def demo_train():
  class Model(object):
    def __init__(self):
      self.W = tfe.Variable(5.0)
      self.b = tfe.Variable(0.0)
    
    def __call__(self, x):
      return self.W * x + self.b
  
  def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

  def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
      training_loss = loss(model(inputs), outputs)
    [dW, db] = t.gradient(training_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

  model = Model()

  TRUE_W = 3.0
  TRUE_b = 2.0
  NUM_EXAMPLES = 1000

  inputs = tf.random_normal(shape=[NUM_EXAMPLES])
  noise  = tf.random_normal(shape=[NUM_EXAMPLES])

  outputs = inputs * TRUE_W + TRUE_b + noise

#   plt.scatter(inputs, outputs, c='b')
#   plt.scatter(inputs, model(inputs), c='r')
#   plt.show()

  print('Current loss: '),
  print(loss(model(inputs), outputs).numpy())
  
  Ws, bs = [], []
  epochs = range(1000)
  for i in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)
    learning_rate = 0.01 * tf.math.exp(tf.convert_to_tensor(-1 * i, dtype=tf.float32))
    train(model, inputs, outputs, 0.01)

    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
          (i, Ws[-1], bs[-1], current_loss.numpy()))

  plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
  plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
  plt.legend(['W', 'b', 'true W', 'true_b'])
  plt.show()


def main():
#   demo()
  demo_train()

if __name__ == "__main__":
  main()