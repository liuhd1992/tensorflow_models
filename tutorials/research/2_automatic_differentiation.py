#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np
from math import pi
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

tf.enable_eager_execution()
tfe = tf.contrib.eager

def demo():
  def f(x):
    return tf.square(tf.sin(x))

  def grad(f):
    return lambda x: tfe.gradients_function(f)(x)[0]

  grad_f = tfe.gradients_function(f)

  print grad_f(pi/2)[0].numpy()
  x = tf.lin_space(-2*pi, 2*pi, 100)

  plt.plot(x, f(x), label='f')
  plt.plot(x, grad(f)(x), label='grad f')
  plt.plot(x, grad(grad(f))(x), label='grad grad f')
  plt.show()

def demo1():
  def f(x, y):
    output = 1
    for i in range(int(y)):
        output = tf.multiply(output, x)
    return output

  def g(x, y):
    return tfe.gradients_function(f)(x, y)

  print(f(10.0, 3.0))
  print(g(10.0, 3.0))

def demo2():
  x = tf.convert_to_tensor(np.array([[1.0, 2.0],[3.0, 2.0]]))
  with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = tf.reduce_min(x)
    z = tf.multiply(y, y)

  dz_dy = t.gradient(z, y)
  print dz_dy.numpy()

  dz_dx = t.gradient(z, x)
  for i in [0, 1]:
    for j in [0, 1]:
      print dz_dx[i][j]

def demo3():
  x = tf.constant(1.0)
  with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
      t2.watch(x)
      y = x * x *x
    dy_dx = t2.gradient(y, x)
  d2y_dx2 = t.gradient(dy_dx, x)
  print dy_dx.numpy()
  print d2y_dx2.numpy()

def main():
  #demo()
  #demo1()
  #demo2()
  demo3()

if __name__ == "__main__":
    main()
