#!/usr/bin/env python

import os
from collections import defaultdict

def gen_tiny_imagenet_list(path):
  train_img_list_path = os.path.join(path, 'train_list.txt')
  eval_img_list_path = os.path.join(path, 'val_list.txt')

  class_map_index_path = os.path.join(path, 'class_map_string.txt')

  train_img_dir = os.path.join(path, 'train')
  eval_img_dir = os.path.join(path, 'val')

  # train data
  with open(train_img_list_path, 'w') as fw,\
       open(class_map_index_path, 'w') as fw1:
    train_file_list = os.listdir(train_img_dir)
    print('total num of class file: ', len(train_file_list))
    total_imgs = 0
    for index, class_name in enumerate(train_file_list):
      fw1.write('{} {}\n'.format(class_name, index))
      class_imgs_num = 0
      class_path = os.path.join(train_img_dir, class_name)
      with open(os.path.join(class_path, class_name + '_boxes.txt')) as fr:
        for line in fr:
          class_imgs_num += 1
          total_imgs += 1
          fields = line.strip().split()
          full_path = os.path.join(class_path, 'images', fields[0])
          labels = '{} {} {} {} {} {}\n'.format(full_path, index, 
                fields[1], fields[2], fields[3], fields[4])
          fw.write(labels)
      print('num of class {} is {}'.format(class_name, class_imgs_num))
    print('total images: {}'.format(total_imgs))
  
  # read maping
  class_map_index_dict = defaultdict()
  with open(class_map_index_path, 'r') as fopen:
    for line in fopen:
      map_fields = line.strip().split(' ')
      print(map_fields)
      class_map_index_dict[map_fields[0]] = map_fields[1]
  print(class_map_index_dict)

  # test data
  with open(eval_img_list_path, 'w') as fw:
    val_file_list = os.listdir(eval_img_dir)
    print('total num of class file: ', len(val_file_list))
    total_imgs = 0
    with open(os.path.join(eval_img_dir, 'val_annotations.txt')) as fr:
      for line in fr:
        total_imgs += 1
        fields = line.strip().split()
        full_path = os.path.join(eval_img_dir, 'images', fields[0])
        labels = '{} {} {} {} {} {}\n'.format(full_path, 
                  class_map_index_dict[fields[1]], 
                  fields[2], fields[3], fields[4], fields[5])
        fw.write(labels)
    print('num of class {} is {}'.format(class_name, class_imgs_num))
  print('total images: {}'.format(total_imgs))
  
  
def demo():
  path_dir = '/Users/liuhaidong01/codes/tensorflow/pyproject/datasets/tiny-imagenet-200'
  gen_tiny_imagenet_list(path_dir)

def main():
  demo()

if __name__ == "__main__":
  main()