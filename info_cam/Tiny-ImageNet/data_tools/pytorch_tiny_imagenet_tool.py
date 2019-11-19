import os
from shutil import copy2


def get_immediate_subdirs(a_dir):
    return [(a_dir + os.sep + name) for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


train_dir_path = '.'
tgt_img_dir = '.'


def get_class_dict():
    class_dict = {}
    immediate_subdirs = get_immediate_subdirs(train_dir_path)
    class_idx = 0
    for a_sub_dir in immediate_subdirs:
        split_path = os.path.split(a_sub_dir)[-1]
        class_dict[split_path] = class_idx
    class_idx += 1
    return class_dict


class_dict = get_class_dict()

var_dir_path = '../tiny-imagenet-200/val'
var_img_dir = os.path.join(var_dir_path, 'images')
print('var img dir: ', var_img_dir)

class_idx = 0
a_f = open(os.path.join(var_dir_path, 'val_annotations.txt'))
for a_line in a_f:
    split_line = a_line.split('\t')
    a_val_img_f = os.path.join(var_dir_path, 'images', split_line[0])
    print('joint line: ', a_val_img_f)
    a_tgt_dir = os.path.join(tgt_img_dir, split_line[1])
    print('tgt img dir: ', a_tgt_dir)
    copy2(a_val_img_f, a_tgt_dir)
