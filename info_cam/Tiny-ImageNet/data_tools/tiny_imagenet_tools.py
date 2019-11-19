import os
import glob

dir_path = '../tiny-imagenet-200/train'


def get_immediate_subdirs(a_dir):
    return [(a_dir + os.sep + name) for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

text_file = open("tmp.txt", "w")

class_dict = {}

class_idx = 0
immediate_subdirs = get_immediate_subdirs(dir_path)
for a_sub_dir in immediate_subdirs:
    split_path = os.path.split(a_sub_dir)[-1]
    class_dict[split_path] = class_idx
    for file in os.listdir(a_sub_dir):
        if file.endswith('.txt'):
            print(os.path.join(a_sub_dir, file))
            a_f = open(os.path.join(a_sub_dir, file))
            for a_line in a_f:
                split_line = a_line.split('\t')
                image_id = os.path.join(split_path, split_line[0])
                split_line[0] = image_id
                split_line.insert(1, str(class_idx))
                print('split line: ', split_line)

                print('split line: ', '\t'.join(split_line))
                text_file.write('\t'.join(split_line))

    box_file = os.path.join(a_sub_dir, )
    class_idx += 1

text_file.close()


val_txt_file = open("test.txt", "w")
var_dir_path = '../tiny-imagenet-200/val'

class_idx = 0
a_f = open(os.path.join(var_dir_path, 'val_annotations.txt'))
for a_line in a_f:
    split_line = a_line.split('\t')
    split_line[1] = str(class_dict[split_line[1]])
    joint_line = '\t'.join(split_line)
    val_txt_file.write(joint_line)

val_txt_file.close()
