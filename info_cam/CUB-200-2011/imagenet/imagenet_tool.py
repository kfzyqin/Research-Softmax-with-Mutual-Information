import pandas as pd
import os
from PIL import Image

imgnet_tra_data_path = '/media/zhenyue-qin/local/Data/imagenet_object_localization/ILSVRC/Data/CLS-LOC/train'
imgnet_val_data_path = '/media/zhenyue-qin/local/Data/imagenet_object_localization/ILSVRC/Data/CLS-LOC/val'

loc_tra_sol_path = '/media/zhenyue-qin/local/Data/LOC_train_solution.csv'
loc_val_sol_path = '/media/zhenyue-qin/local/Data/LOC_val_solution.csv'

loc_tra_sol = pd.read_csv(loc_tra_sol_path)
loc_val_sol = pd.read_csv(loc_val_sol_path)

class_dict = {}
class_idx = 0

inst_idx = 1

# box_file = open('bounding_boxes_2.txt', 'w')
# size_file = open('sizes_2.txt', 'w')
# train_file = open('train_2.txt', 'w')
# val_file = open('val_2.txt', 'w')
# test_file = open('test_2.txt', 'w')

# for i in range(len(loc_tra_sol)):
#     split_ = loc_tra_sol.iloc[i][1].split()
#     a_class_label = split_[0]
#     an_img_path = os.path.join(a_class_label, loc_tra_sol.iloc[i][0])
#     if a_class_label not in class_dict.keys():
#         class_dict[a_class_label] = class_idx
#         class_idx += 1

print('len(loc_tra_sol): ', len(loc_tra_sol))

orig_box_f = open('bounding_boxes_.txt', 'r')
orig_size_f = open('sizes_.txt')
for a_line in orig_box_f:
    box_file.write(a_line)
    if inst_idx > len(loc_tra_sol):
        break
for a_line in orig_size_f:
    size_file.write(a_line)
    if inst_idx > len(loc_tra_sol):
        break

inst_idx = len(loc_tra_sol) + 1

        # for i in range(len(loc_tra_sol)):
#     split_ = loc_tra_sol.iloc[i][1].split()
#     a_class_label = split_[0]
#     an_img_path = os.path.join(a_class_label, loc_tra_sol.iloc[i][0])
#     # print('{} {}'.format(inst_idx, ' '.join(split_[1:5])))
#     box_file.write( '{} {}\n'.format(inst_idx, ' '.join(split_[1:5])) )
#     train_file.write('{} {} {}\n'.format(inst_idx, an_img_path, class_dict[a_class_label]))
#     a_full_img_path = os.path.join(imgnet_tra_data_path, an_img_path) + '.JPEG'
#     im = Image.open(a_full_img_path)
#     width, height = im.size
#     size_file.write('{} {} {}\n'.format(inst_idx, width, height))
#
#     inst_idx += 1

for i in range(len(loc_val_sol)):
    split_ = loc_val_sol.iloc[i][1].split()
    a_class_label = split_[0]
    an_img_path = loc_val_sol.iloc[i][0]
    # print('{} {}'.format(inst_idx, ' '.join(split_[1:5])))
    box_file.write('{} {}\n'.format(inst_idx, ' '.join(split_[1:5])))
    if i % 2 == 0:
        val_file.write('{} {}.JPEG {}\n'.format(inst_idx, an_img_path, class_dict[a_class_label]))
    else:
        test_file.write('{} {}.JPEG {}\n'.format(inst_idx, an_img_path, class_dict[a_class_label]))

    a_full_img_path = os.path.join(imgnet_val_data_path, an_img_path) + '.JPEG'
    im = Image.open(a_full_img_path)
    width, height = im.size
    size_file.write('{} {} {}\n'.format(inst_idx, width, height))

    inst_idx += 1

box_file.close()
size_file.close()
train_file.close()
val_file.close()
test_file.close()
