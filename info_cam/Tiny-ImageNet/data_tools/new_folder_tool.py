import os


def get_immediate_subdirs(a_dir):
    return [(a_dir + os.sep + name) for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


dir_root = '../tiny-imagenet/images'
orig_dir_root = '../tiny-imagenet-200'


def get_bounding_box_list():
    class_box = {}
    train_path = os.path.join(orig_dir_root, 'train')
    val_path = os.path.join(orig_dir_root, 'val')
    train_sub_dirs = get_immediate_subdirs(train_path)

    for a_train_sub_dir in train_sub_dirs:
        for file in os.listdir(a_train_sub_dir):
            if file.endswith('.txt'):
                a_f = open(os.path.join(a_train_sub_dir, file))
                for a_line in a_f:
                    split_line = a_line.split('\t')
                    class_box[split_line[0]] = split_line[1:]

    val_txt = open(os.path.join(val_path, 'val_annotations.txt'))
    for a_line in val_txt:
        split_line = a_line.split('\t')
        class_box[split_line[0]] = split_line[2:]
    return class_box


class_box = get_bounding_box_list()


img_idx = 1
class_idx = 0
# train_txt = open("train.txt", "w")
# test_txt = open("test.txt", "w")
# bbox_txt = open('bounding_boxes.txt', 'w')

size_txt = open('sizes.txt', 'w')

sub_dirs = get_immediate_subdirs(dir_root)
for a_dir in sub_dirs:
    class_id = a_dir.replace(dir_root, '')[1:]
    img_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(a_dir) for f in filenames if
                 os.path.splitext(f)[1] == '.JPEG']
    for an_img_file in img_files:
        an_img_entry = an_img_file.replace(dir_root, '')[1:]
        id_entry = '{} {}\n'.format(str(img_idx), an_img_entry)
        a_bbox = class_box[an_img_entry.replace(class_id + os.sep, '')]
        bbox_line = '{} {}'.format(str(img_idx), ' '.join(a_bbox))
        a_new_line = '{} {} {}\n'.format(img_idx, an_img_entry, class_idx)
        size_line = '{} {} {}\n'.format(img_idx, 64, 64)
        size_txt.write(size_line)
        # if 'val' in an_img_entry:
        #     test_txt.write(a_new_line)
        # else:
        #     train_txt.write(a_new_line)

        # bbox_txt.write(bbox_line)

        img_idx += 1
    class_idx += 1
