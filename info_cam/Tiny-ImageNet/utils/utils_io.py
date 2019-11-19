import datetime
import os
import fnmatch
import shutil
import json


def normalize_data(data_type, channels, sequence):
    if data_type == 'multi_mnist':
        if channels > 1:
            sequence = sequence.permute([0, 3, 1, 2])
        else:
            sequence = sequence.permute([0, 3, 1, 2])
    else:
        raise NotImplementedError()

    return sequence


def get_training_batch(data_loader, data_type, channels):
    while True:
        for sequence, labels in data_loader:
            print('sequence: ', sequence.shape)
            batch = normalize_data(data_type, channels, sequence)
            print('batch: ', batch.shape)
            yield batch, labels


def get_current_time():
    currentDT = datetime.datetime.now()
    return str(currentDT.strftime("%Y-%m-%dT%H-%M-%S"))


def copy_python_files(out_dir, src_dir=None):
    if src_dir is None:
        this_file = os.path.realpath(__file__)
        src_dir = os.path.dirname(this_file)
    out_dir = os.path.join(out_dir, 'py_files')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    py_files = [os.path.join(dirpath, f)
                   for dirpath, dirnames, files in os.walk(src_dir)
                   for f in fnmatch.filter(files, '*.py')]
    # py_files = glob.glob(os.path.join(src_dir, "*.py"), recursive=True)

    for py_file in py_files:
        shutil.copy(py_file, out_dir)
        split_path = os.path.split(py_file)
        a_py_file = py_file.replace(src_dir, '')
        new_dst_file_name = os.path.join(out_dir, split_path[-1])
        os.rename(new_dst_file_name, new_dst_file_name + '.txt')


def load_a_dict(file_name):
    with open(file_name, 'r') as fp:
        data = json.load(fp)
        return data


def save_a_dict(file_name, data):
    with open(file_name, 'w') as fp:
        json.dump(data, fp, indent=4)
