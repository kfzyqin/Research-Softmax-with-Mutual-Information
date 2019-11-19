import os


def read_test_file(file_path):
    dict_ = {}
    a_f = open(file_path)
    for a_line in a_f:
        split_line = a_line.split(' ')
        if split_line[-1] not in dict_.keys():
            dict_[split_line[-1]] = [a_line]
        else:
            dict_[split_line[-1]].append(a_line)

    val_ = open('val_.txt', 'w')
    test_ = open('test_.txt', 'w')
    for a_key in dict_.keys():
        target_ = dict_[a_key]
        for i in range(len(target_)):
            if i % 2 == 0:
                val_.write(target_[i])
            else:
                test_.write(target_[i])



if __name__ == '__main__':
    read_test_file('/home/zhenyue-qin/Research/Research-Sayaka/Sayaka-Folks/Folks-ADL/ADL/Pytorch/datalist/CUB/test.txt')
