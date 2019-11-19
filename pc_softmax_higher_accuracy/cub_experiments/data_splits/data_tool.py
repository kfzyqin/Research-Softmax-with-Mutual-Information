f = open("cub_200/val.txt", "r")

class_dict = {}
rtn = open('val_tmp.txt', 'w')

for x in f:
    class_int = int(x[7:10])
    if class_int not in class_dict.keys():
        class_dict[class_int] = 1
    else:
        class_dict[class_int] += 1
    if class_dict[class_int] > 5:
        if class_int % 2 == 0:
            if class_dict[class_int] <= 15:
                rtn.write(x)
        else:
            pass
    else:
        rtn.write(x)



