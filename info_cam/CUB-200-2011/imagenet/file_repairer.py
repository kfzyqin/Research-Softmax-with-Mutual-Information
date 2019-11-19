import os

a_f = open('val_2.txt', 'r')
train_f = open('val.txt', 'w')
for a_line in a_f:
  split_line = a_line.split(' ')
  split_line[1] = os.path.join('val', split_line[1])
  print(split_line)
  train_f.write(' '.join(split_line))
