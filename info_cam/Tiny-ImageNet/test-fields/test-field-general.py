import pretrainedmodels
from pretrainedmodels import se_resnet50

tmp = se_resnet50(num_classes=1000, pretrained='imagenet')
print('tmp: ', type(tmp))