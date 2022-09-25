from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .resnet import resnet32x4, resnet8x4
from .resnetv2 import ResNet50
from .mobilenetv2 import mobile_half
from .official_resnet import RESNET18, RESNET34, RESNET50
from .MobileNet import MobileNet
from .official_densenet import densenet201


model_dict = {
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,    
    'resnet32x4' : resnet32x4, 
    'resnet8x4' : resnet8x4,
    'ResNet50': ResNet50,
    'MobileNetV2': mobile_half,
    'RESNET18': RESNET18,
    'RESNET34': RESNET34,
    'RESNET50': RESNET50,
    'MobileNet': MobileNet,
    'densenet201': densenet201,
}
