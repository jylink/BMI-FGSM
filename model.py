from torchvision import models

_models = ["inceptionv3", "vgg16", "resnet101", "resnet152"]

def load_net(model='inceptionv3', pretrained=True):
    assert model in _models
    
    if model == 'inceptionv3':
        return models.inception_v3(pretrained=pretrained)
    elif model == 'vgg16':
        return models.vgg16(pretrained=pretrained)
    elif model == 'resnet101':
        return models.resnet101(pretrained=pretrained)
    elif model == 'resnet152':
        return models.resnet152(pretrained=pretrained)