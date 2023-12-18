import torchvision


def get_encoder(name, pretrained=False):
    encoder = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in encoder.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return encoder[name]
