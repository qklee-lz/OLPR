import timm
import torchvision.models as tv_models
import torch.nn as nn

def build_model(name: str, num_classes: int = 3, pretrained: bool = True):
    """
    Args:
        name (str):  (e.g., 'resnet50', 'swinv2_base_window8_256.ms_in1k', etc.)
        num_classes (int) 
        pretrained (bool)
    """
    if hasattr(tv_models, name):
        model_fn = getattr(tv_models, name)
        model = model_fn(weights="IMAGENET1K_V1" if pretrained else None)
        if hasattr(model, "fc"):  
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "classifier"): 
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
    else:  
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    return model
