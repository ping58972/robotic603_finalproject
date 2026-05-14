from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_model(
    num_classes: int,
    dropout: float = 0.20,
    pretrained: bool = False,
    freeze_features: bool = False,
) -> nn.Module:
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model

