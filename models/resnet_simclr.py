import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, use_projection_head=True):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(weights=None),
            "resnet50": models.resnet50(weights=None),
        }

        self.backbone = self._get_basemodel(base_model)
        self.feature_dim = self.backbone.fc.in_features
        self.use_projection_head = use_projection_head

        self.backbone.fc = nn.Identity()
        if self.use_projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.ReLU(),
                nn.Linear(self.feature_dim, out_dim),
            )
            self.projection_dim = out_dim
        else:
            self.projection_head = nn.Identity()
            self.projection_dim = self.feature_dim

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def encode(self, x):
        return self.backbone(x)

    def project(self, features):
        return self.projection_head(features)

    def forward(self, x, return_embedding=False):
        embeddings = self.encode(x)
        projections = self.project(embeddings)
        if return_embedding:
            return embeddings, projections
        return projections
