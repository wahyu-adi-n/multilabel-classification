import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FashionMultilableModel(nn.Module):
    def __init__(self, n_color_classes, n_gender_classes, n_article_classes) -> None:
        super().__init__()
        self.base_model = models.mobilenet_v2().features
        last_channel = models.mobilenet_v2().last_channel
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.color = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_color_classes)
        )

        self.gender = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_gender_classes)
        )

        self.article = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_article_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        return dict(color=self.color(x),
                    gender=self.gender(x),
                    article=self.article(x)
                    )

    def get_loss(self, net_output, ground_truth):
        color_loss = F.cross_entropy(
            net_output['color'], ground_truth['color_labels'])
        gender_loss = F.cross_entropy(
            net_output['gender'], ground_truth['gender_labels'])
        article_loss = F.cross_entropy(
            net_output['article'], ground_truth['article_labels'])

        loss = color_loss + gender_loss + article_loss
        return loss, dict(color=color_loss,
                          gender=gender_loss,
                          article=article_loss
                          )
