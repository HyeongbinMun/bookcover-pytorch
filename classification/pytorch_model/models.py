import timm
import torch
from torch import nn

from utils import GeM, ArcMarginProduct


class ImageModel(nn.Module):
    def __init__(self, model_name, class_n, mode='train'):
        super().__init__()
        self.model_name = model_name.lower()
        self.class_n = class_n
        self.mode = mode
        self.encoder = timm.create_model(self.model_name, pretrained=False)

        names = []
        modules = []
        for name, module in self.encoder.named_modules():
            names.append(name)
            modules.append(module)

        self.fc_in_features = self.encoder.num_features
        print(f'The layer was modified...')

        fc_name = names[-1].split('.')

        if len(fc_name) == 1:
            print(
                f'{getattr(self.encoder, fc_name[0])} -> Linear(in_features={self.fc_in_features}, out_features={class_n}, bias=True)')
            setattr(self.encoder, fc_name[0], nn.Linear(self.fc_in_features, class_n))
        elif len(fc_name) == 2:
            print(
                f'{getattr(getattr(self.encoder, fc_name[0]), fc_name[1])} -> Linear(in_features={self.fc_in_features}, out_features={class_n}, bias=True)')
            setattr(getattr(self.encoder, fc_name[0]), fc_name[1], nn.Linear(self.fc_in_features, class_n))

    def forward(self, x):
        x = self.encoder(x)
        return x


class DOLG(nn.Module):
    def __init__(self, model_name, class_n, mode='train'):
        super(DOLG, self).__init__()
        self.model_name = model_name.lower()
        self.class_n = class_n
        self.mode = mode
        self.encoder = timm.create_model(self.model_name, pretrained=False)

        self.fc_in_features = self.encoder.num_features
        self.global_pool = GeM()
        self.embedding_size = 2048

        self.neck = nn.Sequential(
            nn.Linear(self.fc_in_features, self.embedding_size, bias=True),
            nn.BatchNorm1d(self.embedding_size),
            torch.nn.PReLU()
        )

        self.head = ArcMarginProduct(self.embedding_size, self.class_n)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        x = self.neck(x)
        logits = self.head(x)

        return logits
