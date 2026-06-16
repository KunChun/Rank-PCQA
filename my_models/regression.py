import torch.nn as nn


class ConCat_Regression(nn.Module):
    def __init__(self, infeatures = 6144): # (512*4 + 512*4 + 512*4)
        super(ConCat_Regression, self).__init__()

        self.linear1 = nn.Linear(infeatures, 512)
        self.bn = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 1)

    def forward(self, feats):
        x = self.linear1(feats)
        x = self.bn(x)
        x = self.linear2(x)

        return x