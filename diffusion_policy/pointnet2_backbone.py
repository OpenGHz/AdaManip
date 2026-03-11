import torch.nn as nn

from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2ClassificationSSG(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=self.hparams.get("model.use_xyz", True),
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=self.hparams.get("model.use_xyz", True),
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024],
                use_xyz=self.hparams.get("model.use_xyz", True),
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 40),
        )

    def _break_up_pc(self, pointcloud):
        xyz = pointcloud[..., 0:3].contiguous()
        features = (
            pointcloud[..., 3:].transpose(1, 2).contiguous()
            if pointcloud.size(-1) > 3
            else None
        )
        return xyz, features

    def forward(self, pointcloud):
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))