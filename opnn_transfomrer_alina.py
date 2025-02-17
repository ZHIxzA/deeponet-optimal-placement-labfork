import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class opnn(nn.Module):
    def __init__(self, branch2_dim, trunk_dim, geometry_dim):
        super(opnn, self).__init__()

        # Load a pretrained Vision Transformer (ViT)
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove classifier to get raw features

        # Freeze ViT parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Fully connected layer to adjust ViT output size
        self.fc1 = nn.Linear(768, 64)

        # Source location branch (unchanged)
        self._branch2 = nn.Sequential(
            nn.Linear(branch2_dim[0], branch2_dim[1]),
            nn.ReLU(),
            nn.Linear(branch2_dim[1], branch2_dim[2]),
            nn.ReLU(),
            nn.Linear(branch2_dim[2], branch2_dim[3])
        )

        # Trunk network (unchanged)
        self._trunk = nn.Sequential(
            nn.Linear(trunk_dim[0], trunk_dim[1]),
            nn.Tanh(),
            nn.Linear(trunk_dim[1], trunk_dim[2]),
            nn.Tanh(),
            nn.Linear(trunk_dim[2], branch2_dim[3])
        )

    def forward(self, geometry, source_loc, coords):
        # Process geometry image through ViT (frozen)
        with torch.no_grad():
            x = self.vit(geometry)  # Shape: (batch, 768)

        y_br1 = F.relu(self.fc1(x))  # Shape: (batch, 64)

        # Process source location through FC network
        y_br2 = self._branch2(source_loc)  # Shape: (batch, 64)

        # Combine branch outputs
        y_br = y_br1 * y_br2

        # Process coordinates through trunk network
        y_tr = self._trunk(coords)

        # Perform tensor product over the last dimension of y_br and y_tr
        y_out = torch.einsum("bf,bhwf->bhw", y_br, y_tr)

        return y_out

    def loss(self, geometry, source_loc, coords, target_pressure):
        y_out = self.forward(geometry, source_loc, coords)
        numerator = torch.norm(y_out - target_pressure, p=2)
        denominator = torch.norm(target_pressure, p=2)  # Avoid division by zero
        loss = (numerator / denominator) ** 2
        return loss
