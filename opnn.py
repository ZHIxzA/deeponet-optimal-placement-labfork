import torch
import torch.nn as nn
import torch.nn.functional as F


class opnn(nn.Module):
    def __init__(self, branch2_dim, trunk_dim, geometry_dim):
        super(opnn, self).__init__()

        # Define the CNN for the geometry image (Branch 1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Calculate the output size after 3 pooling layers (assuming 2x2 max pooling)
        pooled_height = geometry_dim[0] // 8  # Divide the height by 8
        pooled_width = geometry_dim[1] // 8   # Divide the width by 8

        # Geometry branch (adjust output size to match trunk network)
        self.fc1 = nn.Linear(64 * pooled_height * pooled_width, 64)  # Output 64 features

        # Source location branch
        self._branch2 = nn.Sequential(
            nn.Linear(branch2_dim[0], branch2_dim[1]),
            nn.ReLU(),
            nn.Linear(branch2_dim[1], branch2_dim[2]),
            nn.ReLU(),
            nn.Linear(branch2_dim[2], branch2_dim[3]) 
        )

        # Trunk network (adjust output to 64 features)
        self._trunk = nn.Sequential(
            nn.Linear(trunk_dim[0], trunk_dim[1]),
            nn.Tanh(),
            nn.Linear(trunk_dim[1], trunk_dim[2]),
            nn.Tanh(),
            nn.Linear(trunk_dim[2], branch2_dim[3])  
        )

    def forward(self, geometry, source_loc, coords):
        # Process geometry image through CNN
        x = F.relu(self.conv1(geometry))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        y_br1 = F.relu(self.fc1(x))

        # Process source location through FC network
        y_br2 = self._branch2(source_loc)

        # Combine branch outputs
        y_br = y_br1 * y_br2

        # Process coordinates through trunk network
        y_tr = self._trunk(coords)
        # Perform tensor product over the last dimension of y_br and y_tr
        y_out = torch.einsum("bf,bhwf->bhw", y_br, y_tr)

        return y_out
    
    def loss(self, geometry, source_loc, coords, target_pressure):
        y_out = self.forward(geometry, source_loc, coords)
        #loss = ((y_out - target_pressure) ** 2).mean() #L2
        numerator = torch.norm(y_out - target_pressure, p=2)
        denominator = torch.norm(target_pressure, p=2) # Avoid division by zero
        loss = (numerator / denominator) ** 2
        #print(f"Relative L2 Loss: {(numerator / denominator).item()}")


        return loss
