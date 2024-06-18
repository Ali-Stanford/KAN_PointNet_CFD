#First try with fake data
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline

# Define a KANLayer (simplified for example)
class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_splines=5, k=3, device='cpu'):
        super(KANLayer, self).__init__()
        self.splines = nn.ModuleList([CubicSpline(0, 1, [0, 1]) for _ in range(out_dim)])
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        # Apply spline functions to the input
        batch_size, num_points, _ = x.size()
        x = x.permute(0, 2, 1).reshape(-1, self.in_dim)  # (batch * num_points, in_dim)
        out = torch.stack([self.splines[i](x[:, i]) for i in range(self.out_dim)], dim=1)
        out = out.view(batch_size, num_points, self.out_dim).permute(0, 2, 1)
        return out

# Updated PointNet using KANLayer
class KANPointNet(nn.Module):
    def __init__(self, input_channels, output_channels, scaling=1.0, device='cpu'):
        super(KANPointNet, self).__init__()

        self.kan1 = KANLayer(input_channels, int(64 * scaling), device=device)
        self.kan2 = KANLayer(int(64 * scaling), int(64 * scaling), device=device)
        self.kan3 = KANLayer(int(64 * scaling), int(64 * scaling), device=device)
        self.kan4 = KANLayer(int(64 * scaling), int(128 * scaling), device=device)
        self.kan5 = KANLayer(int(128 * scaling), int(1024 * scaling), device=device)
        self.kan6 = KANLayer(int(1024 * scaling) + int(64 * scaling), int(512 * scaling), device=device)
        self.kan7 = KANLayer(int(512 * scaling), int(256 * scaling), device=device)
        self.kan8 = KANLayer(int(256 * scaling), int(128 * scaling), device=device)
        self.kan9 = KANLayer(int(128 * scaling), int(128 * scaling), device=device)
        self.kan10 = KANLayer(int(128 * scaling), output_channels, device=device)

    def forward(self, x):
        num_points = x.size(-1)

        # Shared KAN layers for local feature extraction
        x = F.relu(self.kan1(x))
        x = F.relu(self.kan2(x))
        local_feature = x

        # Shared KAN layers for higher-level features
        x = F.relu(self.kan3(x))
        x = F.relu(self.kan4(x))
        x = F.relu(self.kan5(x))

        # Global feature pooling
        global_feature = F.max_pool1d(x, kernel_size=num_points)
        global_feature = global_feature.expand(-1, -1, num_points)

        # Concatenate local and global features
        x = torch.cat([local_feature, global_feature], dim=1)

        # Further processing with KAN layers
        x = F.relu(self.kan6(x))
        x = F.relu(self.kan7(x))
        x = F.relu(self.kan8(x))
        x = F.relu(self.kan9(x))
        x = self.kan10(x)  # No activation for the final layer

        return x

# Example usage
input_channels = 3
output_channels = 40
scaling = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pointnet_kan = KANPointNet(input_channels, output_channels, scaling, device=device).to(device)
input_data = torch.rand((32, input_channels, 1024)).to(device)  # Example input
output = pointnet_kan(input_data)
