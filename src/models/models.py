import torch.nn as nn

class OneLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size_mlp, target_size):
        super(OneLayerMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size_mlp)
        self.fc2 = nn.Linear(hidden_size_mlp, target_size)

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size_mlp, target_size):
        super(TwoLayerMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size_mlp)
        self.fc2 = nn.Linear(hidden_size_mlp, hidden_size_mlp)
        self.fc3 = nn.Linear(hidden_size_mlp, target_size)

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
class TwoLayerCNN(nn.Module):
    def __init__(self, n_channels_input_cnn, target_size, image_size,
                 n_kernels_cnn=8, hidden_size_mlp=64):
        super(TwoLayerCNN, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(n_channels_input_cnn, n_kernels_cnn, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(n_kernels_cnn, n_kernels_cnn*2, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_input_size = (image_size // 4 // 4) * n_kernels_cnn*2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.linear_input_size, hidden_size_mlp)
        self.fc2 = nn.Linear(hidden_size_mlp, target_size)

    def forward(self, x):
        # Convolutional encoder part
        ## Block 1
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool1(out)
        ## Block 2
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        # Flatten
        out = self.flatten(out)
        # Prediction head
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out