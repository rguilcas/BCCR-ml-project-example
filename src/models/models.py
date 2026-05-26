import torch.nn as nn



class OneLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, target_size):
        super(OneLayerMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, target_size)

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, target_size):
        super(TwoLayerMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, target_size)

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
class TwoLayerCNN(nn.Module):
    def __init__(self, input_size, hidden_size, target_size):
        super(TwoLayerCNN, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_input_size = (input_size // 4) * hidden_size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.linear_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, target_size)

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