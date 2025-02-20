import torch
import torch.nn as nn

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, X_train):
        super(TCN, self).__init__()
        self.conv1 = nn.Conv1d(num_inputs, num_channels, kernel_size, padding=(kernel_size-1))
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size-1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Calculate the size after convolution
        self._calculate_conv_output_shape(num_inputs, X_train)
        self.fc = nn.Linear(self.conv_output_size, 1)

    def _calculate_conv_output_shape(self, num_inputs, X_train):
        # Calculate the output shape after the convolution layers
        dummy_input = torch.zeros(1, num_inputs, X_train.size(2))
        dummy_output = self.conv2(self.conv1(dummy_input))
        self.conv_output_size = dummy_output.numel()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x