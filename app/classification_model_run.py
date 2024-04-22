import torch
from torch.utils.data import Dataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PoseNet(nn.Module):
  def __init__(self, input_size, hidden_size_1, num_classes):
    super().__init__()
    self.layer1 = nn.Linear(input_size, hidden_size_1)
    self.relu = nn.ReLU()
    self.layer2 = nn.Linear(hidden_size_1, num_classes)
    self.dropout = nn.Dropout(p=0.3)
    self.bn1 = nn.BatchNorm1d(100)
    self.bn2 = nn.BatchNorm1d(num_classes)
  def forward(self, x):
    x = self.layer1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.layer2(x)
    return x

def run_classification(input_array):
    # Load the model architecture
    model = PoseNet(32, 120, 12) #32, 120, 12
    
    # Load the saved weights
    model.load_state_dict(torch.load('final_models/finalClassification.pth'))  #changed

    input_tensor = torch.from_numpy(input_array).to(device)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Set the model to evaluation mode
    model.eval()

    # Perform prediction
    with torch.no_grad():
        output = model.forward(input_tensor)
        probabilities = F.softmax(output, dim=-1)
        print("probabilities", probabilities)
        #predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_class = np.argmax(probabilities.cpu().numpy(),axis=-1)

    return predicted_class[0][0]
