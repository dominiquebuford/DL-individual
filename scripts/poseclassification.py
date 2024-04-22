import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def define_loaders(df):
    """
    create train and test dataloaders based on the information in the dataframe df
    """
    X = df.drop('Activity',axis=1).to_numpy()
    y = df['Activity'].astype('category').cat.codes
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    y_resampled= np.array(y_resampled)
    y_test = np.array(y_test)
    trainset = TensorDataset(torch.from_numpy(X_resampled).float(),
                         torch.from_numpy(y_resampled).long())

    testset = TensorDataset(torch.from_numpy(X_test).float(),
                         torch.from_numpy(y_test).long())

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=True)

    return trainloader,testloader


class PoseNet(nn.Module):
  """
  Define the structure of the NN model used for classification
  """
  def __init__(self, input_size, hidden_size_1, num_classes):
    super().__init__()
    self.layer1 = nn.Linear(input_size, hidden_size_1)
    self.relu = nn.ReLU()
    self.layer2 = nn.Linear(hidden_size_1,num_classes)
    self.dropout = nn.Dropout(p=0.3)   #0.3
    self.bn1 = nn.BatchNorm1d(100)
    self.bn2 = nn.BatchNorm1d(num_classes)
  def forward(self, x):
    x = self.layer1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.layer2(x)

    return x

def main():
    np.random.seed(42)
    df_annotations = pd.read_csv('data/all_distances.csv')
    categoryOrder = df_annotations['Activity'].astype('category').cat.categories
    print(categoryOrder)
    trainloader, testloader = define_loaders(df_annotations)
    model = PoseNet(120, 250, 12)   #32, 150,12
    criterion = nn.CrossEntropyLoss()
    weight_decay = 1e-5 #1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)  #0.015

    #train the model
    num_epochs = 60 #40
    losses = []
    model = model.to(device) # Send model to GPU if available
    model.train()
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()

        # Set up lists to store true and predicted values
        y_true = []
        test_preds = []

        # Calculate the predictions on the test set and add to list
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # Feed inputs through model to get raw scores
            logits = model.forward(inputs)
            # Convert raw scores to probabilities (not necessary since we just care about discrete probs in this case)
            probs = F.softmax(logits,dim=1)
            preds = np.argmax(probs.cpu().numpy(),axis=1)
            test_preds.extend(preds)
            labels = [tensor.item() for tensor in labels]
            y_true.extend(labels)
    conf_matrix = confusion_matrix(y_true, test_preds)

    # Create a heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=categoryOrder, yticklabels=categoryOrder)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    #torch.save(model.state_dict(), 'flaskr/final_models/finalClassification_versionPTP.pth')

if __name__ == "__main__":
    main()