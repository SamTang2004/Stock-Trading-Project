# imports

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Prediction: input - the stock market price of previous n days  output - stock market price of today
#

DATA_PATH = "MinMaxScaledData.json"


# create fully connected network

class NN(nn.Module):
    def __init__(self, input_size, num_classes):  # 28*28 input size (MNIST image number)
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, input_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(CNN, self).__init__()
        # same conv
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), )
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


class CNN2D(nn.Module):
    def __init__(self, input_size):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), )
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(45, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        print(x.shape)
        return x


class NN(nn.Module):
    def __init__(self, input_size, num_classes):  # 28*28 input size (MNIST image number)
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, input_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


from sklearn import preprocessing
# data processing class
# turns json data into a 180x8 : 1 kvp.
class CustomStockDataset(Dataset):
    def __init__(self, csv_file, discardLoc, transform=None):
        self._data = pd.read_json(csv_file)  # Make sure to import pandas

        self._transform = transform

        self._data["target"] = self._data["High"]
        self._data["target"] = self._data["High"].shift(1)
        self._data["feature"] = list(self._data.rolling(180))


        # throw away last bit of data

        print("Dataset shape:" + str(self._data.shape))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):  # idx is an integer

        # takes previous 6 months / 1 year / something and compiles it into a row Array
        # features: 2D Array of 180 rows / 7 cols
        # label: price of the next item

        sample = self._data.iloc[idx]
        # You may need to modify this part based on your specific CSV format
        _features = sample["feature"]

        featureArray = np.zeros((180, 7))
        for i in range(_features.shape[0]):
            featureArray[i][0] = _features.iloc[i]["Open"]
            featureArray[i][1] = _features.iloc[i]["High"]
            featureArray[i][2] = _features.iloc[i]["Low"]
            featureArray[i][3] = _features.iloc[i]["Close"]
            featureArray[i][4] = _features.iloc[i]["Volume"]
            featureArray[i][5] = _features.iloc[i]["Dividends"]
            featureArray[i][6] = _features.iloc[i]["Stock Splits"]
        _features = featureArray

        label = sample["target"]

        if self._transform:
            _features = self._transform(_features)

        return torch.from_numpy(_features), label


def train_model(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        L = 0

        for idx, (inputs, targets) in enumerate(train_loader):
            L += 1
            print(f"step {L} of total {len(train_loader)}")
            inputs = inputs.float()
            #print(inputs)
            outputs = model(inputs)
            targets = targets
            # print(inputs[0], outputs[0], targets[0])
            loss = criterion(outputs, targets.unsqueeze(1).float())
            print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


# Test function
def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0


    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.float()
            print(f"{idx} / {len(test_loader)}")
            outputs = model(inputs)

            batch_acc = 0
            batch_count = 0
            for i in range(len(outputs)):
                batch_count += 1
                batch_acc += abs(outputs[i]-targets[i])/abs(targets[i])

            print(f"batch avg error = {batch_acc*1.0 / batch_count*1.0}")

            loss = criterion(outputs, targets.unsqueeze(1).float())
            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)
    print(f"Test Loss: {average_loss}")


batch_size = 128
model = CNN2D(batch_size)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
raw_set = CustomStockDataset(DATA_PATH, 365)
test_set = torch.utils.data.Subset(raw_set, range(7000, len(raw_set)))
train_set = torch.utils.data.Subset(raw_set, range(3600, len(raw_set)))

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=True)
train_model(model, train_loader, criterion, optimizer)
test_model(model, test_loader, criterion)

"""
# hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 8
num_epochs = 1

# load data
train_set = datasets.MNIST(root= "dataset/", train= True, transform= transforms.ToTensor(), download= True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, )
test_set = datasets.MNIST(root= "dataset/", train= False, transform= transforms.ToTensor(), download= True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, )
# initialize network

#model = NN(input_size, num_classes).to(device)

model = CNN().to(device)
# CNN is an ND algo
# NN is 1D
x = torch.randn(64,1,28,28).to(device)
#x = torch.randn(16, 784).to(device)

print(model(x).shape)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
# train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # flatten matrix
        #data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        # grad descent
        optimizer.step()



# check accuracy on training results

def check_accuracy(loader, model):
    if loader.dataset.train :
        print("currently checking training data")
    else:
        print("currently checking testing data")

    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device = device)
            #x = x.reshape(x.shape[0],-1)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/num_samples * 100} %")

    model.train()
    return num_correct*1.0 / num_samples*1.0

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)


"""
