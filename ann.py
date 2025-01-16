from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Module, Linear, ReLU, Softmax
from torch.nn.init import xavier_uniform_, kaiming_uniform_
from livelossplot import PlotLosses
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data.dataset import Subset
import torch
from pandas import DataFrame
import matplotlib.pyplot as plt
from colorama import Fore


device = torch.device("cpu")
EPOCHS = 300
LEARNING_RATE = 0.001


class TrainCSVDataset(Dataset):
    def __init__(self, X: DataFrame, y: DataFrame):
        self.X = X.values
        self.y = y.values[:, 0]
        self.X = self.X.astype('float32')
        self.y = torch.tensor(self.y, dtype=torch.long, device=device)
        
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
 
    def get_splits(self, n_test):
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])


def prepare_train_data(X, y, n_test):
    dataset = TrainCSVDataset(X, y)
    train, test = dataset.get_splits(n_test)
    train_dl = DataLoader(train, batch_size=len(train), shuffle=True)
    test_dl = DataLoader(test, batch_size=len(train), shuffle=True)
    return train_dl, test_dl


class TestCSVDataset(Dataset):
    def __init__(self, X: DataFrame):
        self.X = X.values
        self.X = self.X.astype('float32')
        
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx]
    
    def get_splits(self):
        return Subset(self, range(len(self.X)))
    

def prepare_test_data(X):
    dataset = TestCSVDataset(X)
    test = dataset.get_splits()
    test_dl = DataLoader(test, batch_size=len(test), shuffle=True)
    return test_dl


class MLP(Module):
    def __init__(self, n_inputs, num_classes):
        super(MLP, self).__init__()

        # 1st layer
        self.hidden1 = Linear(n_inputs, 256)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # 2nd layer
        self.hidden2 = Linear(256, 128)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # 3rd layer
        self.hidden3 = Linear(128, 64)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # 4th layer
        self.hidden4 = Linear(64, num_classes)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1)

    def forward(self, X):
        # 1st layer
        X = self.hidden1(X)
        X = self.act1(X)
        # 2nd layer
        X = self.hidden2(X)
        X = self.act2(X)
        # 3rd layer
        X = self.hidden3(X)
        X = self.act3(X)
        return X
    

def train_model(train_dl: DataLoader, test_dl: DataLoader, model: MLP, epochs=EPOCHS, lr=LEARNING_RATE, patience=5):
    # Initialize metrics and setup Matplotlib
    epoch_count = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    stale_epochs = 0
    
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        for inputs, labels in train_dl:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.detach() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_dl.dataset)
        epoch_acc = running_corrects.float() / len(train_dl.dataset)
        train_losses.append(epoch_loss.item())
        train_accuracies.append(epoch_acc.item())
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0.0
        with torch.no_grad():
            for inputs, labels in test_dl: 
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.detach() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_val_loss = running_loss / len(test_dl.dataset)
        epoch_val_acc = running_corrects.float() / len(test_dl.dataset)
        val_losses.append(epoch_val_loss.item())
        val_accuracies.append(epoch_val_acc.item())
        
        # Update epoch count
        epoch_count.append(epoch + 1)
        
        # Clear and update plots
        ax[0].clear()
        ax[1].clear()
        
        # Loss plot
        ax[0].plot(epoch_count, train_losses, label='Train Loss', color='blue')
        ax[0].plot(epoch_count, val_losses, label='Validation Loss', color='orange')
        ax[0].set_title('Loss over Epochs')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].grid()        
        
        # Accuracy plot
        ax[1].plot(epoch_count, train_accuracies, label='Train Accuracy', color='blue')
        ax[1].plot(epoch_count, val_accuracies, label='Validation Accuracy', color='orange')
        ax[1].set_title('Accuracy over Epochs')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        ax[1].grid()

        plt.draw()
        plt.pause(0.01)

        if (val_losses[-1] < best_val_loss):
            best_val_loss = val_losses[-1]
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs > patience:
            print(Fore.RED + f"> Early stopping at epoch {epoch + 1}")
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the final plot visible

    return model, val_accuracies[-1]


def get_predictions(model, df_test: DataFrame):
    dataloader = prepare_test_data(df_test)

    model.eval()  # Set the model to evaluation mode
    predictions = []
    
    with torch.no_grad():  # Disable gradient computation to save memory
        for inputs in dataloader:
            inputs = inputs.to(device)  # Move inputs to the correct device (GPU or CPU)
            
            outputs = model(inputs)  # Get model predictions
            _, preds = torch.max(outputs, 1)  # Get the predicted class with the highest score
            
            predictions.extend(preds.cpu().numpy())  # Convert predictions to CPU and store them
    
    return predictions


def mlp(n_inputs, num_classes):
    return MLP(n_inputs, num_classes)