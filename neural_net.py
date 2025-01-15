import os
from datetime import datetime
import json

import pandas as pd

import torch.nn as nn 
from torch.nn.init import xavier_uniform_, kaiming_uniform_
import torch

from preprocessing import preprocess 

import torch.optim as optim
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

print("done with imports")

class MLP_1(nn.Module):
    def __init__(self, n_inputs, n_classes):
        super(MLP_1, self).__init__()
        self.hidden1 = nn.Linear(n_inputs, 512)
        self.act1 = nn.ReLU()

        self.hidden1_1 = nn.Linear(512, 512)
        self.act1_1 = nn.ReLU()
        
        self.hidden2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU()
        
        self.hidden3 = nn.Linear(256, 128)
        self.act3 = nn.ReLU()
        
        self.output = nn.Linear(128, n_classes)
        self.act4 = nn.Softmax(dim=1)

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden1_1(X)
        X = self.act1_1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.output(X)
        X = self.act4(X)
        return X

class RadiomicsClassifier:
    def __init__(self, input_size, num_classes, learning_rate=0.001, batch_size=32, epochs=25):
        self.input_size = input_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Define the model
        self.model = MLP_1(input_size, num_classes)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.training_stats = []

    def preprocess_data(self, df):
        # Train-test split
        X_train, X_test, y_train, y_test, self.le = preprocess(df)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Create DataLoaders
        self.train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=self.batch_size, shuffle=False)

    def train(self):
        self.training_stats = {"epochs":[]}
        self.model.train()
        for epoch in range(self.epochs):
            epoch_training_loss = 0.0
            epoch_test_loss = 0.0
            for (X_training_batch, y_training_batch),(X_test_batch, y_test_batch) in zip(self.train_loader,self.test_loader):
                self.optimizer.zero_grad()
                outputs = self.model(X_training_batch)
                training_loss = self.criterion(outputs, y_training_batch)
                training_loss.backward()
                self.optimizer.step()
                epoch_training_loss += training_loss.item()

                outputs = self.model(X_test_batch)
                test_loss = self.criterion(outputs, y_test_batch)
                self.optimizer.step()
                epoch_test_loss += test_loss.item()

            self.training_stats["epochs"].append({
                'epoch':epoch,
                'training_loss':epoch_training_loss / len(self.train_loader),
                'test_loss':epoch_test_loss / len(self.train_loader)
                })
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_training_loss/len(self.train_loader):.4f}")
        
        return self.model.state_dict()

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                outputs = self.model(X_batch)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds,  average='weighted')
        return accuracy,f1

    def save_model(self, test_data = None):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        filepath = "MLP/" + timestamp + "/"
        os.makedirs(filepath, exist_ok=True)
        torch.save(self.model.state_dict(), filepath+"/model")
        if test_data is not None:
            res = self.predict(test_data)
                
            test_predictions = torch.argmax(res, dim=1).cpu().numpy()
            test_labels = test_labels.cpu().numpy()
            
            test_accuracy = accuracy_score(test_labels, test_predictions)
            test_f1 = f1_score(test_labels, test_predictions, average="weighted")
            
            self.training_stats["accuracy"] = test_accuracy
            self.training_stats["f1"] = test_f1

        self.training_stats["accuracy"], self.training_stats["f1"]  = self.evaluate()

        with open(filepath+"/stats.json","w+") as f:
            json.dump(self.training_stats, f, indent=4)
        print(f"Model saved to {filepath}/model")

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        print(f"Model loaded from {filepath}")

    def predict(self, df):
        # Preprocess the new DataFrame
        X = preprocess(df,mode="test")
        X_new = X.values  # Assuming df contains only features
        # Convert to PyTorch tensor
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32)
        new_loader = DataLoader(TensorDataset(X_new_tensor), batch_size=self.batch_size, shuffle=False)

        # Make predictions
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch in new_loader:
                outputs = self.model(X_batch[0])
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
        return predictions
    

if __name__ == '__main__':
    df_train = pd.read_csv('datasets/train_radiomics_hipocamp.csv')
    df_test = pd.read_csv('datasets/test_radiomics_hipocamp.csv')

    # Initialize and run the classifier
    input_size = 2013  # Number of features
    num_classes = 5  # Number of output classes
    classifier = RadiomicsClassifier(input_size, num_classes)
    classifier.preprocess_data(df_train)
    classifier.train()
    a,f = classifier.evaluate()
    classifier.save_model()