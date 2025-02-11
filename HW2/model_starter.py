from networkx.classes import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset

class TraditionalClassifier(KNeighborsClassifier):
    def __init__(self):
        super().__init__(n_neighbors=4)

    def evaluate(self, train_labels, train_preds, test_labels, test_preds):
        """
            1. Predict labels for the training set.
            2. Predict labels for the testing set.
            3. Compute training accuracy using accuracy_score.
            4. Compute testing accuracy using accuracy_score.
            5. Print the training and testing accuracy percentages.
        """
        # 1. Predict labels for the training set
        print("Training Prediction Labels:")
        print(train_preds)
        # 2. Predict labels for the testing set
        print("Testing Prediction Labels:")
        print(test_preds)
        # 3. Compute training accuracy using accuracy_score
        print("Training Accuracy Score (%):")
        train_acc = accuracy_score(train_labels, train_preds) * 100
        print(train_acc)
        # 4. Compute testing accuracy using accuracy_score
        print("Testing Accuracy Score (%):")
        test_acc = accuracy_score(test_labels, test_preds) * 100
        print(test_acc)
        # 5. Print the training and testing accuracy percentages
        return test_acc



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, activation=nn.ReLU, num_layers=2):
        super(MLP, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
