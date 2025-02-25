from networkx.classes import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset


class TraditionalClassifier(KNeighborsClassifier):
    def __init__(self):
        super().__init__(n_neighbors=4)

    def evaluate(self, train_feats, train_labels, test_feats, test_labels, subject_id, proj):
        """
            1. Predict labels for the training set.
            2. Predict labels for the testing set.
            3. Compute training accuracy using accuracy_score.
            4. Compute testing accuracy using accuracy_score.
            5. Print the training and testing accuracy percentages.
        """
        # 1. Predict labels for the training set
        train_preds = self.predict(train_feats)
        # 2. Predict labels for the testing set
        test_preds = self.predict(test_feats)
        # 3. Compute training accuracy using accuracy_score
        print("Training Accuracy Score (%):")
        train_acc = accuracy_score(train_labels, train_preds) * 100
        print(train_acc)
        # 4. Compute testing accuracy using accuracy_score
        print("Testing Accuracy Score (%):")
        test_acc = accuracy_score(test_labels, test_preds) * 100
        print(test_acc)

        if proj:
            _plot_conf_mats(train_labels=train_labels, pred_train_labels=train_preds, test_labels=test_labels,
                    pred_test_labels=test_preds, subject_id=str(subject_id)+"_proj")
        else:
            _plot_conf_mats(train_labels=train_labels, pred_train_labels=train_preds, test_labels=test_labels,
                    pred_test_labels=test_preds, subject_id=str(subject_id)+"_no_proj")

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


def _plot_conf_mats(dataset="taiji_100", **kwargs):
    """
    Plots the confusion matrices for the training and testing data.
    Args:
        dataset: name of the dataset.
        train_labels: training labels.
        pred_train_labels: predicted training labels.
        test_labels: testing labels.
        pred_test_labels: predicted testing labels.
    """

    train_labels = kwargs['train_labels']
    pred_train_labels = kwargs['pred_train_labels']
    test_labels = kwargs['test_labels']
    pred_test_labels = kwargs['pred_test_labels']
    subject_id = kwargs['subject_id']

    train_confusion = confusion_matrix(train_labels, pred_train_labels)
    test_confusion = confusion_matrix(test_labels, pred_test_labels)

    # Plot the confusion matrices as seperate figures
    try:
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=train_confusion, display_labels=np.unique(train_labels))
        disp.plot(ax=ax, xticks_rotation='vertical')
        ax.set_title('Training Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'results/{dataset}_train_confusion.png', bbox_inches='tight', pad_inches=0)

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=test_confusion, display_labels=np.unique(test_labels))
        disp.plot(ax=ax, xticks_rotation='vertical')
        ax.set_title('Testing Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'results/{dataset}_{subject_id}_test_confusion.png', bbox_inches='tight', pad_inches=0)

    except:
        print("could not save confusion matrix")

