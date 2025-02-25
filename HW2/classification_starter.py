'''
Your Details: (The below details should be included in every python
file that you add code to.)
{
}
'''
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from model_starter import TraditionalClassifier
from model_starter import  MLP
from torch.optim.lr_scheduler import StepLR

# Global Variables
VERBOSE = False

def feature_selection(feats, labels, var_threshold=0.01, top_k=50):
    # Step 1: Remove low-variance features
    selector = VarianceThreshold(threshold=var_threshold)
    feats_var_filtered = selector.fit_transform(feats)
    
    # Step 2: Compute mutual information between features and labels
    mi_scores = mutual_info_classif(feats_var_filtered, labels)
    
    # Step 3: Select top K features based on mutual information
    top_k_indices = np.argsort(mi_scores)[-top_k:]  # Select indices of top features
    selected_feats = feats_var_filtered[:, top_k_indices]
    
    return selected_feats

def convert_features_to_loader(train_feats_proj, train_labels, test_feats_proj, test_labels, batch_size):
    '''
    TODO: Convert NumPy arrays to PyTorch tensors and create DataLoader instances.

    1. Convert `train_feats_proj` to a PyTorch tensor with dtype `torch.float32`.
    2. Convert `train_labels` to a PyTorch tensor with dtype `torch.long` (required for classification tasks).
    3. Create a `TensorDataset` from `train_feats_proj` and `train_labels`.
    4. Initialize a `DataLoader` for training, specifying `batch_size` and enabling shuffling for better generalization.
    5. Convert `test_feats_proj` to a PyTorch tensor with dtype `torch.float32`.
    6. Convert `test_labels` to a PyTorch tensor with dtype `torch.long`.
    7. Create a `TensorDataset` from `test_feats_proj` and `test_labels`.
    8. Initialize a `DataLoader` for testing, specifying `batch_size` but disabling shuffling to maintain order.
    9. Return the `train_loader` and `test_loader` for use in model training and evaluation.

    '''
    # Convert to PyTorch tensors
    train_feats_tensor = torch.tensor(train_feats_proj, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_feats_tensor = torch.tensor(test_feats_proj, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    class JointDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    # Create dataset instances
    train_dataset = JointDataset(train_feats_tensor, train_labels_tensor)
    test_dataset = JointDataset(test_feats_tensor, test_labels_tensor)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def deep_learning(train_feats_proj, train_labels, test_feats_proj, test_labels, subject_id,
                  hidden_dim=64, num_layers=3, batch_size=64, learning_rate=0.001, epochs=100):
    
    input_dim = train_feats_proj.shape[1]
    output_dim = len(torch.unique(torch.tensor(train_labels)))
    
    model = MLP(input_dim, output_dim, hidden_dim, nn.ReLU, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    train_loader,test_loader = convert_features_to_loader(train_feats_proj, train_labels, test_feats_proj, test_labels, batch_size)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)  # Model call
            loss = criterion(outputs, batch_labels)
            loss.backward()

            optimizer.step()  # Update model parameters
            total_loss += loss.item()
        scheduler.step()  # Update learning rate if using a scheduler (optional)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    test_accuracy = correct / total * 100
    print(f"Deep Learning Test Accuracy: {test_accuracy:.2f}%")

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in train_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total * 100
    print(f"Deep Learning Train Accuracy: {accuracy:.2f}%")

    return test_accuracy


def perform_traditional(train_feats, train_labels, test_feats, test_labels, subject_id, proj):
    '''
    Perform Traditional Classification with TraditionalClassifier(KNeighborsClassifier):
    1. Create instance of Classifier
    2. Fit to the Train data {Projected, Not Projected}
    3. Evaluate on both Train and Test data (Seperately)
    4. Return Test Accuracy
    '''
    
    classifiers = {
        "traditional_classifier": TraditionalClassifier(),
    }

    for name, clf in classifiers.items():
        if name == "traditional_classifier":
            # Train the classifier
            clf.fit(train_feats, train_labels)
            # Evaluate on Train and Test 
            return clf.evaluate(train_feats, train_labels, test_feats, test_labels, subject_id, proj)

        else:
            return clf(train_feats_proj, train_labels, test_feats_proj, test_labels, subject_id, proj)


def load_new_dataset(verbose=True,subject_index=1):
    # Using the 3 datasets provided to us {Taiji_dataset_100.csv, Taiji_dataset_200.csv, Taiji_dataset_300.csv}
    # dataset = np.loadtxt("Taiji_dataset_100.csv", delimiter=",", dtype=float,skiprows=1,usecols=range(0, 70) )
    # dataset = np.loadtxt("Taiji_dataset_200.csv", delimiter=",", dtype=float,skiprows=1,usecols=range(0, 70) )
    dataset = np.loadtxt("Taiji_dataset_300.csv", delimiter=",", dtype=float,skiprows=1,usecols=range(0, 70) )
    
    # Extract `person_idxs` (last column)
    person_idxs = dataset[:, -1]  # All rows, last column
    labels = dataset[:, -2]  # All rows, second last column

    # Extract `feats` (remaining columns except the last two)
    #
    feats = dataset[:, :-2].T  # All rows, all columns except the last two, transposed
    #TODO: Feature selection (EXTRA CREDIT). You can comment out the feature selection part if you are not implementing it.
    # feats = feature_selection(feats)
    feature_mask = np.var(feats, axis=1 ) > 0
    # Train mask
    # Leave one subject out (LOSO)
    train_mask = person_idxs != subject_index

    train_feats = feats[feature_mask, :][:, train_mask].T
    train_labels = labels[train_mask].astype(int)
    test_feats = feats[feature_mask, :][:, ~train_mask].T
    test_labels = labels[~train_mask].astype(int)
    if verbose:
        print(f'{dataset} Dataset Loaded')
        print(f'\t# of Classes: {len(np.unique(train_labels))}')
        print(f'\t# of Features: {train_feats.shape[1]}')
        print(f'\t# of Training Samples: {train_feats.shape[0]}')
        print('\t# per Class in Train Dataset:')
        for cls in np.unique(train_labels):
            print (f'\t\tClass {cls}: {np.sum(train_labels == cls)}')
        print(f'\t# of Testing Samples: {test_feats.shape[0]}')
        print('\t# per Class in Test Dataset:')
        for clas in np.unique(test_labels):
            print(f'\t\tClass {clas}: {np.sum(test_labels == clas)}')

    return train_feats, train_labels, test_feats, test_labels


def fisher_projection(train_feats, train_labels):
    '''
    Fisher Projection (LDA):
    Steps:
    1. Compute the overall mean of the training features.
    2. Calculate the mean vector for each class.
    3. Compute the within-class scatter matrix (S_w):
       - For each class, compute the deviation of each sample from the class mean.
       - Compute the scatter contribution for each class and sum them to obtain S_w.
    4. Compute the between-class scatter matrix (S_b):
       - Compute the deviation of each class mean from the overall mean.
       - Compute the scatter contribution for each class weighted by the number of samples.
       - Sum them to obtain S_b.
    5. Compute the transformation matrix J(W) = S_w^-1 * S_b.
    6. Solve for the eigenvalues and eigenvectors of J(W).
    7. Sort eigenvectors in descending order based on their absolute eigenvalues.
    8. Select the top two eigenvectors for dimensionality reduction.
    9. Return the selected eigenvectors for projecting data into a lower-dimensional space.

    Note: Ensure numerical stability while computing the inverse of S_w.
        - Using np.linalg.pinv(S_w), pseudo inverse
    '''

    # 1. Compute the overall mean of the training features.
    overall_mean = np.mean(train_feats, axis=0)

    # 2. Calculate the mean vector for each class.
    class_labels = np.unique(train_labels)
    mean_vectors = {label: np.mean(train_feats[train_labels == label], axis=0) for label in class_labels}

    # 3. Compute within-class scatter matrix S_w
    n_features = train_feats.shape[1]

    S_w = np.zeros((n_features, n_features))

    for label in class_labels:
        class_samples = train_feats[train_labels == label]
        class_mean = mean_vectors[label]
        scatter = np.zeros((n_features, n_features))

        for sample in class_samples:
            # For each class, compute the deviation of each sample from the class mean.
            diff = (sample - class_mean).reshape(n_features, 1)
            # Compute the scatter contribution for each class and sum them to obtain S_w.
            scatter += np.dot(diff,diff.T)  # Outer product
        S_w += scatter  # Accumulate across classes

    # 4. Compute between-class scatter matrix S_b
    S_b = np.zeros((n_features, n_features))

    for label in class_labels:
        class_mean = mean_vectors[label].reshape(n_features, 1)
        overall_mean_vec = overall_mean.reshape(n_features, 1)
        num_samples = train_feats[train_labels == label].shape[0]
        # Compute the deviation of each class mean from the overall mean.
        diff = class_mean - overall_mean_vec
        # Compute the scatter contribution for each class weighted by the number of samples
        # Sum them to obtain S_b
        S_b += num_samples * (np.dot(diff,diff.T))

    # 5. Compute the transformation matrix J(W) = S_w^-1 * S_b
    S_w_inv = np.linalg.pinv(S_w) # Ensure numerical stability while computing the inverse of S_w (pseudo inverse)
    J_W = np.dot(S_w_inv,S_b)

    # 6. Solve for the eigenvalues and eigenvectors of J(W)
    eigvals, eigvecs = np.linalg.eig(J_W)
    eigvecs = np.real(eigvecs)  # Take only the real part

    # 7. Sort eigenvectors in descending order based on their absolute eigenvalues
    sorted_indices = np.argsort(np.abs(eigvals))[::-1]

    # 8. Select the top eigenvectors for dimensionality reduction
    top_eigvecs = eigvecs[:, sorted_indices[:10]]  

    # 9. Return the selected eigenvectors for projecting data into a lower-dimensional space
    return top_eigvecs


def run_deep_learning():
    '''
        Leave-One-Subject-Out (LOSO) cross-validation with Deep Learning MLP:
        Loop over all subjects:
        - Each iteration, select one subject index as the test set.
        - Use all other subjects as the training set.
        Store accuracy scores:
        - Save the accuracy score for each test subject.
        - Repeat this process for all subjects.
        - Then find the average of all 10 subjects to get the final accuracy score
    '''
    total_with_proj = 0
    total_without_proj = 0
    # Loop over all subjects
    for i in range(1,11):
        print("")
        print("-----------------------------------------------------------------------------")
        print("Leave-One-Subject-Out (LOSO), Subject ID: ", str(i))
        # Each iteration, select one subject index as the test set
        # Use all other subjects as the training set
        train_feats, train_labels, test_feats, test_labels = load_new_dataset(verbose=VERBOSE, subject_index=i)
        
        train_eigens = fisher_projection(train_feats, train_labels)

        # Project training/testing feats through top eigen vectors
        train_feats_proj = np.dot(train_feats, train_eigens)
        test_feats_proj = np.dot(test_feats, train_eigens)
        
        print("")
        print("Deep Learning MLP with Fisher Projection (LDA):")
        test_acc_proj = deep_learning(train_feats_proj, train_labels, test_feats_proj, test_labels, i)
        print("")
        print("Deep Learning MLP without Fisher Projection (LDA):")
        test_acc_no_proj = deep_learning(train_feats, train_labels, test_feats, test_labels, i)

        # Store accuracy scores
        # Save the accuracy score for each test subject
        total_with_proj += test_acc_proj
        total_without_proj += test_acc_no_proj

    print("")
    print("-----------------------------------------------------------------------------")
    print("")
    print("Final Results: ")
    print("Leave-One-Subject-Out (LOSO) Cross-Validation, Average across 10 subjects")
    print("With Fisher Projection (LDA): " + str(total_with_proj / 10) + " %")
    print("Without Fisher Projection (LDA): " + str(total_without_proj / 10) + " %")

def run_classification():
    '''
        Leave-One-Subject-Out (LOSO) cross-validation:
        Loop over all subjects:
        - Each iteration, select one subject index as the test set.
        - Use all other subjects as the training set.
        Store accuracy scores:
        - Save the accuracy score for each test subject.
        - Repeat this process for all subjects.
        - Then find the average of all 10 subjects to get the final accuracy score
    '''
    total_with_proj = 0
    total_without_proj = 0
    # Loop over all subjects
    for i in range(1,11):
        print("")
        print("-----------------------------------------------------------------------------")
        print("Leave-One-Subject-Out (LOSO), Subject ID: ", str(i))
        # Each iteration, select one subject index as the test set
        # Use all other subjects as the training set
        train_feats, train_labels, test_feats, test_labels = load_new_dataset(verbose=VERBOSE, subject_index=i)
        train_eigens = fisher_projection(train_feats, train_labels)

        # Project training/testing feats through top eigen vectors
        train_feats_proj = np.dot(train_feats, train_eigens)
        test_feats_proj = np.dot(test_feats, train_eigens)
        
        print("")
        print("Traditional KNeighborsClassifier with Fisher Projection (LDA):")
        test_acc_proj = perform_traditional(train_feats_proj, train_labels, test_feats_proj, test_labels, i, True)
        print("")
        print("Traditional KNeighborsClassifier without Fisher Projection (LDA):")
        test_acc_no_proj = perform_traditional(train_feats, train_labels, test_feats, test_labels, i, False)

        # Store accuracy scores
        # Save the accuracy score for each test subject
        total_with_proj += test_acc_proj
        total_without_proj += test_acc_no_proj

    print("")
    print("-----------------------------------------------------------------------------")
    print("")
    print("Final Results: ")
    print("Leave-One-Subject-Out (LOSO) Cross-Validation, Average across 10 subjects")
    print("With Fisher Projection (LDA): " + str(total_with_proj / 10) + " %")
    print("Without Fisher Projection (LDA): " + str(total_without_proj / 10) + " %")


def main():
    #run_classification()
    run_deep_learning()

    
if __name__ == '__main__':
    main()

