# Building-a-Robust-Anomaly-Detection-Model-for-Financial-Transactions

# Project Title
Project Title: Transaction Anomaly Detection System
Overview
Transaction Anomaly Detection System is designed to identify unusual patterns or anomalies in transaction data. The system preprocesses the dataset, addresses missing values, encodes categorical variables, performs feature engineering, and visualizes the data to gain insights into transaction patterns.

System Requirements
Python 3.x
Required Python Libraries: Pandas, NumPy, scikit-learn, matplotlib, seaborn
Installation
Ensure Python 3.x is installed.
Install required Python libraries:
Pandas
NumPy
scikit-learn
matplotlib
seaborn
Dataset
The dataset should be available in a CSV file format.
The CSV file should contain transaction-related features.
Ensure the dataset is sufficiently large and diverse for effective data modeling.
Dataset link: Dataset Link
Usage Instructions
Data Preprocessing
Load the dataset.
Address missing values using mean imputation for numerical features and forward fill for categorical features.
Encode categorical variables using label encoding.
Standardize numerical features.
Feature Engineering
Generate new features like 'Transaction_Frequency_Ratio' and 'Transaction_Amount_Per_Volume' based on domain knowledge.
Data Visualization
Plot a histogram of Transaction_Amount to visualize its distribution.
Plot a scatter plot of Transaction_Amount vs Frequency_of_Transactions to identify any patterns.
Plot a box plot of Transaction_Amount by Account_Type to compare transaction amounts across different account types.
Dimensionality Reduction (t-SNE)
Use t-SNE for dimensionality reduction to visualize high-dimensional data in 2D space.
Notes
Ensure datasets adhere to the appropriate format.
Customize the code as needed or incorporate it into an ongoing Python project.
Anomaly Detection Methods:
1. DBSCAN:
Features Used: Transaction_Amount, Transaction_Volume
Anomalies Detected:
Indices: [59, 147, 230, 246, 422, 426, 484, 550, 570, 575, 724, 936, 991]
2. Local Outlier Factor (LOF):
Features Used: Transaction_Amount, Transaction_Volume
Parameters: n_neighbors=20, contamination=0.1
Anomalies Detected:
Indices: [13, 21, 22, 34, 35, 44, 55, 58, 59, ... , 991, 992]
3. Variational Autoencoder (VAE):
Features Used: Transaction_Amount, Transaction_Volume
Architecture:
Encoder: Dense(32, activation='relu') -> Dense(latent_dim)
Sampling: Lambda layer
Decoder: Dense(32, activation='relu') -> Dense(X_scaled.shape[1])
Loss Function: MeanSquaredError
Training: 50 epochs, batch_size=32, learning_rate=0.001
Anomalies Detected:
Indices: [2, 14, 77, 129, 141, 153, 158, ... , 951, 958]
Data Preprocessing:
1. Imputation:
Numerical Features: Mean imputation
Categorical Features: Forward fill
2. Encoding:
Categorical Variables: Label Encoding
3. Feature Engineering:
New Features Created:
Transaction_Frequency_Ratio = Frequency_of_Transactions / Time_Since_Last_Transaction
Transaction_Amount_Per_Volume = Transaction_Amount / Transaction_Volume
Data Splitting:
Features: Transaction_Amount, Transaction_Volume, Average_Transaction_Amount, Frequency_of_Transactions, Time_Since_Last_Transaction, Day_of_Week, Time_of_Day, Age, Gender, Income
Target: Account_Type
Training set: 640 samples
Validation set: 160 samples
Test set: 200 samples
Outlier Score Analysis:
1. LOF:
Scores:
Training Data: Negative Outlier Factor
Validation Data: Predicted outlier scores
Test Data: Predicted outlier scores
2. KDE:
Scores:
Training Data: Log probability densities
Validation Data: Log probability densities
Test Data: Log probability densities
Thresholds:
LOF: 95th percentile of outlier_scores_train
KDE: 5th percentile of log_prob_train
LOF Outlier Scores - Training Data:
The outlier scores for the training data range from approximately -2.13 to -0.96.
The threshold for classifying outliers is not provided, but you seem to have used a threshold to classify data points as outliers.
LOF Outlier Scores - Validation Data:
The outlier scores for the validation data are either -1 or 1, which suggests that the LOF analysis has classified most of the validation data as non-outliers.
LOF Outlier Scores - Test Data:
The outlier scores for the test data range from approximately -2.13 to 1.
Similar to the validation data, the LOF analysis mostly classifies the test data as non-outliers.
KDE Log Probability Densities - Training Data:
The KDE log probability densities for the training data are consistently at 7.37.
Based on this information, it seems like the LOF analysis is identifying some outliers in the training data but not as much in the validation and test data. The KDE log probability densities for the training data are consistent, suggesting that the data points are well-distributed according to the KDE model.
Outlier Analysis
Summary Statistics of Outliers - Training Data:
Features with Outliers: Transaction_Amount, Transaction_Volume, Average_Transaction_Amount, Frequency_of_Transactions, Time_Since_Last_Transaction, Day_of_Week, Time_of_Day, Age, Gender, Income
Number of Outliers:
Training Data: 32
Validation Data: 144
Test Data: 180
Observations:
Outliers in the training data have different statistical distributions compared to the overall data.
Larger standard deviations and ranges are observed in outlier features.
Visualization:
A boxplot visualizing the distribution of outlier values across different features is generated.
Model Evaluation
Precision-Recall Curve Metrics:
PR AUC: 1.0 (Perfect performance)
Anomaly Detection Rate (ADR): 1.0
False Positive Rate (FPR): 0.0
Observations:
The model performs exceptionally well in distinguishing anomalies from normal data.
Model Refinement
Optimized Hyperparameters:
Isolation Forest Model:
contamination: 0.01
max_samples: 0.1
Bayesian Optimization Results:
The best hyperparameters were found using Bayesian optimization with 20 iterations.
Grubbs' Test for Outlier Detection
Functionality:
Purpose: Detect outliers in a single feature or dataset.
Parameters: Input data array, significance level (alpha).
Output: Detected outlier value or None if no outlier is detected.
Grubbs' Test for Outlier Detection: Detects outliers in a given dataset using Grubbs' Test.

Outlier Detection using LOF and KDE:

LOF (Local Outlier Factor) is used to detect outliers in a synthetic dataset.
KDE (Kernel Density Estimation) is also used for outlier detection.
Evaluation Metrics for LOF and KDE: Computes precision, recall, F1-score, and AUC-ROC for both LOF and KDE.

Performance Comparison of LOF and KDE Models: Visualizes the performance of LOF and KDE using bar plots.

Linear Regression Model: Fits a linear regression model to the data and calculates mean squared error (MSE), coefficients, and intercept.

Neural Network Model (CNN): Defines a Convolutional Neural Network (CNN) architecture using PyTorch.

Data Preprocessing and Loading: Demonstrates how to preprocess and load data into PyTorch DataLoader.

Training Loop for Neural Network: Illustrates a training loop for a neural network model using PyTorch.

Random Forest Classifier with Hyperparameter Optimization: Trains a Random Forest classifier, evaluates its performance, and performs hyperparameter optimization using GridSearchCV.

CNN Model for MNIST Dataset: Defines and trains a CNN model for the MNIST dataset using PyTorch.

MLP Model for MNIST Dataset: Defines and trains a Multi-Layer Perceptron (MLP) model for the MNIST dataset using PyTorch.
Imports:

torch and related modules for deep learning operations
MNIST dataset from torchvision for loading the MNIST dataset
ToTensor transformation from torchvision.transforms to convert PIL images to tensors
DataLoader from torch.utils.data for batching and shuffling data
matplotlib.pyplot and numpy for visualization
Neural Network Model:

A simple feed-forward neural network (SimpleNN) with three fully connected layers:
Input layer: 784 neurons (flattened MNIST image size)
Hidden layer 1: 128 neurons
Hidden layer 2: 64 neurons
Output layer: 10 neurons (number of classes in MNIST)
Data Preparation:

Transformation (ToTensor) to convert images to PyTorch tensors.
MNIST dataset loaded with training images, labels, and the specified transformation.
DataLoader for batching and shuffling the training dataset.
Model Training:

Loss function: Cross-Entropy Loss
Optimizer: Adam with a learning rate of 0.001
Training loop that iterates over epochs and batches:
Forward pass
Compute loss
Backward pass
Update weights
After each epoch, the average loss over 100 batches is printed.
Model Evaluation:

Model is set to evaluation mode.
Validation dataset loaded and transformed similarly to the training dataset.
Model's accuracy is computed on the validation dataset by comparing predicted labels to actual labels.
Random Accuracy Comparison:

Function generate_random_accuracies(num_models) generates random accuracy values between 90% and 100% for a specified number of models.
Two random accuracy values are generated and stored in the accuracies variable.
The accuracies of the two models are plotted using matplotlib, showing a comparison bar chart.

