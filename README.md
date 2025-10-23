### 💳 Transaction Anomaly Detection System
---
**Goal:** Identify unusual patterns (anomalies) in financial transaction data.

| Component | Status | Details |
| :--- | :--- | :--- |
| **Data Preprocessing** | ✅ Complete | Imputation, Encoding, Standardization |
| **Feature Engineering** | ✅ Complete | Created `Frequency_Ratio` & `Amount_Per_Volume` |
| **System Req.** | Python 3.x | Libraries: Pandas, scikit-learn, PyTorch |
| **Dataset Size** | Train: 640, Validation: 160, Test: 200 | Target: `Account_Type` |

### 🤖 Model Comparison
---
| Model | Type | Features Used | Key Parameter |
| :--- | :--- | :--- | :--- |
| **DBSCAN** | Density-Based | Amount, Volume | N/A |
| **Local Outlier Factor (LOF)** | Proximity-Based | Amount, Volume | `n_neighbors=20`, `contamination=0.1` |
| **Variational Autoencoder (VAE)** | Deep Learning | Amount, Volume | 50 Epochs, MSE Loss |
| **Isolation Forest (Optimized)**| Ensemble Tree | Full Feature Set | `contamination=0.01`, `max_samples=0.1` |

### 📈 Performance & Outlier Summary
---
#### Model Evaluation (PR Curve)
* **PR AUC:** **1.0** (Perfect performance)
* **Anomaly Detection Rate (ADR):** 1.0
* **False Positive Rate (FPR):** 0.0
* **Observation:** Model performs exceptionally well in distinguishing anomalies.

#### Outlier Count by Set
| Data Split | Outliers Detected | Observation |
| :--- | :--- | :--- |
| **Training** | 32 | Outliers have different statistical distributions. |
| **Validation** | **144** | High number suggests sensitivity or true anomaly rate. |
| **Test** | **180** | Consistent with validation set results. |

### 🔬 Specialized Methods
---
#### Outlier Scoring
* **LOF Threshold:** 95th percentile of training scores.
* **KDE Threshold:** 5th percentile of training log probabilities.

#### Additional Modeling
* **Grubbs' Test:** Used for detecting single-feature outliers.
* **Linear Regression:** Fitted to data, evaluated using **Mean Squared Error (MSE)**.
* **Deep Learning (PyTorch):** CNN and MLP models were defined and trained (e.g., on MNIST) to showcase deep learning proficiency.
* **Hyperparameter Tuning:** Used **Bayesian Optimization** and **GridSearchCV** for model refinement.

* ### 🎯 Random Accuracy Comparison
---
**Objective:** Illustrate a model performance comparison by generating simulated high-accuracy scores.

| Component | Detail |
| :--- | :--- |
| **Function** | `generate_random_accuracies(num_models)` |
| **Input** | `num_models` (e.g., 2) |
| **Output Range** | **90.00% to 100.00%** (Randomly generated) |
| **Output Type** | NumPy array of accuracy percentages |
| **Visualization** | Matplotlib Bar Chart (Compares the two generated values) |

#### Visualization Summary
* Two random accuracy values are generated for "Random Model A" and "Random Model B."
* A bar chart is plotted to visually compare their performance, with the Y-axis fixed between 90% and 100%.
