import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class my_svm():
    # __init__() function should initialize all your variables
    def __init__(self):
        # Initialize necessary variables such as feature sets, models, and scaler

        self.scaler = StandardScaler()  # Used for feature normalization
        self.model = SVC(kernel='rbf', C=1.65, gamma='scale')  # Initialize SVM model
        self.features = None  
        self.labels = None  


    # preprocess() function:
    #  1) normalizes the data, 
    #  2) removes missing values
    #  3) assigns labels to target 
    def preprocess(self, base_path):
        # Load datasets 
        neg_features_main_timechange = np.load(f'{base_path}/neg_features_main_timechange.npy')
        neg_features_historical = np.load(f'{base_path}/neg_features_historical.npy')
        neg_features_maxmin = np.load(f'{base_path}/neg_features_maxmin.npy')
        pos_features_main_timechange = np.load(f'{base_path}/pos_features_main_timechange.npy')
        pos_features_historical = np.load(f'{base_path}/pos_features_historical.npy')
        pos_features_maxmin = np.load(f'{base_path}/pos_features_maxmin.npy')


        # Negative features
        neg_features = pd.DataFrame(neg_features_main_timechange).iloc[:, :90]
        neg_features = pd.concat([neg_features, pd.DataFrame(neg_features_historical, columns=['historical_feature']), pd.DataFrame(neg_features_maxmin)], axis=1)
        neg_features['label'] = 0

        # Positive features
        pos_features = pd.DataFrame(pos_features_main_timechange).iloc[:, :90]
        pos_features = pd.concat([pos_features, pd.DataFrame(pos_features_historical, columns=['historical_feature']), pd.DataFrame(pos_features_maxmin)], axis=1)
        pos_features['label'] = 1

        # Combine datasets
        combined_features = pd.concat([neg_features, pos_features], axis=0)

        # Separate 
        features = combined_features.drop(columns=['label']).astype(str)
        labels = combined_features['label'].astype(int)

        # Convert all column names to strings 
        features.columns = features.columns.astype(str)

        # Normalize features
        scaler = StandardScaler()
        normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)


        # Add the integer labels back to the normalized DataFrame
        normalized_combined_data = pd.concat([normalized_features, labels.reset_index(drop=True)], axis=1)

        # Store results
        self.features = normalized_features
        self.labels = labels.reset_index(drop=True)

        # Set display options for better readability
        pd.set_option('display.max_rows', None)  # Display all rows
        pd.set_option('display.max_columns', None)  # Display all columns
        pd.set_option('display.width', 10000000)  
        pd.set_option('display.expand_frame_repr', False) 
        
        #print(normalized_combined_data.iloc[:, :20])  # Print first 20 columns
        # Print the normalized DataFrame
        #print(normalized_combined_data)  # Print entire DataFrame

        #columncount = 0
        #for col in normalized_combined_data.columns:
            #columncount += 1
            #prints each column with a number and a divider '|' 
            #print(columncount," ",col ,end =" | ")
        
        # 1. Combine positive and negative data (features)
        # 2. Normalize the data using StandardScaler
        # 3. Handle any missing values if present (e.g., via imputation or removal)
        # 4. Assign labels (1 for positive events, 0 for negative)
        # 5. Return processed features and labels
    
    # feature_creation() function takes as input the feature set label (e.g. FS-I, FS-II, FS-III, FS-IV)
    # and creates a 2D array of corresponding features for both positive and negative observations.
    def feature_creation(self, fs_value):
        # 1. Depending on the fs_value input (e.g., FS-I, FS-II, etc.), load the corresponding features
        # 2. Concatenate features for both positive and negative events into a 2D array
        # 3. Return the created feature set for model input

        # Initialize an empty list 
        selected_features = pd.DataFrame()
        
        # 2. Depending on fs_value input select the  feature columns
        if 'FS-I' in fs_value:
            fs1_features = self.features.iloc[:, :18]
            selected_features = pd.concat([selected_features, fs1_features], axis=1)

        # FS-II: 
        if 'FS-II' in fs_value:
            fs2_features = self.features.iloc[:, 18:90]
            selected_features = pd.concat([selected_features, fs2_features], axis=1)

        # FS-III: 
        if 'FS-III' in fs_value:
            fs3_features = self.features[['historical_feature']]  #concatenated historical feature column
            selected_features = pd.concat([selected_features, fs3_features], axis=1)

        # FS-IV:
        if 'FS-IV' in fs_value:
            fs4_features = self.features.iloc[:, 90:108]  
            selected_features = pd.concat([selected_features, fs4_features], axis=1)
        
        labels = self.labels
        return selected_features, labels

    
    # cross_validation() function splits the data into train and test splits,
    # Use k-fold with k=10
    # The SVM is trained on the training set and tested on the test set
    # The output is the average accuracy across all train-test splits.
    def cross_validation(self, X, y):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        tss_scores = []
        predictions = []  # Store predictions for each fold

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            self.training(X_train, y_train)
            y_pred = self.model.predict(X_test)
            predictions.append((y_test, y_pred))  # Store true and predicted 

            tss_score = self.tss(y_test, y_pred)
            tss_scores.append(tss_score)

        mean_tss = np.mean(tss_scores)
        std_tss = np.std(tss_scores)

        # Visualize the k-Fold TSS Scores
        self.plot_kfold_tss(tss_scores)

        # Visualize all confusion matrices
        self.plot_all_confusion_matrices(predictions)

        return mean_tss, std_tss


    def training(self, X_train, y_train):
        # 1. Fit the SVM model on the training data (X_train, y_train)
        # 2. Return the trained model
        self.model.fit(X_train, y_train)
        pass
    
    # tss() function computes the accuracy of predicted outputs (i.e., target prediction on test set)

    def tss(self, y_true, y_pred):
        # 1. Use the formula for TSS:
        #    TSS = (TP / (TP + FN)) - (FP / (FP + TN))
        # 2. Calculate TP, FP, TN, FN using a confusion matrix
        # 3. Return the TSS score

        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        #print ("c m ", confusion_matrix(y_true, y_pred).ravel())
    
        # TSS formula: TSS = (TP / (TP + FN)) - (FP / (FP + TN))
        tss_value = (tp / (tp + fn)) - (fp / (fp + tn))
    
        return tss_value
 
    def plot_kfold_tss(self, tss_scores):
        plt.plot(range(1, len(tss_scores)+1), tss_scores, marker='o', label='TSS Score per fold')
        plt.xlabel('Fold Number')
        plt.ylabel('TSS Score')
        plt.title('k-Fold Cross-Validation TSS Scores')
        plt.legend()
        plt.show()

    def plot_all_confusion_matrices(self, predictions):
        fig, axs = plt.subplots(2, 5, figsize=(13, 8))  
        axs = axs.flatten() 

        for i, (y_true, y_pred) in enumerate(predictions):
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Actual Neg", "Actual Pos"])
            disp.plot(ax=axs[i], cmap=plt.cm.Blues, colorbar=False)


            axs[i].set_title(f'Fold {i+1}', pad=20, fontsize=8)

            # Set y-axis label to Actual Pos and Actual Neg
            axs[i].set_ylabel('Actual', fontsize=8)
            axs[i].set_yticks([0, 1])
            axs[i].set_yticklabels(["Actual Neg", "Actual Pos"])

            # Set x-axis label to Pred Pos and Pred Neg
            axs[i].set_xlabel('Predicted', fontsize=8)
            axs[i].set_xticks([0, 1])
            axs[i].set_xticklabels(["Pred Neg", "Pred Pos"])

            axs[i].grid(False)

            axs[i].set_ylim(-0.5, 1.5)
            axs[i].set_xlim(-0.5, 1.5)

        # Increase spacing between subplots
        plt.subplots_adjust(wspace=0.6, hspace=0.4) 

        plt.tight_layout()
        
        plt.show()
    

# feature_experiment() function executes experiments with all 4 feature sets.
def feature_experiment(base_path):
    # 1. Instantiate the my_svm class
    svm_instance = my_svm()
    svm_instance.preprocess(base_path)
    
    # 2. For each feature set combination (e.g., FS-I, FS-II, etc.), do the following:
    #    a. Load the dataset
    #    b. Preprocess the data
    #    c. Perform cross-validation
    #    d. Store the TSS scores and confusion matrices
    #    e. Plot the results (TSS score for each fold)
    #    f. Identify the best-performing feature set
    
    # 3. Print the best feature set combination and associated performance metrics
    
    # Define the feature set combinations
    feature_sets = [['FS-I'], ['FS-II'], ['FS-III'], ['FS-IV'], ['FS-I', 'FS-II'], ['FS-I', 'FS-III'],
                    ['FS-I', 'FS-IV'], ['FS-II', 'FS-III'], ['FS-II', 'FS-IV'], ['FS-III', 'FS-IV'],
                    ['FS-I', 'FS-II', 'FS-III'], ['FS-I', 'FS-II', 'FS-IV'], 
                    ['FS-I', 'FS-III', 'FS-IV'], ['FS-II', 'FS-III', 'FS-IV'], ['FS-I', 'FS-II', 'FS-III', 'FS-IV']]
    
    results = {}

    for fs in feature_sets:
        X, y = svm_instance.feature_creation(fs)
        print("\nCross-validation for feature set combination: ", fs)
        mean_tss, std_tss = svm_instance.cross_validation(X, y)
        print(f"Mean TSS = {mean_tss}, Std TSS = {std_tss}")
        results[str(fs)] = (mean_tss, std_tss)
    
    # Find the best performing feature set combination
    best_feature_set = max(results, key=lambda x: results[x][0])
    print("\nBest feature set: ", best_feature_set, "with TSS: ", results[best_feature_set][0])
    
    return best_feature_set, results[best_feature_set][0]  # Return best feature set and its mean TSS

# svm is trained (and tested) on both 2010 data and 2020 data
def data_experiment():
    datasets = [
        ('data-2010-15', '2010 dataset'),
        ('data-2020-24', '2020 dataset')
    ]

    best_feature_set, _ = feature_experiment(datasets[0][0])  # Get best feature set from 2010 dataset

    for base_path, label in datasets:
        print(f"Evaluating on {label} with best feature set: {best_feature_set}")
        svm_instance = my_svm()
        svm_instance.preprocess(base_path)
        X, y = svm_instance.feature_creation(eval(best_feature_set))
        mean_tss, std_tss = svm_instance.cross_validation(X, y)
        print(f"\n{label} - Mean TSS = {mean_tss}, Std TSS = {std_tss}")
        # # Get class distribution
        print ("\ndistribution: ", svm_instance.labels.value_counts())    



#feature_experiment()

# Run experiments with both datasets (2010-2015 and 2020-2024)

data_experiment()


