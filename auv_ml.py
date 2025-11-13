# auv_ml_analysis.py
# This script loads the AUV simulation data, trains machine learning models
# (SVM, Random Forest, Decision Tree) to classify an "ideal" AUV state,
# and evaluates their performance with metrics and plots.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os

def run_ml_analysis():
    """
    Main function to load data, train models, and show results.
    """
    # --- 1. Load and Prepare Data ---
    
    csv_file = 'osar_log.csv'

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find the data file '{csv_file}'.")
        print("Please make sure the log file from your simulation is in the same directory and named correctly.")
        return

    # --- 2. Feature Engineering ---

    # Create 'Node_Density' feature
    df['Node_Density'] = df['Covered_Nodes_in_Radius'].str.split(',').str.len().fillna(0)

    # Redefine the target variable 'Is_Ideal' using a weighted score
    
    # Normalize Speed and Node_Density to be on a 0-1 scale
    scaler = MinMaxScaler()
    df[['Speed_Scaled', 'Density_Scaled']] = scaler.fit_transform(df[['Speed_m/s', 'Node_Density']])
    
    # Calculate a composite 'AUV_Score'. We can weigh node density higher.
    speed_weight = 0.4
    density_weight = 0.6
    df['AUV_Score'] = (df['Speed_Scaled'] * speed_weight) + (df['Density_Scaled'] * density_weight)
    
    # --- NEW: Redefine 'Is_Ideal' based on the best AUV per timestamp ---
    # This simulates a control center choosing the single best AUV at any given moment,
    # creating a more complex and realistic classification problem.
    
    # Identify the index of the row with the maximum AUV_Score for each timestamp.
    best_auv_indices = df.loc[df.groupby('Timestamp')['AUV_Score'].idxmax()]

    # Create the 'Is_Ideal' column, defaulting to 0 (not ideal).
    df['Is_Ideal'] = 0

    # Set 'Is_Ideal' to 1 only for the rows corresponding to the best AUV at each timestamp.
    df.loc[best_auv_indices.index, 'Is_Ideal'] = 1


    print("--- Data Head ---")
    print(df[['Timestamp', 'AUV_ID', 'AUV_Score', 'Is_Ideal']].head(10))
    print("\n--- Ideal vs. Not Ideal Counts ---")
    print(df['Is_Ideal'].value_counts())
    
    # Define features (X) and target (y)
    features = ['Speed_m/s', 'Position_X', 'Position_Y', 'Position_Z', 'Node_Density']
    X = df[features]
    y = df['Is_Ideal']

    # --- 3. Split Data and Scale Features ---
    # Stratify ensures the train/test split has a similar proportion of ideal/not ideal samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features for SVM
    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train)
    X_test_scaled = std_scaler.transform(X_test)

    # --- 4. Initialize and Train Models ---
    # --- CHANGE: Add stricter hyperparameters (max_depth, min_samples_leaf) to prevent overfitting ---
    models = {
        "SVM": SVC(kernel='rbf', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42)
    }

    results = {}
    
    # Save text results to a file
    report_filename = "model_evaluation_report.txt"
    with open(report_filename, "w") as report_file:
        report_file.write("--- Model Training and Evaluation Report ---\n")
        print("\n--- Model Training and Evaluation ---")
        for name, model in models.items():
            print(f"\n--- {name} ---")
            report_file.write(f"\n\n--- {name} ---\n")
            
            if name == "SVM":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {'model': model, 'accuracy': accuracy, 'report': report, 'cm': cm, 'predictions': y_pred}
            
            accuracy_str = f"Accuracy: {accuracy:.4f}"
            print(accuracy_str)
            report_file.write(accuracy_str + "\n")
            
            print("Classification Report:")
            print(report)
            report_file.write("Classification Report:\n")
            report_file.write(report + "\n")
    
    print(f"\nEvaluation metrics saved to '{report_filename}'")

    # --- 5. Visualization (with saving) ---
    
    # Bar plot for model comparison
    accuracies = {name: res['accuracy'] for name, res in results.items()}
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.savefig('model_accuracy_comparison.png')
    plt.show()
    print("Saved 'model_accuracy_comparison.png'")

    # Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for i, (name, res) in enumerate(results.items()):
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()
    print("Saved 'confusion_matrices.png'")
    
    # Feature Importance Plot for Random Forest
    rf_model = results['Random Forest']['model']
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance (Random Forest)', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig('feature_importance.png')
    plt.show()
    print("Saved 'feature_importance.png'")

    # 3D Scatter plot of predictions
    fig = plt.figure(figsize=(20, 15))
    all_predictions = {'Test Data (True Labels)': y_test, **{name: res['predictions'] for name, res in results.items()}}

    for i, (title, preds) in enumerate(all_predictions.items()):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        scatter = ax.scatter(X_test['Position_X'], X_test['Position_Y'], X_test['Position_Z'], 
                             c=preds, cmap='coolwarm', s=20, alpha=0.7)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Position X')
        ax.set_ylabel('Position Y')
        ax.set_zlabel('Position Z')
        legend1 = ax.legend(*scatter.legend_elements(), title="Is Ideal?")
        ax.add_artist(legend1)

    plt.suptitle('3D Visualization of AUV Positions and Ideal State Predictions', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('3d_predictions.png')
    plt.show()
    print("Saved '3d_predictions.png'")

if __name__ == '__main__':
    run_ml_analysis()

