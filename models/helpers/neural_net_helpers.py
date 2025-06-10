import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score
)


def custom_objective_score(y_true, y_pred, damage_series):
    """
    Custom objective function:
    Score = -Sum of Damage from FN + (5 * TP) - (10 * FP)
    """
    # Ensure inputs are numpy arrays for robust indexing
    y_true = np.asarray(y_true)
    damage_series = np.asarray(damage_series)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    # Create a boolean mask for false negatives
    fn_mask = (y_true == 1) & (y_pred == 0)
    sum_damage_fn = damage_series[fn_mask].sum()

    # Calculate the final score based on your cost function
    score = -sum_damage_fn + (5 * TP) - (10 * FP)
    return float(score)

def evaluate_and_find_best_threshold(model, x_test, y_test, damage_test):
    """
    Evaluates the model and efficiently finds the optimal classification threshold
    by only testing the unique predicted values, plus a case for classifying all as normal.

    Args:
        model: A model object with a .predict() method.
        x_test: The test features.
        y_test: The true labels for the test set.
        damage_test: The damage values associated with the test set.

    Returns:
        A tuple containing the best score and the corresponding best threshold.
    """
    print("Generating predictions from the model...")
    y_pred_values = model.predict(x_test).flatten()

    candidate_thresholds = np.unique(y_pred_values)

    # <<< FIX: Handle the edge case of no predictions >>>
    # If the model predicts nothing, we can't proceed.
    if len(candidate_thresholds) == 0:
        return -np.inf, None

    # <<< FIX: Add a threshold that guarantees all predictions are classified as 0 (Normal) >>>
    # This value is simply 1 greater than the highest predicted score.
    # This ensures we always test the "predict everything as Normal" scenario.
    super_threshold = candidate_thresholds.max() + 1
    all_thresholds_to_test = np.append(candidate_thresholds, super_threshold)

    print(f"Identified {len(all_thresholds_to_test)} unique thresholds to test.")

    best_score = -np.inf
    best_threshold = None

    # Iterate through each unique prediction value as a potential threshold
    for threshold in all_thresholds_to_test:
        # The classification rule remains the same: score >= threshold is positive class
        y_pred_class = (y_pred_values >= threshold).astype(int)

        score = custom_objective_score(y_test, y_pred_class, damage_test)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_score, best_threshold



def generate_model_diagnostics(models, x_test, y_test, damage_test, show_plots=True):
    """
    Generates a diagnostic report for each model. Can optionally show plots.

    Args:
        models (list of dicts): List of models to evaluate.
        x_test: The test features.
        y_test: The true binary labels.
        damage_test: The true damage values.
        show_plots (bool): If True, all visualizations will be displayed. 
                           If False, only text-based metrics will be printed.
    """
    print("--- Generating Comprehensive Model Diagnostics ---\n")
    
    for model_item in models:
        model_name = model_item['name']
        model_object = model_item['model']
        
        print(f"==================== DIAGNOSTICS FOR: {model_name} ====================")

        # --- Key Metric Calculation ---
        y_pred_values = model_object.predict(x_test).flatten()
        best_score, best_threshold = evaluate_and_find_best_threshold(
            model_object, x_test, y_test, damage_test
        )
        y_pred_class = (y_pred_values >= best_threshold).astype(int)

        # --- ALWAYS PRINT: High-Level Score Summary ---
        print(f"\n--- Summary ---")
        print(f"  \U0001F3C6 Best Custom Score: {best_score:.2f}")
        print(f"  \U0001F5AF Optimal Threshold: {best_threshold:.4f}")
        print("---------------")

        # --- 1. Performance at Optimal Threshold ---
        if show_plots:
            print(f"\n\n--- 1. Performance at Optimal Threshold ---\n")
        
        cm = confusion_matrix(y_test, y_pred_class, labels=[0, 1])
        
        if show_plots:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Predicted Normal (0)', 'Predicted Damage (1)'],
                        yticklabels=['Actual Normal (0)', 'Actual Damage (1)'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix for {model_name}\n(at Threshold = {best_threshold:.4f})', fontsize=14)
            plt.show()

        if show_plots:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_class, target_names=['No Damage (0)', 'Damage (1)'], zero_division=0))


        # --- 2. Overall Classifier Performance (Threshold-Agnostic) ---
        if show_plots:
            print("\n\n--- 2. Overall Classifier Performance ---\n")
        
        roc_auc = roc_auc_score(y_test, y_pred_values)
        pr_auc = average_precision_score(y_test, y_pred_values)
        
        if show_plots:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_values)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title(f'ROC Curve for {model_name}', fontsize=14)
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_values)
            no_skill = len(y_test[y_test==1]) / len(y_test)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
            plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label=f'No Skill Baseline ({no_skill:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve for {model_name}', fontsize=14)
            plt.legend(loc="best")
            plt.grid(True)
            plt.show()

        print(f"AUC-ROC Score: {roc_auc:.4f}")
        print(f"AUC-PR Score (Average Precision): {pr_auc:.4f}")
        
        
        # --- 3. Damage Prediction Visualization ---
        if show_plots:
            print("\n\n--- 3. Damage Prediction Visualization ---\n")
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(10, 7))
            plot_df = pd.DataFrame({
                'Actual Damage': damage_test.flatten(),
                'Predicted Damage Score': y_pred_values,
                'Is Damage': y_test.astype(bool).flatten()
            })
            sns.scatterplot(
                data=plot_df, x='Actual Damage', y='Predicted Damage Score',
                hue='Is Damage', palette={True: 'red', False: 'blue'}, alpha=0.6, ax=ax
            )
            max_val = max(plot_df['Actual Damage'].max(), plot_df['Predicted Damage Score'].max())
            ax.plot([0, max_val], [0, max_val], 'k--', label='Perfect Prediction (y=x)')
            ax.set_title(f'Actual vs. Predicted Damage Score for: {model_name}', fontsize=16, pad=20)
            ax.set_xlabel('Actual Damage', fontsize=12)
            ax.set_ylabel('Predicted Damage Score', fontsize=12)
            ax.legend()
            plt.show()
        
        print(f"==================== END DIAGNOSTICS FOR: {model_name} ====================\n\n")
