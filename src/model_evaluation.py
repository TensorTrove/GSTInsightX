import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def load_data(test_path, target_path):
    X_test = pd.read_csv(test_path)
    y_test = pd.read_csv(target_path)
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, cm

def plot_confusion_matrix(cm, output_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    test_engineered_path = 'data/processed/test_engineered.csv'
    target_path = 'data/raw/Y_Test_Data_Target.csv'
    model_path = 'models/best_model.pkl'
    
    X_test, y_test = load_data(test_engineered_path, target_path)
    
    model = joblib.load(model_path)
    
    accuracy, report, cm = evaluate_model(model, X_test, y_test['target'])
    
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\\n{report}")
    
    plot_confusion_matrix(cm, 'reports/figures/confusion_matrix.png')