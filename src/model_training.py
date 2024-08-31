import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib

def load_engineered_data(train_path, target_path):
    X = pd.read_csv(train_path)
    y = pd.read_csv(target_path)
    
    if len(X) != len(y):
        print(f"Warning: The number of samples in X ({len(X)}) and y ({len(y)}) do not match.")
        
        min_rows = min(len(X), len(y))
        X = X.iloc[:min_rows, :]
        y = y.iloc[:min_rows, :]
        print(f"Datasets aligned by index. New shape: X = {X.shape}, y = {y.shape}")

    return X, y

def train_model(X_train, y_train, model, params):
    y_train = y_train['target'].astype(int).values.ravel()
    
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_score_

def save_model(model, path):
    joblib.dump(model, path)

if __name__ == "__main__":
    train_engineered_path = 'data/processed/train_engineered.csv'
    target_path = 'data/processed/train_target.csv'
    
    X, y = load_engineered_data(train_engineered_path, target_path)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'RandomForest': (RandomForestClassifier(), {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
        'XGBoost': (xgb.XGBClassifier(), {'n_estimators': [100, 200], 'max_depth': [3, 6]}),
        'LightGBM': (lgb.LGBMClassifier(), {'n_estimators': [100, 200], 'max_depth': [3, 6]}),
        'CatBoost': (cb.CatBoostClassifier(verbose=0), {'iterations': [100, 200], 'depth': [3, 6]})
    }
    
    best_model = None
    best_accuracy = 0
    
    for name, (model, params) in models.items():
        print(f"Training {name}...")
        trained_model, accuracy = train_model(X_train, y_train, model, params)
        print(f"{name} Accuracy: {accuracy}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_model
    
    save_model(best_model, 'models/best_model.pkl')
    print(f"Best model saved with accuracy: {best_accuracy}")
