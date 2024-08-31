import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(train_input_path, train_target_path, test_input_path, test_target_path):
    X_train = pd.read_csv(train_input_path)
    y_train = pd.read_csv(train_target_path)
    X_test = pd.read_csv(test_input_path)
    y_test = pd.read_csv(test_target_path)
    
    return X_train, y_train, X_test, y_test

def preprocess_data(X, y=None, train=True):
    imputer = SimpleImputer(strategy='mean')
    X.iloc[:, 1:] = pd.DataFrame(imputer.fit_transform(X.iloc[:, 1:]), columns=X.columns[1:])
    
    scaler = StandardScaler()
    X.iloc[:, 1:] = pd.DataFrame(scaler.fit_transform(X.iloc[:, 1:]), columns=X.columns[1:])
    
    if y is not None and train:
        label_encoder = LabelEncoder()
        y['target'] = label_encoder.fit_transform(y['target'])
    
    return X, y

def save_processed_data(X, y, X_path, y_path):
    X.to_csv(X_path, index=False)
    if y is not None:
        y.to_csv(y_path, index=False)

if __name__ == "__main__":
    train_input_path = 'data/raw/X_Train_Data_Input.csv'
    train_target_path = 'data/raw/Y_Train_Data_Target.csv'
    test_input_path = 'data/raw/X_Test_Data_Input.csv'
    test_target_path = 'data/raw/Y_Test_Data_Target.csv'
    
    X_train, y_train, X_test, y_test = load_data(train_input_path, train_target_path, test_input_path, test_target_path)
    
    X_train_processed, y_train_processed = preprocess_data(X_train, y_train)
    X_test_processed, _ = preprocess_data(X_test, train=False)
    
    save_processed_data(X_train_processed, y_train_processed, 'data/processed/train_processed.csv', 'data/processed/train_target.csv')
    save_processed_data(X_test_processed, None, 'data/processed/test_processed.csv', None)
