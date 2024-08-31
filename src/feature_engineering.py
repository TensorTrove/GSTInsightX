import pandas as pd
from sklearn.decomposition import PCA

def load_processed_data(train_path, test_path):
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    return X_train, X_test

def create_features(X):
    pca = PCA(n_components=5)
    principal_components = pca.fit_transform(X)
    
    principal_df = pd.DataFrame(data=principal_components, columns=[f'PCA_{i+1}' for i in range(5)])
    X = pd.concat([X, principal_df], axis=1)
    
    return X

def save_engineered_data(X, path):
    X.to_csv(path, index=False)

if __name__ == "__main__":
    train_processed_path = 'data/processed/train_processed.csv'
    test_processed_path = 'data/processed/test_processed.csv'
    
    X_train, X_test = load_processed_data(train_processed_path, test_processed_path)
    
    X_train_engineered = create_features(X_train)
    X_test_engineered = create_features(X_test)
    
    save_engineered_data(X_train_engineered, 'data/processed/train_engineered.csv')
    save_engineered_data(X_test_engineered, 'data/processed/test_engineered.csv')