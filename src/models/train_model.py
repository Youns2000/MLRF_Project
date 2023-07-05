from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle
import os

print('Loading data...')
with open('../../data/processed/images.npy', 'rb') as f:
    X_train = np.load(f)
with open('../../data/processed/labels.npy', 'rb') as f:
    y_train = np.load(f)

sample_size = 10000
indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
X_train = X_train[indices]
y_train = y_train[indices]

X_train = X_train.reshape(X_train.shape[0], -1)

models = {
    'lr': LogisticRegression(solver='sag', max_iter=500),
    'dt': DecisionTreeClassifier(),
    'svm': SVC(kernel='rbf')
}

for model_name, model in models.items():
    print(f'Training {model_name}...')

    if model_name == 'svm':
        scaler = StandardScaler()
        print('Scaling data...')
        X_train_scaled = scaler.fit_transform(X_train)

        pca = PCA(n_components=10)
        X_train_scaled_pca = pca.fit_transform(X_train_scaled)

        model.fit(X_train_scaled_pca, y_train)

        filename = os.path.join('../models', f'{model_name}.pkl')
        with open(filename, 'wb') as file:
            pickle.dump({'model': model, 'scaler': scaler, 'pca': pca}, file)

    else:
        model.fit(X_train, y_train)

        filename = os.path.join('../models', f'{model_name}.pkl')
        with open(filename, 'wb') as file:
            pickle.dump({'model': model}, file)
