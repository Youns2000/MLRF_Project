from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle
import os


def predict():
    models = {'lr': None, 'dt': None, 'svm': None}

    print('Loading data...')
    with open(os.path.join('../../data/processed/test_images.npy'), 'rb') as f:
        X_test = np.load(f)
    with open(os.path.join('../../data/processed/test_labels.npy'), 'rb') as f:
        y_test = np.load(f)

    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test_used = None

    for model_name in models.keys():
        print(f'Predicting with {model_name}...')
        with open(os.path.join('../../models', f'{model_name}.pkl'), 'rb') as f:
            saved_model = pickle.load(f)

        model = saved_model['model']

        if model_name == 'svm':
            scaler = saved_model['scaler']
            pca = saved_model['pca']

            print('Scaling data...')
            X_test_scaled = scaler.transform(X_test)

            print('Applying PCA...')
            X_test_scaled_pca = pca.transform(X_test_scaled)

            print('Making predictions...')
            y_pred = model.predict(X_test_scaled_pca)

            X_test_used = X_test_scaled_pca

        else:
            print('Making predictions...')
            y_pred = model.predict(X_test)

            X_test_used = X_test
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy of {model_name}: {accuracy * 100}%')

    return y_pred, y_test, X_test_used, model


if __name__ == "__main__":
    predict()
