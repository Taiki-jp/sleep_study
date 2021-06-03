from tensorflow.keras.layers import *
from keras import backend as K
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

# ref : https://www.kaggle.com/kabirnagpal/xception-resnet-learn-how-to-stack
def recall_m(y_true, y_pred):

    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), 0.5),K.floatx())
    true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.clip(y_true, 0, 1))
    recall_ratio = true_positives / (possible_positives + K.epsilon())
    return recall_ratio

def precision_m(y_true, y_pred):

    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), 0.5), K.floatx())
    true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(y_pred)
    precision_ratio = true_positives / (predicted_positives + K.epsilon())
    return precision_ratio

def f1_m(y_true, y_pred):
    
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# pca
def make_pca(data=None, pse_data=True, path=None):
    if pse_data:
        digits = datasets.load_digits()
        X_reduced = PCA(n_components=2).fit_transform(digits.data)
        plt.scatter(X_reduced[:,0], X_reduced[:,1], c=digits.target)
        plt.colorbar()
        plt.show()
    else:
        print("TODO : 実装してね")

# t-sne

if __name__ == "__main__":
    make_pca()