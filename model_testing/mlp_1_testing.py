#!/usr/bin/env C:/Users/ryanm/miniconda3/envs/tf/python

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns


# Disable GPU: Issues currently with GPU memory growth
tf.config.set_visible_devices([], 'GPU')

def load_datasets(train_path, test_path):
    print('Loading the training and testing datasets')
    df_training = pd.read_csv(train_path)
    X_train = df_training.values[:, 0]
    y_train = df_training.values[:, 1].astype(np.float32)

    df_testing = pd.read_csv(test_path)  
    X_test = df_testing.values[:, 0]
    y_test = df_testing.values[:, 1].astype(np.float32)
    
    return X_train, y_train, X_test, y_test

def load_embeddings(embeddings_path):
    print('Loading the GLoVe word embeddings')
    return KeyedVectors.load_word2vec_format(embeddings_path, binary=False)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def preprocess_data(X_train, X_test):
    print('Preprocessing the training and testing data')
    X_train = [preprocess_text(sentence) for sentence in X_train]
    X_test = [preprocess_text(sentence) for sentence in X_test]

    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    max_len = max([len(x) for x in X_train]) 
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    return X_train, X_test, tokenizer.word_index

def embed_text_glove(token_sequences, word_index, glove):
    embedded_text = []
    for sequence in token_sequences:
        embedded_sentence = []
        for word_idx in sequence:  # Rename the loop variable to word_idx
            try:
                if (word_idx == 0):
                    embedded_sentence.append(np.zeros(300))
                    continue
                word = next(word for word, index in word_index.items() if index == word_idx)  # Rename word_index to word_idx
                embedded_word = glove[word]
                embedded_sentence.append(embedded_word)
            except KeyError:
                embedded_sentence.append(np.zeros(300))
        embedded_text.append(embedded_sentence)
    return np.array(embedded_text)

def train_model(X_train, y_train, batch_size, epochs, n_folds, model_params):
    # Flatten the embedded dimensions
    X_train = X_train.reshape(X_train.shape[0], -1)
    kf = KFold(n_splits=n_folds, shuffle=True)
    total_training_time = 0 
    all_training_accuracies, all_validation_accuracies = [], []
    all_training_losses, all_validation_losses = [], []

    # Evaluate initial accuracy and loss
    # initial_loss, initial_accuracy = evaluate_model(X_train, y_train, model_params)

    # print("Initial Loss:", initial_loss)
    # print("Initial Accuracy:", initial_accuracy)

    for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train)):
        print(f"Fold {fold_idx + 1}/{n_folds}")
       
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # evaluate the model pre-training for better analysis
        print ('Evaluating the initial model before training')
        initial_training_loss, initial_training_accuracy = evaluate_model(X_train_fold, y_train_fold, model_params)
        initial_validation_loss, initial_validation_accuracy = evaluate_model(X_val_fold, y_val_fold, model_params)
        # all_training_accuracies.append(initial_training_accuracy)
        # all_validation_accuracies.append(initial_validation_accuracy)
        # all_training_losses.append(initial_training_loss)
        # all_validation_losses.append(initial_validation_loss)
        print (initial_training_accuracy, initial_validation_accuracy, initial_training_loss, initial_validation_loss)

        mlp = Sequential()
        mlp.add(Dense(**model_params['hl1'])) # hidden layer
        mlp.add(Dense(**model_params['output'])) 

        mlp.compile(**model_params['compile'])

        start_time = time.time()
        history = mlp.fit(X_train_fold, y_train_fold, batch_size=batch_size, epochs=epochs, validation_data=(X_val_fold, y_val_fold), verbose=0)
        end_time = time.time()
        total_training_time += end_time - start_time

        all_training_accuracies.append((initial_training_accuracy, *history.history['accuracy']))
        all_validation_accuracies.append((initial_validation_accuracy, *history.history['val_accuracy']))
        all_training_losses.append((initial_training_loss, *history.history['loss']))
        all_validation_losses.append((initial_validation_loss, *history.history['val_loss']))

        print(f"Fold {fold_idx + 1}/{n_folds} - Training Accuracys: {all_training_accuracies[-1]}, Validation Accuracys: {all_validation_accuracies[-1]}")
    print(f"Average training time: {total_training_time / n_folds:.2f} seconds")

    return mlp, np.array(all_training_accuracies), np.array(all_validation_accuracies), np.array(all_training_losses), np.array(all_validation_losses), total_training_time

def evaluate_model(X, y, model_params):
    model = Sequential()
    model.add(Dense(**model_params['hl1'])) # hidden layer
    model.add(Dense(**model_params['output'])) 

    model.compile(**model_params['compile'])

    # Evaluate the model on validation data
    loss, accuracy = model.evaluate(X, y)

    return loss, accuracy

def plot_metrics(plot_title, train_metrics_list, val_metrics_list, metric_names):
    fig, ax = plt.subplots()  
    line_styles = ['-', '--', ':']  # Different line styles for losses and accuracies
    colors = ['blue', 'orange', 'green']  # Different colors for losses and accuracies

    for i, (train_metric, val_metric, metric_name) in enumerate(zip(train_metrics_list, val_metrics_list, metric_names)):
        # Select line style and color based on the index
        line_style = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]

        ax.plot(train_metric, label='Training ' + metric_name, linestyle=line_style, color=color)
        ax.plot(val_metric, label='Validation ' + metric_name, linestyle=line_style, color=color, marker='o')

    ax.set_xlabel('Epochs')
    ax.set_title(plot_title)
    ax.legend()
    return fig, ax 

def test_model_and_get_metrics(mlp, X_test, y_test):
    print ('Testing the model')
    X_test = X_test.reshape(X_test.shape[0], -1)
    mlp_preds = mlp.predict(X_test).round()
    cm = confusion_matrix(y_test, mlp_preds)

    # Calculate the accuracy, precision, recall and F1 score
    accuracy = accuracy_score(y_test, mlp_preds.round())
    precision = precision_score(y_test, mlp_preds.round())
    recall = recall_score(y_test, mlp_preds.round())
    f1 = f1_score(y_test, mlp_preds.round())

    return cm, accuracy, precision, recall, f1

if __name__ == "__main__":
    train_path = 'C:\\Users\\ryanm\\OneDrive\\Desktop\\deep learning\\assignments\\Deep-Learning-Assignment-2\\train.csv'
    test_path = 'C:\\Users\\ryanm\\OneDrive\\Desktop\\deep learning\\assignments\\Deep-Learning-Assignment-2\\test.csv'
    embeddings_path = 'C:\\Users\\ryanm\\OneDrive\\Desktop\\deep learning\\assignments\\Deep-Learning-Assignment-2\\glove.6B.300d.txt.word2vec'

    X_train, y_train, X_test, y_test = load_datasets(train_path, test_path)
    glove = load_embeddings(embeddings_path)
    X_train, X_test, word_index = preprocess_data(X_train, X_test)
    X_train_glove_embedded = embed_text_glove(X_train, word_index, glove)
    X_test_glove_embedded = embed_text_glove(X_test, word_index, glove)

    model_params = {
        'hl1': {'units': 5, 'activation': 'relu', 'input_shape': (X_train_glove_embedded.shape[1] * 300,)},
        'output': {'units': 1, 'activation': 'sigmoid'},
        'compile': {'optimizer': tf.keras.optimizers.Adam(learning_rate=0.05), 'loss': 'binary_crossentropy', 'metrics': ['accuracy']}
    }

    mlp, training_accuracies, validation_accuracies, training_losses, validation_losses, total_training_time = train_model(
        X_train_glove_embedded, y_train, batch_size=64, epochs=2, n_folds=10, model_params=model_params)

    print ('Training average accuracies', str(training_accuracies.mean(axis=0)))
    print ('Training average losses', str(training_losses.mean(axis=0)))
    print ('Validation average accuracies:', str(validation_accuracies.mean(axis=0)))
    print ('Validation average losses', str(validation_losses.mean(axis=0)))

    plot_title = 'Model Metrics Over Epochs for Baseline MLP Using Single Vector of Pre-Trained Word Embeddings'
    train_metrics_list = [training_losses.mean(axis=0), training_accuracies.mean(axis=0)]
    val_metrics_list = [validation_losses.mean(axis=0), validation_accuracies.mean(axis=0)]
    metric_names = ['Loss', 'Accuracy']

    plot_metrics(plot_title, train_metrics_list, val_metrics_list, metric_names)
    plt.show()

    cm, accuracy, precision, recall, f1 = test_model_and_get_metrics(mlp=mlp, X_test=X_test_glove_embedded, y_test=y_test)
