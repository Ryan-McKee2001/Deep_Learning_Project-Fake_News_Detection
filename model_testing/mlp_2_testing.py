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
from tensorflow.keras import regularizers
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import os
from tensorflow.keras.layers import Embedding


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

def preprocess_data(X_train, X_test, preprocess=True):
    print('Preprocessing the training and testing data')
    if preprocess:
        X_train = [preprocess_text(sentence) for sentence in X_train]
        X_test = [preprocess_text(sentence) for sentence in X_test]

    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    max_len = max([len(x) for x in X_train]) 
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    return X_train, X_test, tokenizer

def train_model(X_train, y_train, batch_size, epochs, n_folds, model_params, tokenizer):
    # Flatten the embedded dimensions

    vocab_size = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    embedding_matrix = create_embedding_matrix(tokenizer, glove, vocab_size)

    model_params['hl1']['input_shape'] = vocab_size

    kf = KFold(n_splits=n_folds, shuffle=True)
    total_training_time = 0 
    all_training_accuracies, all_validation_accuracies = [], []
    all_training_losses, all_validation_losses = [], []

    for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train)):
        print(f"Fold {fold_idx + 1}/{n_folds}")
       
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # evaluate the model pre-training for better analysis
        print ('Evaluating the initial model before training')
        initial_training_loss, initial_training_accuracy = evaluate_model(X_train_fold, y_train_fold, model_params)
        initial_validation_loss, initial_validation_accuracy = evaluate_model(X_val_fold, y_val_fold, model_params)

        mlp = Sequential()
        mlp.add(Embedding(input_dim=vocab_size, output_dim=300, input_length=X_train.shape[1], trainable=False, weights=))
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
    average_training_time = total_training_time / n_folds
    return mlp, np.array(all_training_accuracies), np.array(all_validation_accuracies), np.array(all_training_losses), np.array(all_validation_losses), average_training_time

def create_embedding_matrix(tokenizer, glove, num_words):
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in tokenizer.word_index.items():
        if word in glove.key_to_index:
            embedding_vector = glove[word]
            embedding_matrix[i] = embedding_vector

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

def test_model_configurations(X_train, y_train, X_test, y_test, preprocess, model_params, batch_size, epochs, n_folds):
    
    X_train, X_test, tokenizer = preprocess_data(X_train, X_test, preprocess)

    # model_params['hl1']['input_shape'] = (X_train.shape[1] * 300,)

    mlp, training_accuracies, validation_accuracies, training_losses, validation_losses, total_training_time = train_model(
        X_train, y_train, batch_size=batch_size, epochs=epochs, n_folds=n_folds, model_params=model_params, tokenizer=tokenizer)

    plot_title = 'MLP Using Flattened Glove Embeddings \nTraining And Validation Metrics'
    train_metrics_list = [training_losses.mean(axis=0), training_accuracies.mean(axis=0)]
    val_metrics_list = [validation_losses.mean(axis=0), validation_accuracies.mean(axis=0)]
    metric_names = ['Loss', 'Accuracy']
    fig, ax = plot_metrics(plot_title, train_metrics_list, val_metrics_list, metric_names)
    
    # Save training metrics plot
    directory_path = ('C:\\Users\\ryanm\\OneDrive\\Desktop\\deep learning\\reports\\documentation\\graphs\\MLP single vector\\' +
            str(model_params['hl1']['units']) + '-' + str(model_params['hl1']['activation']) + '-' + 'preprocess-' + str(preprocess) + '-' +
            'epochs-' + str(epochs) + '-' + 'batch_size-' + str(batch_size) + '-' + 'n_folds-' + str(n_folds) + '-' + 'learning_rate-' + 
            str(model_params['compile']['optimizer'].learning_rate.numpy()) + '-without-reguralisation'+'\\training_metrics.png')
    os.makedirs(os.path.dirname(directory_path), exist_ok=True)
    fig.savefig(directory_path)

    # Test the model and get metrics
    cm, accuracy, precision, recall, f1 = test_model_and_get_metrics(mlp=mlp, X_test=X_test_glove_embedded, y_test=y_test)
    
    # Save metrics to CSV file
    metrics_df = pd.DataFrame({
        '': ['Predicted Positive', 'Predicted Negative', '','Accuracy', 'Precision', 'Recall', 'F1', 'Average Training Time'],
        'Actual Positive': [cm[1,1], cm[1,0], '', accuracy, precision, recall, f1, str(total_training_time)],
        'Actual Negative': [cm[0,1], cm[0,0], '', '', '', '', '', ''],
    })

    #  
    file_path = ('C:\\Users\\ryanm\\OneDrive\\Desktop\\deep learning\\reports\\documentation\\graphs\\MLP single vector\\' +
                str(model_params['hl1']['units']) + '-' + str(model_params['hl1']['activation']) + '-' + 'preprocess-' + str(preprocess) + '-' +
                'epochs-' + str(epochs) + '-' + 'batch_size-' + str(batch_size) + '-' + 'n_folds-' + str(n_folds) + '-' + 'learning_rate-' + 
                str(model_params['compile']['optimizer'].learning_rate.numpy()) + '-without-reguralisation' + '\\testing_metrics.csv')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    metrics_df.to_csv(file_path, index=False)
    print("Metrics have been saved to 'testing_metrics.csv'")

def create_embedding_matrix(tokenizer, glove, num_words):
    # Construct the model weight matrix for embedding layer
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in tokenizer.word_index.items():
        if word in glove.key_to_index:
            embedding_vector = glove[word]
            embedding_matrix[i] = embedding_vector



if __name__ == "__main__":

    # Paths to files needed 
    TRAIN_PATH = 'C:\\Users\\ryanm\\OneDrive\\Desktop\\deep learning\\assignments\\Deep-Learning-Assignment-2\\train.csv'
    TEST_PATH = 'C:\\Users\\ryanm\\OneDrive\\Desktop\\deep learning\\assignments\\Deep-Learning-Assignment-2\\test.csv'
    EMBEDDINGS_PATH = 'C:\\Users\\ryanm\\OneDrive\\Desktop\\deep learning\\assignments\\Deep-Learning-Assignment-2\\glove.6B.300d.txt.word2vec'

    # Static configurations - Loading the test and training data and the pre-trained glove embeddings file
    X_train, y_train, X_test, y_test = load_datasets(TRAIN_PATH, TEST_PATH)
    glove = load_embeddings(EMBEDDINGS_PATH)

    # Without pre-processing
    # test 1: baseline

    model_params = {
        'hl1': {'units': 5, 'activation': 'relu', 'input_shape': None, 'kernel_initializer': 'he_uniform', 'kernel_regularizer': None},
        'output': {'units': 1, 'activation': 'sigmoid'},
        'compile': {'optimizer': tf.keras.optimizers.Adam(learning_rate=0.05), 'loss': 'binary_crossentropy', 'metrics': ['accuracy']}
    }
    test_model_configurations(X_train, y_train, X_test, y_test, preprocess=False, model_params=model_params, batch_size=64, epochs=10, n_folds=10)

    # add L1 regularisation
    # model_params = {
    #     'hl1': {'units': 5, 'activation': 'relu', 'input_shape': None, 'kernel_initializer': 'he_uniform', 'kernel_regularizer': regularizers.l1(0.01)},
    #     'output': {'units': 1, 'activation': 'sigmoid'},
    #     'compile': {'optimizer': tf.keras.optimizers.Adam(learning_rate=0.05), 'loss': 'binary_crossentropy', 'metrics': ['accuracy']}
    # }
    # test_model_configurations(X_train, y_train, X_test, y_test, preprocess=False, model_params=model_params, batch_size=64, epochs=10, n_folds=10)



    # Test 2: pre-processing
    # model_params = {
    #     'hl1': {'units': 5, 'activation': 'relu', 'input_shape': None, 'kernel_initializer': 'he_uniform', 'kernel_regularizer':None},
    #     'output': {'units': 1, 'activation': 'sigmoid'},
    #     'compile': {'optimizer': tf.keras.optimizers.Adam(learning_rate=0.05), 'loss': 'binary_crossentropy', 'metrics': ['accuracy']}
    # }
    # test_model_configurations(X_train, y_train, X_test, y_test, preprocess=True, model_params=model_params, batch_size=64, epochs=10, n_folds=10)

    # Test 3: regularisation
    # L1 regularisation
    # model_params = {
    #     'hl1': {'units': 5, 'activation': 'relu', 'input_shape': None, 'kernel_initializer': 'he_uniform', 'kernel_regularizer': regularizers.l1(0.01)},
    #     'output': {'units': 1, 'activation': 'sigmoid'},
    #     'compile': {'optimizer': tf.keras.optimizers.Adam(learning_rate=0.05), 'loss': 'binary_crossentropy', 'metrics': ['accuracy']}
    # }
    # test_model_configurations(X_train, y_train, X_test, y_test, preprocess=True, model_params=model_params, batch_size=64, epochs=10, n_folds=10)

    # # L2 reguralisation
    # model_params = {
    #     'hl1': {'units': 5, 'activation': 'relu', 'input_shape': None, 'kernel_initializer': 'he_uniform', 'kernel_regularizer': regularizers.l2(0.01)},
    #     'output': {'units': 1, 'activation': 'sigmoid'},
    #     'compile': {'optimizer': tf.keras.optimizers.Adam(learning_rate=0.05), 'loss': 'binary_crossentropy', 'metrics': ['accuracy']}
    # }
    # test_model_configurations(X_train, y_train, X_test, y_test, preprocess=True, model_params=model_params, batch_size=64, epochs=10, n_folds=10)

    # # Learning rate: 0.05, 0.025, 0.001
    # model_params = {
    #     'hl1': {'units': 5, 'activation': 'relu', 'input_shape': None, 'kernel_initializer': 'he_uniform', 'kernel_regularizer': regularizers.l1(0.01)},
    #     'output': {'units': 1, 'activation': 'sigmoid'},
    #     'compile': {'optimizer': tf.keras.optimizers.Adam(learning_rate=0.025), 'loss': 'binary_crossentropy', 'metrics': ['accuracy']}
    # }
    # test_model_configurations(X_train, y_train, X_test, y_test, preprocess=True, model_params=model_params, batch_size=64, epochs=10, n_folds=10)

    # model_params = {
    #     'hl1': {'units': 5, 'activation': 'relu', 'input_shape': None, 'kernel_initializer': 'he_uniform', 'kernel_regularizer': regularizers.l1(0.01)},
    #     'output': {'units': 1, 'activation': 'sigmoid'},
    #     'compile': {'optimizer': tf.keras.optimizers.Adam(learning_rate=0.01), 'loss': 'binary_crossentropy', 'metrics': ['accuracy']}
    # }
    # test_model_configurations(X_train, y_train, X_test, y_test, preprocess=True, model_params=model_params, batch_size=64, epochs=10, n_folds=10)


    # Test 4: Batch size: 32, 64, 128

 

    

    

   
