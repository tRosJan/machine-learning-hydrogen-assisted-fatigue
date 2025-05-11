import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, Nadam
import numpy as np
import itertools
import os
import absl.logging
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.FATAL)


# Load dataset
df = pd.read_csv('/users/aduwenye/Documents/ann_h2_fcgr_model/x52_combined.csv')
df = df.drop('Label', axis=1)
df = df[df['P'] > 0]
df = df.sample(frac=1).reset_index(drop=True)

# Filter dummies out here from dataframe if necessary
dfg = df[(df['P'] > 5.50) & (df['P'] != 6.89)]

X = dfg.drop('dadN', axis=1)
y = dfg['dadN']


X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define model-building function
def build_model(neuron_config, optimizer, activation):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))  
    for neurons in neuron_config:
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(1))  
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    return model

# Define parameter ranges
layer_range = range(2, 8)  
neuron_choices = [2**n for n in range(2, 7)]  
epoch_range = [150, 300, 900, 1500, 2000]  # Epochs
learning_rate_range = [0.01, 0.001, 0.0001]  
batch_size_range = [2, 4, 8]  
activation_choices = ['relu', 'selu', 'tanh']  
optimizer_choices = ['adam', 'nadam']  

# Generate valid neuron configurations
def is_valid_config(neuron_config):
    return all(neuron_config[i] >= neuron_config[i+1] for i in range(len(neuron_config) - 1))

neuron_configs = [
    neurons
    for num_layers in layer_range
    for neurons in itertools.product(neuron_choices, repeat=num_layers)
    if is_valid_config(neurons)
]



# Grid search, GridSearchCV is also available
best_mae = float('inf')
best_config = None

# Save results periodically, !!recent addons
results = []

for neuron_config in neuron_configs:
    for epochs in epoch_range:
        for lr in learning_rate_range:
            for batch_size in batch_size_range:
                for activation in activation_choices:
                    for optimizer_name in optimizer_choices:
                        print(f"Training with {neuron_config}, {epochs} epochs, learning rate {lr}, batch size {batch_size}, activation {activation}, optimizer {optimizer_name}...")

                        # Select optimizer
                        optimizer = Adam(learning_rate=lr) if optimizer_name == 'adam' else Nadam(learning_rate=lr)

                        # Build and train model
                        model = build_model(neuron_config, optimizer, activation)
                        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)

                        val_mae = np.min(history.history['val_mae'])
                        print(f"Validation MAE: {val_mae}")

                        # Save results, !!recent additions
                        results.append({
                            'neuron_config': neuron_config,
                            'epochs': epochs,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'activation': activation,
                            'optimizer': optimizer_name,
                            'val_mae': val_mae
                        })

                        if val_mae < best_mae:
                            best_mae = val_mae
                            best_config = (neuron_config, epochs, lr, batch_size, activation, optimizer_name)

                        # Periodically save results to a JSON file, !!recent additions
                        with open('/users/aduwenye/Documents/ann_h2_fcgr_model/grid_search_results.json', 'w') as f:
                            json.dump(results, f)

print(f"Best configuration: {best_config} with MAE: {best_mae}")
