import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, explained_variance_score
from scipy.stats import spearmanr, pearsonr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Nadam
from numpy import savetxt

# Load and preprocess the dataset
df = pd.read_csv('xxx100refi_steel_fatigue_crack_growth.csv')
df = df.drop('Label', axis=1)

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Filter data (if necessary, adjust filtering logic)
dfg = df  # Adjusted to include all data


X = dfg.drop('dadN', axis=1)
y = dfg['dadN']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Build the model
model = Sequential([
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Nadam(), loss='mae')

# Train the model
history = model.fit(
    x=X_train, y=y_train.values,
    validation_data=(X_val, y_val.values),
    batch_size=2, epochs=900
)

# Save the trained model
model.save('Trained_Model_Fatigue_Crack_Growth.keras')

# Save training and validation losses
losses = pd.DataFrame(history.history)
savetxt('Training_Loss_Epochs.csv', losses['loss'].values, delimiter=',')
savetxt('Validation_Loss_Epochs.csv', losses['val_loss'].values, delimiter=',')

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate error metrics
mae = mean_absolute_error(y_test, predictions)
variance = explained_variance_score(y_test, predictions)

# Convert y_test and predictions for correlation calculations
y_test_array = y_test.values
predictions_flat = predictions.flatten()

spearman_corr, _ = spearmanr(y_test_array, predictions_flat)
pearson_corr, _ = pearsonr(y_test_array, predictions_flat)

# Save error metrics
error_metrics = pd.DataFrame({
    'Metric': ['Mean Absolute Error', 'Explained Variance', 'Spearman Rank Correlation', 'Pearson Correlation'],
    'Value': [mae, variance, spearman_corr, pearson_corr]
})
error_metrics.to_csv('Error_Metrics_Fatigue_Crack_Growth.csv', index=False)

# Save predictions and actual values
savetxt('Predicted_dadN.csv', predictions_flat, delimiter=',')
savetxt('Actual_dadN.csv', y_test_array, delimiter=',')
savetxt('Test_Features.csv', X_test, delimiter=',')
savetxt('Train_Features.csv', X_train, delimiter=',')

def process_and_save(subset, label, model, scaler):
    # Extract features (X) and target (y)
    X_subset = subset.drop('dadN', axis=1)
    y_subset = subset['dadN']

    # Scale the features
    X_scaled = scaler.transform(X_subset)

    # Make predictions
    predictions = model.predict(X_scaled)

    # Combine predictions with original features
    combined_df = X_subset.copy()  # Start with the original features
    combined_df['dadN_actual'] = y_subset.values  # Add actual dadN values
    combined_df['dadN_predicted'] = predictions.flatten()  # Add predicted dadN values

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(f'Predictions_with_features_{label}.csv', index=False)

# Subsets of data for additional predictions
afg = df[df['P'] < 4]
cfg = df[(df['P'] > 2) & (df['P'] < 20)]
qfg = cfg.copy()
qfg['P'] = qfg['P'].replace(7.0, 14) #predicting for 14 MPa - replace with any pressure of interest
kfg = df[df['P'] > 15]

process_and_save(afg, "P_less_than_4", model, scaler)
process_and_save(cfg, "P_between_2_and_20", model, scaler)
process_and_save(qfg, "P_adjusted_7_to_14", model, scaler)
process_and_save(kfg, "P_greater_than_15", model, scaler)
