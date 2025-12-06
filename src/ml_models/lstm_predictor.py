"""
LSTM-based Workload Predictor for Cloud Resource Utilization
Predicts future CPU and RAM utilization based on historical time-series data
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class LSTMPredictor:
    """
    LSTM model for predicting VM resource utilization
    """
    
    def __init__(self, sequence_length=10, prediction_horizon=1, 
                 lstm_units=64, dropout_rate=0.2):
        """
        Initialize LSTM predictor
        
        Args:
            sequence_length: Number of past timesteps to use for prediction
            prediction_horizon: Number of future timesteps to predict
            lstm_units: Number of LSTM units in each layer
            dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        
    def build_model(self, input_features=1):
        """
        Build LSTM model architecture
        
        Args:
            input_features: Number of input features (CPU, RAM, etc.)
        """
        self.model = Sequential([
            # First LSTM layer with return sequences
            LSTM(self.lstm_units, return_sequences=True, 
                 input_shape=(self.sequence_length, input_features)),
            Dropout(self.dropout_rate),
            
            # Second LSTM layer
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            
            # Dense layers for prediction
            Dense(32, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.prediction_horizon)
        ])
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def prepare_sequences(self, data):
        """
        Prepare time-series sequences for LSTM training
        
        Args:
            data: Array of resource utilization values
            
        Returns:
            X: Input sequences
            y: Target values
        """
        # Scale data
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        
        for i in range(len(data_scaled) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            X.append(data_scaled[i:i + self.sequence_length])
            # Target value(s)
            y.append(data_scaled[i + self.sequence_length:
                                  i + self.sequence_length + self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def train(self, data, validation_split=0.2, epochs=100, batch_size=32, 
              verbose=1, model_path='models/lstm_model.h5'):
        """
        Train LSTM model
        
        Args:
            data: Training data (CPU utilization time series)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            model_path: Path to save trained model
            
        Returns:
            history: Training history
        """
        # Prepare sequences
        X, y = self.prepare_sequences(data)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(input_features=X.shape[2])
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        return history
    
    def predict(self, sequence):
        """
        Predict future resource utilization
        
        Args:
            sequence: Recent utilization history (last 'sequence_length' values)
            
        Returns:
            prediction: Predicted future utilization value(s)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare input
        sequence_scaled = self.scaler.transform(sequence.reshape(-1, 1))
        sequence_reshaped = sequence_scaled.reshape(1, self.sequence_length, 1)
        
        # Predict
        prediction_scaled = self.model.predict(sequence_reshaped, verbose=0)
        
        # Inverse transform to original scale
        prediction = self.scaler.inverse_transform(prediction_scaled)
        
        return prediction[0]
    
    def predict_trend(self, sequence):
        """
        Predict workload trend (increasing/stable/decreasing)
        
        Args:
            sequence: Recent utilization history
            
        Returns:
            trend: 'increasing', 'stable', or 'decreasing'
        """
        prediction = self.predict(sequence)
        current_value = sequence[-1]
        predicted_value = prediction[0]
        
        # Define threshold for trend classification
        threshold = 0.05  # 5% change
        
        if predicted_value > current_value * (1 + threshold):
            return 'increasing'
        elif predicted_value < current_value * (1 - threshold):
            return 'decreasing'
        else:
            return 'stable'
    
    def save_model(self, model_path='models/lstm_model.h5', 
                   scaler_path='models/scaler.pkl'):
        """
        Save trained model and scaler
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_model(self, model_path='models/lstm_model.h5', 
                   scaler_path='models/scaler.pkl'):
        """
        Load pre-trained model and scaler
        """
        self.model = keras.models.load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.is_trained = True


class MultiResourceLSTM:
    """
    Multi-output LSTM for predicting multiple resources (CPU, RAM, Bandwidth)
    """
    
    def __init__(self, sequence_length=10, prediction_horizon=1, 
                 lstm_units=64, num_resources=2):
        """
        Initialize multi-resource predictor
        
        Args:
            num_resources: Number of resources to predict (e.g., CPU + RAM)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.lstm_units = lstm_units
        self.num_resources = num_resources
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
    
    def build_model(self):
        """Build multi-output LSTM model"""
        self.model = Sequential([
            LSTM(self.lstm_units, return_sequences=True,
                 input_shape=(self.sequence_length, self.num_resources)),
            Dropout(0.2),
            
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dense(self.num_resources * self.prediction_horizon)
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def prepare_sequences(self, data):
        """
        Prepare multi-feature sequences
        
        Args:
            data: Array of shape (timesteps, num_resources)
        """
        data_scaled = self.scaler.fit_transform(data)
        
        X, y = [], []
        
        for i in range(len(data_scaled) - self.sequence_length - self.prediction_horizon + 1):
            X.append(data_scaled[i:i + self.sequence_length])
            y.append(data_scaled[i + self.sequence_length:
                                 i + self.sequence_length + self.prediction_horizon].flatten())
        
        return np.array(X), np.array(y)
    
    def train(self, data, validation_split=0.2, epochs=100, batch_size=32):
        """Train multi-resource model"""
        X, y = self.prepare_sequences(data)
        
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def predict(self, sequence):
        """
        Predict multiple resources
        
        Args:
            sequence: Shape (sequence_length, num_resources)
            
        Returns:
            predictions: Shape (num_resources,)
        """
        sequence_scaled = self.scaler.transform(sequence)
        sequence_reshaped = sequence_scaled.reshape(1, self.sequence_length, self.num_resources)
        
        prediction_scaled = self.model.predict(sequence_reshaped, verbose=0)
        prediction_reshaped = prediction_scaled.reshape(self.prediction_horizon, self.num_resources)
        
        prediction = self.scaler.inverse_transform(prediction_reshaped)
        
        return prediction[0]


if __name__ == "__main__":
    # Example usage
    print("Testing LSTM Predictor...")
    
    # Generate synthetic workload data
    np.random.seed(42)
    time_steps = 1000
    cpu_data = np.sin(np.linspace(0, 20, time_steps)) * 50 + 50  # Oscillating workload
    cpu_data += np.random.normal(0, 5, time_steps)  # Add noise
    cpu_data = np.clip(cpu_data, 0, 100)
    
    # Initialize and train predictor
    predictor = LSTMPredictor(sequence_length=10, prediction_horizon=1)
    history = predictor.train(cpu_data, epochs=50, verbose=1)
    
    # Make predictions
    test_sequence = cpu_data[-10:]
    prediction = predictor.predict(test_sequence)
    trend = predictor.predict_trend(test_sequence)
    
    print(f"\nCurrent utilization: {test_sequence[-1]:.2f}%")
    print(f"Predicted utilization: {prediction[0]:.2f}%")
    print(f"Predicted trend: {trend}")
    
    # Save model
    predictor.save_model()
    print("\nModel saved successfully!")
