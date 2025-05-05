import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout

class NavigationNN:
    def __init__(self, input_shape):
        """
        Args:
            input_shape: Tuple (height, width) ou (height, width, channels)
        """
        # Adiciona dimensão de canal se necessário
        if len(input_shape) == 2:
            input_shape = input_shape + (1,)  # Transforma em (height, width, 1)
        
        self.model = tf.keras.Sequential([
            InputLayer(input_shape=input_shape),  # Camada explícita de input
            Conv2D(32, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            MaxPooling2D((2,2)),
            Dropout(0.3),  # Adicionado para reduzir overfitting
            Flatten(),
            Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.3),  # Adicionado para reduzir overfitting
            Dense(2, activation='tanh')  # Saída: (Δx, Δy)
        ])
        # Compilação com métricas adicionais (MAE) e learning rate ajustável
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'cosine_similarity']  # Monitora erro absoluto e similaridade
        )

    def train(self, X, y, epochs=10, batch_size=32, validation_split=0.2):
        # Garante que X tem 4 dimensões: (amostras, height, width, channels)
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)
        
        # Callback para early stopping (evita overfitting)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
        
        history = self.model.fit(
            X, y, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            callbacks=callbacks  # Adiciona early stopping
        )
        return history

    def predict_next_move(self, grid):
        # Adiciona dimensões de batch e canal se necessário
        if grid.ndim == 2:
            grid = np.expand_dims(grid, axis=(0, -1))
        elif grid.ndim == 3:
            grid = np.expand_dims(grid, axis=0)
        return self.model.predict(grid, verbose=0)[0]

    def save_model(self, path):
        # Salva no formato moderno (.keras)
        self.model.save(path, save_format='keras')  # Ou use extensão .keras no path

    @classmethod
    def load_model(cls, path):
        model = tf.keras.models.load_model(path)
        nn = cls(input_shape=model.input_shape[1:])
        nn.model = model
        return nn