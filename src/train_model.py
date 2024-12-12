import os
import tensorflow as tf
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, model, train_generator, validation_generator):
        """
        Initialize model trainer
        
        Args:
            model (tf.keras.Model): Compiled Keras model
            train_generator: Training data generator
            validation_generator: Validation data generator
        """
        self.model = model
        self.train_generator = train_generator
        self.validation_generator = validation_generator
    
    def train(self, epochs=30, early_stopping=True):
        """
        Train the model
        
        Args:
            epochs (int): Number of training epochs
            early_stopping (bool): Whether to use early stopping
        
        Returns:
            History object from model training
        """
        # Callbacks
        callbacks = []
        
        if early_stopping:
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True
            )
            callbacks.append(early_stop)
        
        # Train the model
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.train_generator.batch_size,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples // self.validation_generator.batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def plot_training_history(self, history):
        """
        Visualize training history
        
        Args:
            history: Training history from model.fit()
        """
        # Create a directory for plots if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('results/training_history.png')
        plt.close()
    
    def save_model(self, save_path='models/good_no_good_classifier.h5'):
        """
        Save trained model
        
        Args:
            save_path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)

        print(f"Model saved to {save_path}")
