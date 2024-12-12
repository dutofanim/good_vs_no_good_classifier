import tensorflow as tf
from tensorflow.keras import layers, models

class GoodNoGoodClassifier:
    def __init__(self, input_shape=(150, 150, 3), num_classes=1):
        """
        Initialize CNN model for binary classification
        
        Args:
            input_shape (tuple): Input image dimensions
            num_classes (int): Number of output classes (1 for binary)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def create_model(self):
        """
        Create Convolutional Neural Network model
        
        Returns:
            tf.keras.Model: Compiled CNN model
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                     tf.keras.metrics.Precision(), 
                     tf.keras.metrics.Recall()]
        )
        
        return model
    
    def model_summary(self, model):
        """
        Print model summary
        
        Args:
            model (tf.keras.Model): Keras model
        """
        model.summary()
