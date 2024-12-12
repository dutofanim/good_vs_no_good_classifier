import os
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

class DataPreprocessor:
    def __init__(self, train_dir, validation_dir, img_height=150, img_width=150, batch_size=32):
        """
        Initialize data preprocessor for image classification
        
        Args:
            train_dir (str): Path to training data directory
            validation_dir (str): Path to validation data directory
            img_height (int): Target image height
            img_width (int): Target image width
            batch_size (int): Batch size for data generators
        """
        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        
    def create_data_generators(self):
        """
        Create data generators with augmentation for training and validation
        
        Returns:
            tuple: (train_generator, validation_generator, class_names)
        """
        # Training data augmentation and generator
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values
            rotation_range=20,  # Random rotation
            width_shift_range=0.2,  # Random width shifts
            height_shift_range=0.2,  # Random height shifts
            shear_range=0.2,  # Shear transformation
            zoom_range=0.2,  # Random zoom
            horizontal_flip=True,  # Random horizontal flipping
            fill_mode='nearest'
        )
        
        # Validation data generator (only rescaling)
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Flow training images in batches
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary'  # Good vs No Good classification
        )
        
        # Flow validation images in batches
        validation_generator = validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary'
        )
        
        # Get class names
        class_names = list(train_generator.class_indices.keys())
        
        return train_generator, validation_generator, class_names
    
    def count_data_samples(self, generator):
        """
        Count total samples in data generator
        
        Args:
            generator: Keras ImageDataGenerator
        
        Returns:
            dict: Number of samples for each class
        """
        samples_per_class = {}
        for class_name, class_index in generator.class_indices.items():
            samples_per_class[class_name] = len(generator.classes[generator.classes == class_index])
        return samples_per_class
