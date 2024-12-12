# src/__init__.py

# Import main components to make them easily accessible
from .data_preprocessing import DataPreprocessor
from .model_architecture import GoodNoGoodClassifier
from .train_model import ModelTrainer

# Define version
__version__ = '1.0.0'

# This allows users to do:
# from src import DataPreprocessor, GoodNoGoodClassifier, ModelTrainer
