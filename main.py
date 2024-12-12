import os
from sklearn.utils import validation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Import from our organized packages
from src import DataPreprocessor, GoodNoGoodClassifier, ModelTrainer
from utils.visualization import MLVisualization

class ModelPredictor:
    """
    Handles loading and using a trained model for predictions.
    This class encapsulates all prediction-related functionality.
    """
    def __init__(self, model_path):
        """
        Initialize the predictor with a saved model path.

        Args:
            model_path (str): Path to the saved .h5 model file
        """
        self.model_path = model_path
        self.model = None

    def load_saved_model(self):
        """
        Load the trained model from the .h5 file.
        Returns True if successful, False otherwise.
        """
        try:
            self.model = load_model(self.model_path)
            logging.info(f"Successfully loaded model from {self.model_path}")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return False

    def predict_batch(self, data_generator):
        """
        Make predictions on a batch of images using the data generator.

        Args:
            data_generator: Keras data generator containing images

        Returns:
            predictions: Model predictions
        """
        if self.model is None:
            self.load_saved_model()

        return self.model.predict(data_generator)

def setup_logging():
    """
    Configure logging for the training process.
    Creates a timestamped log file and also prints to console.
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Create timestamp for unique log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Configure logging
    logging.basicConfig(
        filename=os.path.join('logs', f'training_{timestamp}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Also print to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

def check_gpu_availability():
    """
    Check and log GPU availability for training.
    Returns True if GPU is available, False otherwise.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f"Available GPUs: {len(gpus)}")
        for gpu in gpus:
            logging.info(f"GPU device: {gpu}")
    else:
        logging.warning("No GPU available. Training will proceed on CPU.")
    return bool(gpus)

def verify_data_directories(train_dir, validation_dir):
    """
    Verify that data directories exist and contain required class folders.

    Args:
        train_dir (str): Path to training directory
        validation_dir (str): Path to validation directory

    Raises:
        FileNotFoundError: If required directories or class folders are missing
    """
    required_classes = ['good', 'no_good']

    for directory in [train_dir, validation_dir]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        available_classes = os.listdir(directory)
        for required_class in required_classes:
            if required_class not in available_classes:
                raise FileNotFoundError(
                    f"Required class '{required_class}' not found in {directory}"
                )

def train_model(TRAIN_DIR, VALIDATION_DIR, RESULTS_DIR):
    """
    Handles the complete model training workflow.

    Args:
        TRAIN_DIR (str): Path to training data directory
        VALIDATION_DIR (str): Path to validation data directory
        RESULTS_DIR (str): Path to save results

    Returns:
        tuple: (validation_generator, class_names)
    """
    # Initialize data preprocessing
    logging.info("Initializing data preprocessing")
    preprocessor = DataPreprocessor(
        train_dir=TRAIN_DIR,
        validation_dir=VALIDATION_DIR,
        img_height=150,
        img_width=150,
        batch_size=32
    )

    # Create and verify data generators
    train_generator, validation_generator, class_names = preprocessor.create_data_generators()
    logging.info(f"Created data generators with classes: {class_names}")

    # Log data distribution
    train_samples = preprocessor.count_data_samples(train_generator)
    validation_samples = preprocessor.count_data_samples(validation_generator)
    logging.info(f"Training samples distribution: {train_samples}")
    logging.info(f"Validation samples distribution: {validation_samples}")

    # Create the CNN model
    logging.info("Creating the CNN model")
    model_creator = GoodNoGoodClassifier(
        input_shape=(150, 150, 3),
        num_classes=1  # Binary classification
    )
    model = model_creator.create_model()

    # Log model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    logging.info("Model Architecture:\n" + "\n".join(model_summary))

    # Initialize trainer and start training
    logging.info("Initializing model training")
    trainer = ModelTrainer(
        model=model,
        train_generator=train_generator,
        validation_generator=validation_generator
    )

    # Train the model
    history = trainer.train(
        epochs=30,
        early_stopping=True
    )

    # Save visualizations and model
    visualizer = MLVisualization()

    visualizer.plot_training_history(
        history=history,
        save_path=os.path.join(RESULTS_DIR, 'training_history.png')
    )

    # Generate predictions on validation set
    validation_predictions = model.predict(validation_generator)
    validation_labels = validation_generator.classes
    predicted_classes = (validation_predictions > 0.5).astype(int)

    # Save evaluation metrics
    visualizer.plot_confusion_matrix(
        y_true=validation_labels,
        y_pred=predicted_classes,
        class_names=class_names,
        save_path=os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    )

    visualizer.generate_classification_report(
        y_true=validation_labels,
        y_pred=predicted_classes,
        class_names=class_names,
        save_path=os.path.join(RESULTS_DIR, 'classification_report.txt')
    )

    # Save the model
    model_save_path = os.path.join(RESULTS_DIR, 'good_no_good_classifier.h5')
    trainer.save_model(model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    return validation_generator, class_names

def main():
    """
    Main function that handles both training and prediction modes.
    Run with 'train' argument for training mode, or no argument for prediction mode.
    """
    # Set up logging
    setup_logging()
    logging.info("Starting the classification project")

    # Check GPU availability
    has_gpu = check_gpu_availability()

    try:
        # Project configuration
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        TRAIN_DIR = os.path.join(PROJECT_ROOT, 'data', 'train')
        VALIDATION_DIR = os.path.join(PROJECT_ROOT, 'data', 'validation')
        RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
        MODEL_PATH = os.path.join(RESULTS_DIR, 'good_no_good_classifier.h5')

        # Create results directory
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Determine mode from command line argument
        import sys
        mode = sys.argv[1] if len(sys.argv) > 1 else 'predict'

        if mode.lower() == 'train':
            # Training mode
            logging.info("Starting training mode")
            verify_data_directories(TRAIN_DIR, VALIDATION_DIR)
            validation_generator, class_names = train_model(TRAIN_DIR, VALIDATION_DIR, RESULTS_DIR)
            logging.info("Training process completed successfully")

        else:
            # Prediction mode
            logging.info("Starting prediction mode")
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError("No trained model found. Please run training first.")

            # Initialize predictor and load model
            predictor = ModelPredictor(MODEL_PATH)
            if not predictor.load_saved_model():
                raise RuntimeError("Failed to load the model")

            # Create data generator for validation data
            preprocessor = DataPreprocessor(
                train_dir=TRAIN_DIR,
                validation_dir=VALIDATION_DIR,
                img_height=150,
                img_width=150,
                batch_size=32
            )
            _, validation_generator, class_names = preprocessor.create_data_generators()

            # Make predictions
            logging.info("Making predictions on validation data")
            predictions = predictor.predict_batch(validation_generator)
            predicted_classes = (predictions > 0.5).astype(int)

            # Generate evaluation reports
            visualizer = MLVisualization()
            visualizer.plot_confusion_matrix(
                y_true=validation_generator.classes,
                y_pred=predicted_classes,
                class_names=class_names,
                save_path=os.path.join(RESULTS_DIR, 'prediction_confusion_matrix.png')
            )

            visualizer.generate_classification_report(
                y_true=validation_generator.classes,
                y_pred=predicted_classes,
                class_names=class_names,
                save_path=os.path.join(RESULTS_DIR, 'prediction_classification_report.txt')
            )

            logging.info("Prediction process completed successfully")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
