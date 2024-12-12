import os

from sklearn.utils import validation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
from datetime import datetime
import tensorflow as tf

# Import from our organized packages
from src import DataPreprocessor, GoodNoGoodClassifier, ModelTrainer
from utils.visualization import MLVisualization

def setup_logging():
    """Configure logging for the training process"""
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
    """Check and log GPU availability for training"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f"Available GPUs: {len(gpus)}")
        for gpu in gpus:
            logging.info(f"GPU device: {gpu}")
    else:
        logging.warning("No GPU available. Training will proceed on CPU.")
    return bool(gpus)

def verify_data_directories(train_dir, validation_dir):
    """Verify that data directories exist and contain required class folders"""
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

def main():
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
        
        # Verify data directories
        logging.info("Verifying data directories")
        verify_data_directories(TRAIN_DIR, VALIDATION_DIR)
        
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
        
        # Create results directory and save visualizations
        os.makedirs(RESULTS_DIR, exist_ok=True)
        logging.info(f"Saving results to {RESULTS_DIR}")
        
        # Generate and save training visualizations
        visualizer = MLVisualization()

        visualizer.plot_training_history(
            history=history,
            save_path=os.path.join(RESULTS_DIR, 'training_history.png')
        )
        
        # Generate predictions on validation set for evaluation
        validation_predictions = model.predict(validation_generator)
        validation_labels = validation_generator.classes
        predicted_classes = (validation_predictions > 0.5).astype(int)
        
        # Create and save confusion matrix
        visualizer.plot_confusion_matrix(
            y_true=validation_labels,
            y_pred=predicted_classes,
            class_names=class_names,
            save_path=os.path.join(RESULTS_DIR, 'confusion_matrix.png')
        )
        
        # Generate and save classification report
        visualizer.generate_classification_report(
            y_true=validation_labels,
            y_pred=predicted_classes,
            class_names=class_names,
            save_path=os.path.join(RESULTS_DIR, 'classification_report.txt')
        )
        
        # Save the trained model
        model_save_path = os.path.join(RESULTS_DIR, 'good_no_good_classifier.h5')
        trainer.save_model(model_save_path)
        logging.info(f"Model saved to {model_save_path}")
        
        logging.info("Training process completed successfully")
        
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}", exc_info=True)
        raise
    
if __name__ == '__main__':
    main()
