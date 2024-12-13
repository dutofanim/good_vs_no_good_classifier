# Good vs No Good Image Classification Project

This project implements a Convolutional Neural Network (CNN) to classify images into "good" and "no good" categories. The system uses TensorFlow and Keras to create a robust image classification pipeline with training and prediction capabilities.

## Project Overview

The system consists of several key components:
- A CNN model architecture optimized for binary image classification
- Data preprocessing and augmentation pipeline
- Training workflow with early stopping
- Prediction capabilities for new images
- Comprehensive visualization tools for model performance
- Detailed logging system for tracking training and prediction

## System Requirements

Before running this project, ensure your system meets these requirements:

- Python 3.8 or higher
- CUDA-capable GPU (recommended but not required)
- At least 8GB of RAM
- Approximately 2GB of free disk space

## Installation

Follow these steps to set up the project on your machine:

1. First, clone the repository:
```bash
git clone https://github.com/yourusername/good-no-good-classifier.git
cd good-no-good-classifier
```

2. Create a virtual environment (recommended):
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Linux/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

After cloning, organize your data in the following structure:
```
good-no-good-classifier/
│
├── data/
│   ├── train/
│   │   ├── good/
│   │   │   └── (your good training images)
│   │   └── no_good/
│   │       └── (your no good training images)
│   │
│   └── validation/
│       ├── good/
│       │   └── (your good validation images)
│       └── no_good/
│           └── (your no good validation images)
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_architecture.py
│   └── train_model.py
│
├── utils/
│   ├── __init__.py
│   └── visualization.py
│
├── notebooks/
│   └── data_exploration.ipynb
│
├── requirements.txt
└── main.py
```

## Data Preparation

Your images should be:
- Organized in the directory structure shown above
- In common image formats (JPEG, PNG)
- Consistently sized (the model will resize them to 150x150 pixels)
- Split between training and validation sets

## Running the Project

The system operates in two modes: training and prediction.

### Training Mode

To train a new model:
```bash
python main.py train
```

This will:
1. Load and preprocess your training data
2. Train the CNN model
3. Generate performance visualizations
4. Save the trained model as 'good_no_good_classifier.h5'
5. Create detailed logs in the 'logs' directory

### Prediction Mode

To use a trained model for predictions:
```bash
python main.py predict
```
or simply:
```bash
python main.py
```

This will:
1. Load the trained model
2. Make predictions on the validation dataset
3. Generate performance metrics
4. Save results in the 'results' directory

## Project Outputs

The system generates several outputs in the 'results' directory:
- training_history.png: Plot of training and validation metrics
- confusion_matrix.png: Visualization of model performance
- classification_report.txt: Detailed performance metrics
- good_no_good_classifier.h5: The saved model file

## Logging

The system maintains detailed logs in the 'logs' directory. Each run creates a timestamped log file containing:
- Training progress
- Performance metrics
- Error messages (if any)
- System configuration details

## Troubleshooting

Common issues and solutions:

1. **GPU Not Detected:**
   - Verify CUDA installation
   - Check TensorFlow GPU compatibility
   - The system will fall back to CPU if no GPU is available

2. **Memory Issues:**
   - Reduce batch_size in main.py
   - Close other memory-intensive applications
   - Ensure sufficient free RAM

3. **Missing Data Directories:**
   - Verify the data directory structure matches the expected format
   - Ensure image files are in the correct folders
   - Check file permissions

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

If you encounter any issues:
1. Check the detailed logs in the 'logs' directory
2. Review the troubleshooting section
3. Open an issue on GitHub with:
   - Your system details
   - Complete error message
   - Steps to reproduce the issue

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project uses several open-source libraries:
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn

Special thanks to all contributors and the open-source community.
