import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class MLVisualization:
    """
    A utility class for visualizing machine learning model results
    and data characteristics, specifically designed for good/no good classification.
    """
    
    def __init__(self):
        # Default class names for the project
        self.default_class_names = ['good', 'no_good']
        
        # Default color scheme for visualizations
        self.colors = {
            'good': '#2ecc71',      # Green for good class
            'no_good': '#e74c3c'    # Red for no good class
        }
    
    @staticmethod
    def create_results_directory(base_path='results'):
        """
        Create a directory to store visualization results
        
        Args:
            base_path (str): Base directory for saving visualizations
        
        Returns:
            str: Path to the created results directory
        """
        results_dir = os.path.abspath(base_path)
        os.makedirs(results_dir, exist_ok=True)
        return results_dir
    
    def plot_training_history(self, history, save_path=None):
        """
        Visualize model training history with custom styling
        
        Args:
            history: Keras training history
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 4))
        
        # Accuracy subplot with custom colors
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], color=self.colors['good'], 
                label='Training Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], color=self.colors['no_good'], 
                label='Validation Accuracy', linewidth=2, linestyle='--')
        plt.title('Model Accuracy Over Time', pad=15)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Loss subplot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], color=self.colors['good'],
                label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], color=self.colors['no_good'],
                label='Validation Loss', linewidth=2, linestyle='--')
        plt.title('Model Loss Over Time', pad=15)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """
        Create and plot confusion matrix with enhanced styling
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            class_names (list, optional): Names of classification classes
            save_path (str, optional): Path to save the plot
        """
        if class_names is None:
            class_names = self.default_class_names
            
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap with custom styling
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   square=True)
        
        # Enhance the plot appearance
        plt.title('Confusion Matrix\nGood vs No Good Classification', pad=20)
        plt.xlabel('Predicted Label', labelpad=10)
        plt.ylabel('True Label', labelpad=10)
        
        # Add accuracy score
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        plt.figtext(0.02, -0.1, 
                   f"Overall Accuracy: {accuracy:.2%}\n\n"
                   f"True Positives (Correct 'good'): {cm[0,0]}\n"
                   f"False Positives ('no_good' as 'good'): {cm[0,1]}\n"
                   f"False Negatives ('good' as 'no_good'): {cm[1,0]}\n"
                   f"True Negatives (Correct 'no_good'): {cm[1,1]}",
                   fontsize=10, ha='left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_classification_report(self, y_true, y_pred, class_names=None, save_path=None):
        """
        Generate and optionally save a detailed classification report
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            class_names (list, optional): Names of classification classes
            save_path (str, optional): Path to save the report
        
        Returns:
            str: Classification report text
        """
        if class_names is None:
            class_names = self.default_class_names
            
        report = classification_report(y_true, y_pred, target_names=class_names)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("Classification Report for Good vs No Good Classification\n")
                f.write("="*50 + "\n\n")
                f.write(str(report))
                f.write("\n\nNote: \n")
                f.write("- Precision: Accuracy of positive predictions\n")
                f.write("- Recall: Fraction of positives correctly identified\n")
                f.write("- F1-score: Harmonic mean of precision and recall\n")
        
        return report
    
    def plot_sample_predictions(self, images, true_labels, predicted_labels, 
                              num_samples=5, save_path=None):
        """
        Visualize model predictions with sample images
        
        Args:
            images (array): Input images
            true_labels (array): True labels
            predicted_labels (array): Predicted labels
            num_samples (int): Number of samples to plot
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(15, 3))
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            plt.subplot(1, num_samples, i+1)
            plt.imshow(images[idx])
            
            # Determine color based on prediction correctness
            color = self.colors['good'] if true_labels[idx] == predicted_labels[idx] else self.colors['no_good']
            
            # Get class names
            true_name = self.default_class_names[true_labels[idx]]
            pred_name = self.default_class_names[predicted_labels[idx]]
            
            plt.title(f'True: {true_name}\nPred: {pred_name}', color=color)
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
