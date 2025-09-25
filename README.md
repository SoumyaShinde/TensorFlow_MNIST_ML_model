# MNIST Handwritten Digit Classification

A simple neural network implementation using TensorFlow/Keras for classifying handwritten digits from the MNIST dataset.

## Overview

This project contains a Jupyter notebook (`TensorFlow.ipynb`) that demonstrates the complete workflow of building, training, and saving a neural network model for digit recognition.

## Model Architecture

The implemented model uses a Sequential architecture with the following layers:
- **Flatten Layer**: Converts 28x28 images to 1D vectors
- **Dense Layer**: 128 neurons with ReLU activation
- **Dropout Layer**: 0.2 dropout rate to prevent overfitting
- **Output Layer**: 10 neurons with softmax activation (for 10 digit classes 0-9)

## Dataset

- **MNIST Dataset**: Contains handwritten digits (0-9)
- **Training Data**: 60,000 images
- **Test Data**: 10,000 images
- **Image Size**: 28x28 pixels, grayscale
- **Preprocessing**: Pixel values normalized from 0-255 to 0-1 range

## Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metric**: Accuracy
- **Epochs**: 5
- **Final Training Accuracy**: ~97.63%
- **Test Accuracy**: 97.75%

## Files

```
├── TensorFlow.ipynb          # Main notebook with complete implementation
└── my_mnist_model.keras      # Saved trained model (generated after running notebook)
```

## Usage

### Prerequisites

```bash
pip install tensorflow jupyter matplotlib
```

### Running the Notebook

1. Open the Jupyter notebook:
```bash
jupyter notebook TensorFlow.ipynb
```

2. Run all cells sequentially to:
   - Load and preprocess the MNIST dataset
   - Build the neural network model
   - Compile and train the model
   - Evaluate model performance
   - Save the trained model

### Using the Trained Model

After running the notebook, you can load and use the saved model:

```python
import tensorflow as tf

# Load the saved model
loaded_model = tf.keras.models.load_model('my_mnist_model.keras')

# Load test data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  # Normalize

# Make predictions
predictions = loaded_model.predict(x_test)
```

## Model Performance

The trained model achieves:
- **Test Loss**: 0.0756
- **Test Accuracy**: 97.75%

Training progression over 5 epochs:
- Epoch 1: 85.60% accuracy
- Epoch 2: 95.43% accuracy
- Epoch 3: 96.65% accuracy
- Epoch 4: 97.20% accuracy
- Epoch 5: 97.63% accuracy

## Key Features

- Simple and straightforward implementation
- Efficient training (completes in seconds)
- High accuracy on test data
- Model persistence (save/load functionality)
- Well-commented code explaining each step

## Technical Details

- **Framework**: TensorFlow 2.x with Keras API
- **Model Type**: Sequential feedforward neural network
- **Activation Functions**: ReLU (hidden layer), Softmax (output layer)
- **Regularization**: Dropout (0.2 rate)
- **Data Normalization**: Pixel values scaled to [0,1] range

## Getting Started

1. Clone this repository
2. Install required dependencies
3. Open and run the Jupyter notebook
4. The trained model will be automatically saved as `my_mnist_model.keras`

## Notes

- The notebook includes warnings about input_shape usage which can be ignored
- All training output and model evaluation results are preserved in the notebook
- The implementation follows TensorFlow/Keras best practices for beginners
