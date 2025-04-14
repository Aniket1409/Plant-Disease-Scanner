This model was trained on the [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) (CC BY-SA 4.0).  
Tutorial reference: [Plant Disease Detection System](https://www.youtube.com/playlist?list=PLvz5lCwTgdXDNcXEVwwHsb9DwjNXZGsoy).  

# Plant Disease Scanner

# Overview
This project uses a Convolutional Neural Network (CNN) to classify plant diseases based on leaf images from the Plant Village dataset. 
The model is trained to detect 38 different classes of plant diseases and healthy leaves.

# Dataset
- **Plant Village Dataset**: 
  - [Kaggle Link](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
  - [Original GitHub](https://github.com/spMohanty/PlantVillage-Dataset)

- **Dataset Structure**:
  - 87k RGB images of 38 types of crop leaves
  - **Train**: 70,295 images (80%)
  - **Valid**: 17,572 images (20%)
  - **Test**: 33 images for prediction
  - Subfolders named as `[plant.name_disease.name]` or `[plant.name_healthy]`

# Prerequisites
- Python 3.9
- Anaconda3 2024.10-1 (64-bit)
- NVIDIA GPU with latest drivers
- Microsoft Visual C++ 2015-2022 (x64)
- TensorFlow 2.10 (last version with GPU support)

# Installation (Anaconda Prompt)

### Show GPU driver, current GPU usage & CUDA version
```nvidia-smi```

### Create new environment named 'tensorflow_environment' with Python 3.9
```conda create -n tensorflow_environment python==3.9```

### Activate the created environment
```conda activate tensorflow_environment```

### Deactivate the current environment (corrected from 'conda activate')
```conda deactivate```

### List all available Conda environments
```conda env list```

### Install CUDA Toolkit 11.2 and cuDNN 8.1.0 for GPU support
```conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0```

### Upgrade pip to the latest version
```python -m pip install --upgrade pip```

### Change directory to the specified folder location
```cd <folder_location>```

### Install all Python libraries listed in requirements.txt
```pip install -r requirements.txt```

## Note: 
- using pip install > CPU version gets installed
- using requirements.txt > TF detects CUDA installation [for GPU] > installs GPU version

# Model

## 1. Importing Libraries

### Python Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

### Keras Components
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

###  Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix


## 2. Data Preprocessing [Image Data Loading]
- Input dimensions: 256×256 RGB images
- Batch size: 32 samples


## 3. Import CNN Model & Layers

### Sequential Model
- A linear stack of layers which can be added one by one.

### Conv2D
- A 2D convolutional layer to detect leaf features (edges, spots, textures).

### MaxPool2D
- A pooling layer that reduces (halves) spatial dimensions from (126, 126, 32) to (63, 63, 32), further reducing computation.

### Flatten
- Converts multi-dimensional data from MaxPool2D (6×6×256) into a 1D vector (9216 values) so it can be fed into a Dense (fully connected) layer.

### Dense (Fully Connected)
- A regular fully-connected neural network layer.


## 4. CNN Architecture

#### Sequential Model
- Linear stack of layers added sequentially
- Simple feed-forward architecture

#### Conv2D (2D Convolutional Layer)
- **Purpose**: Detects visual features (edges, spots, textures)
- **Operation**: Applies learned filters across spatial dimensions
- **Example**: Input (126, 126, 32) → Output (126, 126, 64)

#### MaxPool2D (Max Pooling Layer) 
- **Purpose**: Reduces spatial dimensions while preserving important features
- **Operation**: Takes maximum value from each window
- **Example**: Input (126, 126, 64) → Output (63, 63, 64) [with 2×2 pooling]

#### Flatten
- **Purpose**: Converts multi-dimensional data to 1D vector
- **Operation**: Reshapes (6, 6, 256) → (9216)
- **Why**: Prepares data for dense layers

#### Dense (Fully Connected Layer)
- **Purpose**: Final classification layer
- **Operation**: Each neuron connects to all inputs
- **Typical Use**: Last layer with softmax activation for classification

## CNN Architecture Used: 5 x [Pairs of Conv2D + MaxPooling] to balance detail preservation and computational cost


## 5. Build CNN Layers

- **feature map**: Resulting output
- **filters**: Number of patterns (features) to learn 
- **kernel_size**: Size of the sliding window (window to scan the image)
- **padding=same** [preserve image size]: Size of input image matches the size of feature matrix for each Conv2D layer
- **padding=valid** [reduce flatten parameters to avoid overfitting]: Shrinks for each Conv2D layer 
- **strides**: Stepwise movement speed of the sliding window (in pixels)
- **pool_size**: Window size (e.g., (2,2) halves dimensions)
- **dropout()**: Regularization step which randomly drops a percentage of neurons during each training step
- **relu**: If a leaf has a symptom - keeps the input. If no symptoms - ignores the input
- **relu units**: Number of neurons looking for different patterns (e.g., spot, color changes). More units give more detailed detection (but slower)
- **softmax**: Converts the detected features into probabilities for each class. Highest probability is the predicted class
- **softmax units**: Number of possible diseases (classes)

## Gradual Filter Increase with Each Conv2D [32 → 256 filters: simple → complex features]


## 6. Compiling Model

- **Adam(Adaptive Moment Estimation)**: optimization algorithm, adjusts learning rate adaptively to minimize prediction errors
- **learning_rate**: adjusts model weights (patterns)
- **categorical_crossentropy**: measures difference between model’s predictions and true labels
- **accuracy**: % of correctly classified leaves


## 7. Model Summary

- Each Conv2D reduces image size only slightly (e.g., 128×128 → 126×126)
- Dense(1024) outputs 1024 high-level features for the final classification layer
- Dense(38) outputs probabilities for 38 disease classes

## 8. Model Training

- **epoch**: number of times the model will be trained, adjust till loss/accuracy becomes still

### Challenges and Fixes

#### Overshooting
- **Description**: Model updates weights too aggressively (due to high learning rate), missing the optimal solution.
- **Signs of Overshooting**: Loss/accuracy fluctuates wildly during training.
- **Fix**: Use a smaller learning rate (changed from default 0.001 to 0.0001).

#### Overfitting
- **Description**: Model memorizes specific leaf images but fails on new images.
- **Signs of Overfitting**: High training accuracy, low validation accuracy.
- **Fix**: 
  - Add Dropout after dense layers.
  - Reduce the number of neurons (model size).

### Changes Made To The Model

- Increased dense layer neurons from 1024 to 1500
- Decreased learning rate size from adam default 0.001 to 0.0001
- Added Dropouts after conv2d layers (25%) and dense layer (40%)
- Added another conv2d layer with 512 filters to capture tiny disease signs (e.g., tiny lesions, texture changes)
- Removed padding from second conv2d layer to boost training speed

### Before Changes

- **Total params**: 10,649,414
- **Trainable params**: 10,649,414
- **Non-trainable params**: 0

### After Changes

- **Total params**: 7,842,762
- **Trainable params**: 7,842,762
- **Non-trainable params**: 0


### Model Testing

- Access class names of the dataset
- Load the validation set for testing the model, then use it to predict classes
- **Output**: 38 probabilities for 17572 images present in validation folder
- Vertically calculate the maximum probability for each image
- Iterate over test set

### Confusion Matrix
- The confusion matrix is generated to evaluate the model's performance

# CSV Data Exporter
- Python script to export plant disease data to a CSV file
- It generates `plant_disease_data.csv` with these columns:
   - Class Name
   - Disease
   - Symptoms
   - Treatment


# Website Using Streamlit

## Plant Disease Scanner
- A Streamlit app that identifies plant diseases from leaf photos using a trained TensorFlow model.
- Camera & Upload: Snap a photo or upload leaf images (JPG/PNG)
- AI Detection: Predicts diseases with confidence scores
- Treatment Guide: Shows symptoms and solutions for detected diseases

## Required Files
model.keras - Trained TensorFlow model
combined_disease_data.csv - Disease database Class Name, Disease, Symptoms, Treatment columns

## How It Works
Data Loading: Reads CSV into dictionary  
Prediction: Resizes images to 128x128, uses model.keras to predict disease class

## Results:
- Disease name with confidence %
- Expandable treatment guide

## Error Handling
- Catches invalid/corrupt images
- Handles missing disease data gracefully
- Works best with clear leaf photos against neutral backgrounds.

