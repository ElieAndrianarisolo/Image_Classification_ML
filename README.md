# Image Classification using Machine Learning

### Overview
This project implements a Convolutional Neural Network (CNN) to classify images of dogs and cats. The model is built using TensorFlow and Keras, and is trained on the popular *Dogs vs. Cats* dataset. The dataset is downloaded from Kaggle and processed to create a binary classification model. The key stages of the project include data preprocessing, model building, training, evaluation, and prediction.

### Dataset
The dataset used for this project is the *Dogs vs. Cats* dataset, available on Kaggle. It consists of labeled images of dogs and cats for binary classification.

- **Download location**: [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/salader/dogs-vs-cats)
- **Train/Test split**: The dataset is divided into training and testing sets to evaluate the performance of the model.

### Project Workflow
1. **Downloading the Dataset**: The dataset is downloaded from Kaggle using the `kaggle` API and extracted for further processing.
2. **Data Preprocessing**:
   - The images are loaded from their respective directories using `image_dataset_from_directory`.
   - Data normalization is applied to scale the pixel values to a range of 0 to 1.
3. **Model Creation**:
   - A Convolutional Neural Network (CNN) is constructed using Keras’ Sequential API.
   - The model contains multiple convolutional layers with batch normalization, followed by max pooling and dropout for regularization.
4. **Model Compilation**: The model is compiled with the Adam optimizer and binary cross-entropy loss for binary classification.
5. **Model Training**: The model is trained for 10 epochs with accuracy and loss being tracked for both the training and validation datasets.
6. **Model Evaluation**: The model's accuracy and loss are plotted over time to visualize performance.
7. **Testing the Model**: The trained model is tested on new images of dogs and cats to predict their class.

### Model Architecture
The CNN model consists of the following layers:
- Convolutional layers with ReLU activation and batch normalization
- MaxPooling layers
- Dense (fully connected) layers with dropout
- Output layer with sigmoid activation for binary classification (Dog = 0, Cat = 1)
