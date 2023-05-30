# Age Detection using Deep Learning

This GitHub repository contains a deep learning model for age detection using facial images. The goal of the project is to accurately predict the age of individuals based on their facial features. The model is built using two different approaches: one from scratch and the other utilizing a pre-trained MobileNet CNN model for obtaining embeddings.

## Dataset
The dataset used in this project is sourced from Kaggle and can be found [here](https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset). It consists of a collection of facial images with corresponding age labels, allowing the development of an age detection model. The dataset provides a diverse range of facial images, enabling the model to learn and generalize effectively.

## Model Architecture
Two deep learning models are implemented in this project. The first model is built from scratch, utilizing a custom architecture designed specifically for age detection. This model is trained on the provided dataset using transfer learning techniques and data augmentation to enhance performance.

The second model employs a MobileNet pre-trained CNN model as a feature extractor. The pre-trained model is fine-tuned on the custom dataset, leveraging the knowledge gained from a large-scale image recognition task. This approach aims to leverage the powerful feature extraction capabilities of the pre-trained model, enhancing the model's ability to detect age accurately.

## Performance Evaluation
The models' performance is evaluated using various metrics, including accuracy, confusion matrices, and ROC curves. The accuracy metric provides an overall measure of the models' predictive capabilities. Confusion matrices allow for a detailed analysis of the model's performance across different age groups, highlighting potential biases or misclassifications. ROC curves provide insights into the model's ability to balance true positive and false positive rates, enabling the determination of an optimal threshold for age classification.

## Results
After extensive training and evaluation, the MobileNet-based model achieved an accuracy of 79% on the age detection task. The model's performance is discussed in detail, providing insights into its strengths and areas for potential improvement. The results demonstrate the effectiveness of deep learning techniques for age detection and showcase the benefits of leveraging pre-trained models for improved performance.

## Usage
The repository includes the necessary code and instructions to train and evaluate the age detection models. The dataset and pre-trained MobileNet model are provided, enabling easy replication and exploration of the project. Detailed documentation and tutorials are included to guide users through the process of building and evaluating the models.

## Dependencies
The project is implemented in Python, utilizing popular deep learning libraries such as TensorFlow and Keras. The required dependencies are listed in the repository, along with instructions on how to set up the environment.

## Contributions and Feedback
Contributions to this project are welcome! If you have any ideas or suggestions for improvement, please feel free to open an issue or submit a pull request. Your feedback and contributions will help enhance the project and make it more valuable to the community.

## License
The code and resources in this repository are provided under the [MIT License](LICENSE). Feel free to use and modify the code for academic, research, or commercial purposes. Please refer to the license file for more details.

We hope this project provides a valuable resource for age detection using deep learning techniques. Enjoy exploring and experimenting with the models!

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)