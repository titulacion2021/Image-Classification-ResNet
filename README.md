# Residual Neural Network for COVID-19 Classification with Bayesian Data Augmentation
We implement a Residual Convolutional Neural Network (ResNet) for COVID-19 medical image (CXR) classification task. 
ResNet solves the vanishing gradient problem, allowing deeper networks constructions by adding more layers and making the algorithm easier to train.
A dimensional reduction technique (PCA) is used for an efficient usage of the available computational resources. 
Moreover, data augmentation approach implemented enrich the data set by obtaining more images for the algorithm training, 
in consequence, nine hyperparameters were chosen related to model configurations and data augmentation operations. 
Hyperparameters were fixed using a Bayesian hyperparameter optimization approach, which allows to improve the model performance and accuracy. 
Experimental results showed that techniques applied to ResNet architecture helps to obtain good accuracy and training performance when a small
data sets and limited computational resources are available.

# Files
You will find two .ipynb files. The 'Pre_processing.ipynb' is the code related to the whole data set preprocessing (*RGB2GRAY, RESIZE, CLAHE*). The second file 'ResNet.ipynb' is related to the processed images loading, Exploratory Data Analysis, Principal Component Analysis, Bayesian Optimization and the ResNet model with their final results.

It is not necessary to run the first file, since the uploaded dataset images are already enhaced, however feel free to use this code to pre-process aditional chest-CXR images. The second file 'ResNet.ipynb' will load the whole data sets from a Google Drive directory so be sure to update the directories in the loading data section. In next sections you will see the Bayesian Hyperparameter optimization experiments. Finally the ResNet is presented with the optimal found hyperparameters.

# Data
The original data was obtanined from Github (https://github.com/abzargar/COVID-Classifier/tree/master/dataset/original_images).However, the processed enhaced images are available in the folder DataSet. This folder contains the images related to the training, validation and test set. 

# Computational Requeriments
This is a deep learning convolutional neural networK, so it will require good computational resources, in order to obtain the final result. The Bayesian Hyperparameter Optimization needs 2.79 available memory RAM while the ResNet model with the optimized hyperparameters needs 15.79 memory RAM. If this resources are not available for running the code you can use Google Colaboratory PRO.
# Test Results

                    precision    recall  f1-score   support

   Covid (Class 0)       0.84      1.00      0.91        21
  Normal (Class 1)       1.00      1.00      1.00        21
Pneumonia(Class 2)       1.00      0.81      0.89        21

          accuracy                           0.94        63
         macro avg       0.95      0.94      0.94        63
      weighted avg       0.95      0.94      0.94        63


# Software Dependecies
This model was writen in Google Colab and some libraries were used or installed, such as:
 - [Matplotlib] (https://matplotlib.org/)
 - [Seaborn] (https://seaborn.pydata.org/)
 - [Scikit-Image] (https://scikit-image.org/)
 - [Keras] (https://keras.io/)
 - [TensorFlow] (https://www.tensorflow.org/?hl=es-419)
 - [Numpy] (https://numpy.org/)
 - Install bayesian-optimization, GPy and GpyOpt.

# Background
In order to understand this project, a basic understanding of machine learning is requeried. There are also a number of specific resources that were used the ResNet
was developed:
- [Standford Course] (https://cs231n.github.io/)
- [Jiang, et al 2019] (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0214587)
- [TensorFlow tutorials] (https://www.tensorflow.org/versions/r0.11/tutorials/index.html)
- [Goodfellow et al. 2013] (https://arxiv.org/pdf/1312.6082v4.pdf)

# Accompanying Report
This project has a PDF report, where it is explained and analized the code and the results. 

# License
This project uses the next license [https://github.com/titulacion2021/Image-Classification-ResNet/blob/main/LICENSE]

