# Residual Neural Network for COVID-19 Classification with Bayesian Data Augmentation
In this work we implement a Residual Convolutional Neural Network (ResNet) for COVID-19 medical image (CXR) classification task. 
ResNet solves the vanishing gradient problem, allowing deeper networks constructions by adding more layers and making the algorithm easier to train.
A dimensional reduction technique (PCA) is used for an efficient usage of the available computational resources. 
Moreover, data augmentation approach implemented enrich the data set by obtaining more images for the algorithm training, 
in consequence, nine hyperparameters were chosen related to model configurations and data augmentation operations. 
Hyperparameters were fixed using a Bayesian hyperparameter optimization approach, which allows to improve the model performance and accuracy. 
Experimental results showed that techniques applied to ResNet architecture helps to obtain good accuracy and training performance when a small
data sets and limited computational resources are available.

# Data
The original data was obtanined from Github (https://github.com/abzargar/COVID-Classifier/tree/master/dataset/original_images). If you want to run 
the project, you will have to download these images. Each folder has name, such as, pneumonia, covid and normal, in order to identify the different images. 

# Computational Requeriments
This is a deep learning convolutional neural networf, so it will require good computational resources, in order to obtain the final result. If you don´t 
have the enough resourses, you could run this model on a powerful cloud or decrease the number of epochs. 

# Software Dependecies
This model was writen in Google Colab and some libraries were used or installed, such as:
 - [Matplotlib] (https://matplotlib.org/)
 - [Seaborn] (https://seaborn.pydata.org/)
 - [Scikit-Image] (https://scikit-image.org/)
 - [Keras] (https://keras.io/)
 - [TensorFlow] (https://www.tensorflow.org/?hl=es-419)
 - [Numpy] (https://numpy.org/)
 - Install bayesian-optimization, GPy, GpyOpt.

# Accompanying Report
This project has a PDF report, where it is explained and analized the code and the results. 

# License
poner explicaci[on y link de la licencia

