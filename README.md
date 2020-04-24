# SML_Project
Statistical Machine Learning Project

vggface:
  Siamese neural network for chest x-ray decease classification using an adapted vggface network (available: [TODO]).

autoencoder:
  An autoencoder for chest x-rays of healthy patients and patients affected by COVID-19 and Pneumonia. The hope is to use 
  the encoder with a shallow network to classify patients.
 
siamese:
  Siamese and traditional networks for chest x-ray decease classification using vgg16 networks pretrained on ImageNet database.

covidImageLoad2.py: Relabels the the files in the COVID-19 dataset by their class and patient ID number (avaliable: https://github.com/ieee8023/covid-chestxray-dataset).

covidImageLoad3.py: Relabels the the files in the NIH dataset by their class and patient ID number (avaliable: https://www.kaggle.com/nih-chest-xrays/data)
