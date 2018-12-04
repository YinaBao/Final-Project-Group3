# Final-Project-Group3
##  THE STREET VIEW HOUSE NUMBERS (SVHN) DATASET PROJECT
### Author: Yina Bao, Jue Wang, Tianyi Wang
In this project, we used street view house numbers dataset to explore how computer recognizes digits numbers from images. For this real-world image dataset, we engage in can be applied in image recognition and classification from image to digits which is an important topic in machine learning field. This real-world street view house number image dataset is different than the other dataset because it has over 600,000-digit images and the numbers are in the natural scene images.  SVHN is obtained from house numbers in Google Street View images. By giving pixel value vectors as features, we are leveraging ten classification models to predict the label of images. In this project, we aimed at applying various machine learning including deep learning techniques in order to do the image classification. The models respectively are Multilayer Perceptron Networks (MLP), Convolutional Neural Network (CNN) on PyTorch and 1 Convolutional Neural Network on Caffe. Though these different models, we can also find out how these algorithms compare to each other and how can we improve our model.

Link of Dataset: http://ufldl.stanford.edu/housenumbers/


### Running the code
Environment Sepecification: 
* Python 2.7 or above
* Set DISPLAY=localhost:10.0
* Connect to GPU

Before running the code, please make sure you have the following packages installed: torch, torchvision, matplotlib, sklearn, itertools, numpy, scipy, lmdb, caffe. 

Our main project is based on Pytorch data frame. 
* Run 'MLP.py' first. This file include the step of downloading the data. The dataset will save in the './data_svhn' folder under the same directory. (MLP model, 80% accuracy)
* Run 'CNN4.py' file, which is the code of our best performance model (CNN model 4, 91% accuracy)
* Run 'CNN4_extra.py' file, which is the code of the best performance model by using extra dataset (94% accuracy). 

Each file above will generate sample images of the batch, plot of loss, plot of ROC curve for each class, confusion matrix, training process, testing accuracy and f1 score for each model. 


We also transfer one of our experimental model to Caffe data frame. 
For Running the code, please set the PYTHONPTAH=/home/ubuntu/caffe/python, DISPLAY=localhost:10.0 and change the interpreter to Python 2.7 and connected to GPU.
* Run 'caffe_covert_lmdb.py' file first to convert the data to lmdb format for Caffe. 
* Run 'mlcnn_sol.py' file to get the result. (In order to run this code, make sure 'mlcnn_train_test.prototxt','mlcnn_solver.prototxt' and 'mean.binaryproto' at the same directory)

The code we used to convert the data to lmdb format is cited from the website below: https://github.com/junyuseu/uncommon-datasets-caffe/blob/master/scripts/convert_svhn_to_lmdb.py


Please read the report in the 'Final-Group-Project-Report' folder for detailed analysis. We put our individual reports into the 'Individual-Report' folder.

## Credit
This project is a collaborative effort of Tianyi Wang(jessicawang@gwu.edu), Yina Bao(baoyina@gwu.edu) and Jue Wang(jue_w72@gwu.edu).
