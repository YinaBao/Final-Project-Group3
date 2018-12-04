# Final-Project-Group3
THE STREET VIEW HOUSE NUMBERS (SVHN) DATASET PROJECT

In this project, we used street view house numbers dataset to explore how computer recognizes digits numbers from images. For this real-world image dataset, we engage in can be applied in image recognition and classification from image to digits which is an important topic in machine learning field. This real-world street view house number image dataset is different than the other dataset because it has over 600,000-digit images and the numbers are in the natural scene images. SVHN is obtained from house numbers in Google Street View images. By giving pixel value vectors as features, we are leveraging ten classification models to predict the label of images. In this project, we aimed at applying various machine learning including deep learning techniques in order to do the image classification. The models respectively are Multilayer Perceptron Networks (MLP), Convolutional Neural Network (CNN) on PyTorch and 1 Convolutional Neural Network on Caffe. Though these different models, we can also find out how these algorithms compare to each other and how can we improve our model.

Link of Dataset:http://ufldl.stanford.edu/housenumbers/

Our main project is based on Pytorch data frame. Environment Sepecification: Python 2.7 or above Set DISPLAY=localhost:10.0 Connect to GPU

Before running the code, please make sure Please run ‘MLP.py’ first. This file include the step of downloading the data. The dataset will save in the ‘./data_svhn’ folder under the same directory. ‘CNN4.py’ is the code of our best performance model (model 4) ‘CNN4_extra.py’ is the code of the best performance model by using extra dataset, which can reach 94% accuracy.

We also transfer one of our experimental model to Caffe data frame. For Running the code, please set the PYTHONPTAH=/home/ubuntu/caffe/python, DISPLAY=localhost:10.0 and change the interperter to Python 2.7 and connected to GPU Run ‘caffe_covert_lmdb.py’ file first to convert the data to lmdb format for Caffe. Run ‘mlcnn_sol.py’ to get the result. (In order to run this code, make sure ‘mlcnn_train_test.prototxt’ and ‘mlcnn_solver.prototxt’ at the same directory)

The code we used to convert the data to lmdb format is cited from the website below:
