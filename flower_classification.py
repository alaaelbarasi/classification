from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd 

cols_names=['sepalLength','sepalWidth','petalLength','petalWidth']
species=['Setosa','Versicolor','Virginica']# classes the datapoints belong to
#uploading data frame
train_path=tf.keras.utils.get_file('train.csv','https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')
test_path=tf.keras.utils.get_file('test.csv','https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')

#reading data frame
train=pd.read_csv(train_path,names=cols_names,header=0)
test=pd.read_csv(test_path,names=cols_names,header=0)
print(train.head())