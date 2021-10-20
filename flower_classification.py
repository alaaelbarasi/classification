from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd 

cols_names=['sepalLength','sepalWidth','petalLength','petalWidth','Species']
species=['Setosa','Versicolor','Virginica']# classes the datapoints belong to

# sepcies
# 0=setosa
#1=Versicolor
#2=Virginica

#uploading data frame
train_path=tf.keras.utils.get_file('train.csv','https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')
test_path=tf.keras.utils.get_file('test.csv','https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')

#reading data frame
train=pd.read_csv(train_path,names=cols_names,header=0)
test=pd.read_csv(test_path,names=cols_names,header=0)
#all the stored data has numerical values.
# poping out species because it's the one we want to predicit
y_train=train.pop('Species')
y_test=test.pop('Species')
