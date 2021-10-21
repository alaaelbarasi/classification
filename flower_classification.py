from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd 
import os

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

#input function the trainig process
def input_fn(features, labels, trainig=True,batch_zise=256):
    #The batch size is a number of samples processed before the model is updated
    ##The number of epochs is the number of complete passes through the training dataset
    ###Convert the input into dataset
    ds=tf.data.Dataset.from_tensor_slices((dict(features),labels))
    if trainig:
        ds=ds.shuffle(1000).repeat()
    return ds.batch(batch_zise)
# creating feature columns
feature_col=[] 
for key in train.keys():
     feature_col.append(tf.feature_column.numeric_column(key=key))

# building the model 
classifier=tf.estimator.DNNClassifier(feature_columns=feature_col,hidden_units=[30,10],n_classes=3)
# Trainig the model 
classifier.train(input_fn=lambda:input_fn(train,y_train,trainig=True),steps=5000)
#evaluating the model 
classifier.evaluate(input_fn=lambda:input_fn(test,y_test,trainig=False))
# clearinig the terminal 
os.system('cls' if os.name == 'nt' else 'clear') 
#prediction

def input_fn(features,batc_size=256):
    #converting the input into a dataset without labels
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batc_size)
features=['sepalLength','sepalWidth','petalLength','petalWidth']
predict={}
print(' ****Enter numeric values****')
for fetaure in features:
    vaild=True
    while vaild:
        val= input(fetaure+':')
        if not val.isdigit():vaild=False
    predict[fetaure]=[float(val)]

prediction= classifier.predict(input_fn=lambda:input_fn(predict))
for pred_dict in prediction:
    class_id=pred_dict['class_ids'][0]
    probability=pred_dict['probabilities'][class_id]
print('Prediction is "{}" ({:.1f}%)'.format(species[class_id], 100 * probability))
