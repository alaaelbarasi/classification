# classification
Using DNN (Deep neural network) for classifying flowers into Setosa, Versicolor and Virginica depending on the following features Sepal length, Sepal width, Petal length and Petal width.
Expected questions:
1) What is lambda?
  Lambda is anonymous function we use it to execute another function by passing it after the colon with its arguments , instead of using nested functions the main advantage is to make the code shorter.
_____________________________________________________________

2)Nested function?
  Nested functions are used in languages like ALGOL, Simula 67 , Pascal and Python of course.
  Is simply defining a function inside another function but WHY? to encapsulate the information to hide it from external access.
_____________________________________________________________

3)Deep neural network (short: Deep net)
 It's an evaluation of ANN , where the network has multiple hidden layers that is why we call it "Deep" (2 layers at least) these hidden layers are used to store and evaluate how important one of the features (input) is to the labels which help us in predcting the class (in our cases) more acuuratly.
 classifier=tf.estimator.DNNClassifier(feature_columns=feature_col,hidden_units=[30,10],n_classes=3)
 in this line we used the DNN as a model where feature_columns takes all the names of the columns in the dataset that we are using.
 hidden_unit determine the number of the layers used in the network where 30 refer to the number of nodes in the first layes and 10 refer to the numner of nodes in the secand layer.
 n_classes the number of the output the we have in our case it we have 3 classes which are the 3 species (Setosa, Versicolor and Virginica) in general the value of this argument should be greater than 1 otherwise why we are classifing if we have one class, right !
 _____________________________________________________________
 
 
