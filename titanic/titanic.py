#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]
#FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch","Ticket","Fare","Cabin","Embarked"]

COLUMNS_TEST = ["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]

#FIXME: which features can i use with this??
# Age can be empty. How do I account for that? What about other inputs??
FEATURES = ["Pclass","Fare"]
LABEL = "Survived"


# Read the training set using pandas
training_set = pd.read_csv("data/train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
# test_set = pd.read_csv("data/test.csv", skipinitialspace=True,
#                        skiprows=1, names=COLUMNS_TEST)
prediction_set = pd.read_csv("data/test.csv", skipinitialspace=True,
                            skiprows=1, names=COLUMNS_TEST)



# tells tensorflow to expect real valued input for all entries in FEATURES
feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

# Define the neural network
regressor = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,
                                          hidden_units=[10, 10],
                                          n_classes=2,
                                          model_dir="./tmp/titanic_model")

# Make an input function to provide data
def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values)
                  for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels

# Make an input function to provide data
def input_fn_predict(data_set):
  feature_cols = {k: tf.constant(data_set[k].values)
                  for k in FEATURES}
  #labels = tf.constant(data_set[LABEL].values)
  return feature_cols


# Train the NN on the training data set
# loss function is apparently a mean squared error
regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

# Eval against training set itself
ev = regressor.evaluate(input_fn=lambda: input_fn(training_set), steps=1)
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

y = regressor.predict(input_fn=lambda: input_fn_predict(training_set))
predictions = list(itertools.islice(y, len(prediction_set)))
print ("Predictions on training set: {}".format(str(predictions)))


y = regressor.predict(input_fn=lambda: input_fn_predict(prediction_set))
# .predict() returns an iterator; convert to a list and print predictions
predictions = list(itertools.islice(y, len(prediction_set)))
print ("Predictions: {}".format(str(predictions)))