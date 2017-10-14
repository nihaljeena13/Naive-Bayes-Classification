# Logistic Regression classifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score

data = pd.read_csv('spambase.data.csv',header=None,index_col=57)
data_train, data_test, train_label, test_label= train_test_split(data,data.index.values,test_size=0.5)
lg = LogisticRegression()
lg = lg.fit(data_train,train_label)
predict = lg.predict(data_test)

# Calculate the accuracy 
accuracy = (np.count_nonzero(predict == test_label) * 100) / (test_label.shape[0])
print("Accuracy of the classifier is %f \n"%(accuracy))

# Calculate the confusion matrix of the model
print(confusion_matrix(test_label,predict))

# Calcualte the precision and recall
precision = precision_score(test_label,predict)
recall = recall_score(test_label,predict)

print("The precision and recall are %f %f \n"%(precision,recall))