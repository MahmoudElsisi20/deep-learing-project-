from os import error
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier #model 1
from sklearn.model_selection import train_test_split #split for data
from sklearn import metrics
from sklearn.metrics import confusion_matrix #resluts
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier #model 2
from sklearn.metrics import accuracy_score 
from sklearn.neural_network import  MLPClassifier #model 3
import seaborn as sns

sns.set_style("whitegrid")

#----------------------read dataset----------------------------
col_names=['age','gender','TB','DB','alkphos','sgpt','sgot','TP','ALB','A_G','label']
dataset=pd.read_csv("indian_liver_patient_weka_dataset.csv",header=None,names=col_names)
dataset.head()
dataset['gender'].value_counts()

print(dataset.head())
print(dataset['gender'].value_counts())
print(dataset.describe())

 # make a correlation matrix a little prettier
corr_matrix = dataset.corr()
fig, ax = plt.subplots(figsize=(15, 15))
ax = sns.heatmap(corr_matrix,
annot=True,
linewidths=0.5,
fmt=".2f",
cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
#---------------------feature selection X,Y-----------------------------

feature_cols=['age','gender','TB','DB','alkphos','sgpt','sgot','TP','ALB','A_G']
x=dataset[feature_cols]
y=dataset.label
									
#-------------------------------Split data-------------------------------
x_train , x_test , y_train , y_test = train_test_split( x , y , test_size=0.3)#70 % train 30% test

#-----------------------------create a decision tree classifier-----------
Model1= DecisionTreeClassifier()
#train for decision tree
Model1=Model1.fit(x_train,y_train)
#predect response of training data
y_pred=Model1.predict(x_test)

accuracy_score_1= accuracy_score(y_pred,y_test)
print("------------Model1 Acurracy-------------")
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))
plt.figure(figsize=(25,10))
a=plot_tree(Model1,feature_names=feature_cols,class_names=['0','1'], filled=True, rounded= True,fontsize=14)
b=ConfusionMatrixDisplay.from_predictions( y_test, y_pred)
plt.show()
#------------------------KNeighborsClassifier-----------------------------

x_train1 , x_test1 , y_train1 , y_test1 = train_test_split( x , y , test_size=0.2)#80 % train 20%test

Model2= KNeighborsClassifier(n_neighbors=5)
Model2.fit(x_train1,y_train1)
y_pred=Model2.predict(x_test1)
accuracy_score_2= accuracy_score(y_pred,y_test1)

print("------------Model2 Acurracy-------------")
print("Accuracy:",metrics.accuracy_score(y_test1,y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test1,y_pred))
print(metrics.classification_report(y_test1,y_pred))
c=ConfusionMatrixDisplay.from_predictions( y_test1, y_pred)
plt.show()


#compare between 2 algorithms :

if accuracy_score_1 > accuracy_score_2 :
    print("DecisionTreeClassifier is best ")
else:
    print(" The  KNN is best ")
