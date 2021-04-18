import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix,f1_score,classification_report
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib 
import seaborn as sn
import warnings
import yaml
import time
import sys
import os
warnings.filterwarnings("ignore")

#Applying Grid search on Random forest
def RandomForestGridSearch(train_data,train_labels):
  param_grid = {
      'bootstrap': [True],
      'max_depth': [None,80,100],
      'max_features': [2, 3],
      'min_samples_leaf': [1, 3, 5],
      'n_estimators': [100, 200, 300, 1000]
  }
  rf = RandomForestClassifier()
  RFclf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                            cv = 5, n_jobs = -1, verbose = 2)
  RFclf.fit(train_data, train_labels)
  RFclf.cv_results_
  df = pd.DataFrame(RFclf.cv_results_)
  return df


#Applying Grid search on Logistic Regression
def LogisticRegressionGridSearch(train_data,train_labels):
  param_grid = {
      'penalty' : ['l1','l2'],
      'C' : [0.5,1.0],
      'max_iter':[30,50,100],
      'solver' : ['liblinear','lbfgs','sag']
      }
  classifier = LogisticRegression()
  LRclf = GridSearchCV(classifier, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)
  LRclf.fit(train_data, train_labels)
  LRclf.cv_results_
  df = pd.DataFrame(LRclf.cv_results_)
  return df

#Applying Grid search on KNN
def KNNGridSearch(train_data,train_labels):
  param_grid = {
      'n_neighbors': [3,5,7],
      'algorithm':['auto','brute']
  }
  # Create a based model
  knn = KNeighborsClassifier()
  # Instantiate the grid search model
  KNNclf = GridSearchCV(estimator = knn, param_grid = param_grid, 
                            cv = 5, n_jobs = -1, verbose = 2)
  KNNclf.fit(train_data, train_labels)
  KNNclf.cv_results_
  df = pd.DataFrame(KNNclf.cv_results_)
  return df

#Applying Grid search on SVM
def SVMGridSearch(train_data,train_labels):
  SVMclf = GridSearchCV(svm.SVC(gamma='auto'), {
      'C': [1,10,20],
      'kernel': ['rbf','linear']
  }, cv=5, return_train_score=False)
  SVMclf.fit(train_data, train_labels)
  SVMclf.cv_results_
  df = pd.DataFrame(SVMclf.cv_results_)
  return df

#Config file
def load_config():
    with open('/content/drive/MyDrive/Config.yaml') as file:
        config = yaml.safe_load(file)
    return config
config = load_config()

#Logistic Regression model
dirName = 'models'
def LogisticRegressionClass(train_data,train_labels,test_data,test_labels,key,identifier,save_CSV=False):
  start_time = time.time()
  param=config[identifier]["Parameters"]["LogisticRegression parameters"]
  model = LogisticRegression(param["penalty"],param["dual"],param["C"],param["fit_intercept"],param["intercept_scaling"],param["class_weight"]) 

  model.fit(train_data,train_labels)
  predicted_labels = model.predict(test_data)

  if (save_CSV==True):
    pd.DataFrame(zip(test_labels,predicted_labels),columns=['actual labels','predicted labels']).to_csv('/content/drive/MyDrive/CSVs/OriginalCensusLRmodel.csv')

  LR_train_accuracy=model.score(train_data,train_labels)
  LR_test_accuracy=model.score(test_data,test_labels)

  #Train Accuracy
  print("Model Score of Logistic Regression on training data",LR_train_accuracy)

  #Test accuracy
  print("Model Score of Logistic Regression  on testing data",LR_test_accuracy)

  #Saving the model into specified directory
  if not os.path.exists(dirName):
      os.mkdir(dirName)

  joblib.dump(model, f'models/{key}_Logistic_Regression_Model.pkl')
  
  print("--- Logistic Regression takes %s seconds ---" % (time.time() - start_time))
#Random ForestApplying
def RandomForest(train_data,train_labels,test_data,test_labels,key,identifier,save_CSV=False):
  start_time = time.time()
  param=config[identifier]["Parameters"]["RandomForest parameters"]
  model = RandomForestClassifier(param["n_estimators"],param["criterion"])
 
  print(param["n_estimators"])
  
  model.fit(train_data,train_labels)
  predicted_labels = model.predict(test_data)

  if (save_CSV==True):
    pd.DataFrame(zip(test_labels,predicted_labels),columns=['actual labels','predicted labels']).to_csv('/content/drive/MyDrive/CSVs/OriginalCensusLRmodel.csv')

  RFF_train_accuracy=model.score(train_data,train_labels)
  RFF_test_accuracy=model.score(test_data,test_labels)

  #Train accuracy
  print("Model Score of Random Forest on training data",RFF_train_accuracy)

  #Test accuracy
  print("Model Score of Random Forest on testing data",RFF_test_accuracy)
  
  if not os.path.exists(dirName):
    os.mkdir(dirName)
  
  #Saving the model into specified directory
  joblib.dump(model,f'models/{key}_Random_Forest_Model.pkl') 
  print("--- Random Forest takes %s seconds ---" % (time.time() - start_time))

#K-Nearest Neighbours applying
def KNN(train_data,train_labels,test_data,test_labels,key,identifier,save_CSV=False):
  start_time = time.time()
  param=config[identifier]["Parameters"]["kNN parameters"]
  model = KNeighborsClassifier(param['n_neighbors'],param['weights'],param['algorithm'])
  model.fit(train_data,train_labels)
  predicted_labels = model.predict(test_data)

  if (save_CSV==True):
    pd.DataFrame(zip(test_labels,predicted_labels),columns=['actual labels','predicted labels']).to_csv('/content/drive/MyDrive/CSVs/OriginalCensusLRmodel.csv')

  KNN_train_accuracy=model.score(train_data,train_labels)
  KNN_test_accuracy=model.score(test_data,test_labels)

  #Train accuracy
  print("Model Score of KNN on training data",KNN_train_accuracy)

  #Test accuracy
  print("Model Score of KNN on testing data",KNN_test_accuracy)
  
  if not os.path.exists(dirName):
    os.mkdir(dirName)
  
  #Saving the model into specified directory
  joblib.dump(model,f'models/{key}_KNN_Model.pkl')
  
  print("--- KNN takes %s seconds ---" % (time.time() - start_time))

#Applying SVM
def SVM(train_data,train_labels,test_data,test_labels,key,identifier,save_CSV=False):
  start_time = time.time()
  param=config[identifier]["Parameters"]["SVM parameters"]
  model = SVC(param['C'],param['kernel'],param['degree'],param['gamma'])  
  model.fit(train_data,train_labels)
  predicted_labels = model.predict(test_data)

  if (save_CSV==True):
    pd.DataFrame(zip(test_labels,predicted_labels),columns=['actual labels','predicted labels']).to_csv('/content/drive/MyDrive/CSVs/OriginalCensusLRmodel.csv')

  SVM_train_accuracy=model.score(train_data,train_labels)
  SVM_test_accuracy=model.score(test_data,test_labels)

  #Train accuracy
  print("Model Score of SVM  on training data",SVM_train_accuracy)

  #Test accuracy
  print("Model Score of SVM on testing data",SVM_test_accuracy)
  
  if not os.path.exists(dirName):
    os.mkdir(dirName)
  
  #Saving the model into specified directory
  joblib.dump(model,f'models/{key}_Support_Vector_Machine_Model.pkl') 
  
  print("--- SVM takes %s seconds ---" % (time.time() - start_time))
