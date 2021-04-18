from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


# Read CSV
def read_csv(csv1,csv2,csv3,csv4,sdv_csv1,sdv_csv2,sdv_csv3,sdv_csv4,ds_csv1,ds_csv2,ds_csv3,ds_csv4,synthpop_csv1,synthpop_csv2,synthpop_csv3,synthpop_csv4):
  csv1,csv2,csv3,csv4,sdv_csv1,sdv_csv2,sdv_csv3,sdv_csv4,ds_csv1,ds_csv2,ds_csv3,ds_csv4,synthpop_csv1,synthpop_csv2,synthpop_csv3,synthpop_csv4 = pd.read_csv(csv1).drop('Unnamed: 0',axis=1),pd.read_csv(csv2).drop('Unnamed: 0',axis=1),pd.read_csv(csv3).drop('Unnamed: 0',axis=1),pd.read_csv(csv4).drop('Unnamed: 0',axis=1),pd.read_csv(sdv_csv1).drop('Unnamed: 0',axis=1),pd.read_csv(sdv_csv2).drop('Unnamed: 0',axis=1),pd.read_csv(sdv_csv3).drop('Unnamed: 0',axis=1),pd.read_csv(sdv_csv4).drop('Unnamed: 0',axis=1),pd.read_csv(ds_csv1).drop('Unnamed: 0',axis=1),pd.read_csv(ds_csv2).drop('Unnamed: 0',axis=1),pd.read_csv(ds_csv3).drop('Unnamed: 0',axis=1),pd.read_csv(ds_csv4).drop('Unnamed: 0',axis=1),pd.read_csv(synthpop_csv1).drop('Unnamed: 0',axis=1),pd.read_csv(synthpop_csv2).drop('Unnamed: 0',axis=1),pd.read_csv(synthpop_csv3).drop('Unnamed: 0',axis=1),pd.read_csv(synthpop_csv4).drop('Unnamed: 0',axis=1)
  return csv1,csv2,csv3,csv4,sdv_csv1,sdv_csv2,sdv_csv3,sdv_csv4,ds_csv1,ds_csv2,ds_csv3,ds_csv4,synthpop_csv1,synthpop_csv2,synthpop_csv3,synthpop_csv4

# Get Metrices for binary 
def get_metrices_binary(csv):
  accuracy = accuracy_score(csv['actual labels'], csv['predicted labels'])
  precision = precision_score(csv['actual labels'], csv['predicted labels'])
  recall = recall_score(csv['actual labels'], csv['predicted labels'])
  f_score = f1_score(csv['actual labels'], csv['predicted labels'])
  return accuracy, precision, recall, f_score

# Get Metrices for multiclass
def get_metrices_multi(csv):
  accuracy = accuracy_score(csv['actual labels'], csv['predicted labels'])
  precision = precision_score(csv['actual labels'], csv['predicted labels'],average='micro')
  recall = recall_score(csv['actual labels'], csv['predicted labels'],average='macro')
  f_score = f1_score(csv['actual labels'], csv['predicted labels'],average='weighted')
  return accuracy, precision, recall, f_score

# Displays the charts of models and their confusion matrices
def Statistics(csv1,csv2,csv3,csv4,sdv_csv1,sdv_csv2,sdv_csv3,sdv_csv4,ds_csv1,ds_csv2,ds_csv3,ds_csv4,synthpop_csv1,synthpop_csv2,synthpop_csv3,synthpop_csv4,datasetname):
  if datasetname =='Census':

    accuracy_csv1, precision_csv1, recall_csv1, f_score_csv1 = get_metrices_binary(csv1)
    accuracy_csv2, precision_csv2, recall_csv2, f_score_csv2 = get_metrices_binary(csv2)
    accuracy_csv3, precision_csv3, recall_csv3, f_score_csv3 = get_metrices_binary(csv3)
    accuracy_csv4, precision_csv4, recall_csv4, f_score_csv4 = get_metrices_binary(csv4)

    sdv_accuracy_csv1, sdv_precision_csv1, sdv_recall_csv1, sdv_f_score_csv1 = get_metrices_binary(sdv_csv1)
    sdv_accuracy_csv2, sdv_precision_csv2, sdv_recall_csv2, sdv_f_score_csv2 = get_metrices_binary(sdv_csv2)
    sdv_accuracy_csv3, sdv_precision_csv3, sdv_recall_csv3, sdv_f_score_csv3 = get_metrices_binary(sdv_csv3)
    sdv_accuracy_csv4, sdv_precision_csv4, sdv_recall_csv4, sdv_f_score_csv4 = get_metrices_binary(sdv_csv4)

    ds_accuracy_csv1, ds_precision_csv1, ds_recall_csv1, ds_f_score_csv1 = get_metrices_binary(ds_csv1)
    ds_accuracy_csv2, ds_precision_csv2, ds_recall_csv2, ds_f_score_csv2 = get_metrices_binary(ds_csv2)
    ds_accuracy_csv3, ds_precision_csv3, ds_recall_csv3, ds_f_score_csv3 = get_metrices_binary(ds_csv3)
    ds_accuracy_csv4, ds_precision_csv4, ds_recall_csv4, ds_f_score_csv4 = get_metrices_binary(ds_csv4)

    synthpop_accuracy_csv1, synthpop_precision_csv1, synthpop_recall_csv1, synthpop_f_score_csv1 = get_metrices_binary(synthpop_csv1)
    synthpop_accuracy_csv2, synthpop_precision_csv2, synthpop_recall_csv2, synthpop_f_score_csv2 = get_metrices_binary(synthpop_csv2)
    synthpop_accuracy_csv3, synthpop_precision_csv3, synthpop_recall_csv3, synthpop_f_score_csv3 = get_metrices_binary(synthpop_csv3)
    synthpop_accuracy_csv4, synthpop_precision_csv4, synthpop_recall_csv4, synthpop_f_score_csv4 = get_metrices_binary(synthpop_csv4)
  
  elif datasetname=='Forest':
    accuracy_csv1, precision_csv1, recall_csv1, f_score_csv1 = get_metrices_multi(csv1)
    accuracy_csv2, precision_csv2, recall_csv2, f_score_csv2 = get_metrices_multi(csv2)
    accuracy_csv3, precision_csv3, recall_csv3, f_score_csv3 = get_metrices_multi(csv3)
    accuracy_csv4, precision_csv4, recall_csv4, f_score_csv4 = get_metrices_multi(csv4)

    sdv_accuracy_csv1, sdv_precision_csv1, sdv_recall_csv1, sdv_f_score_csv1 = get_metrices_multi(sdv_csv1)
    sdv_accuracy_csv2, sdv_precision_csv2, sdv_recall_csv2, sdv_f_score_csv2 = get_metrices_multi(sdv_csv2)
    sdv_accuracy_csv3, sdv_precision_csv3, sdv_recall_csv3, sdv_f_score_csv3 = get_metrices_multi(sdv_csv3)
    sdv_accuracy_csv4, sdv_precision_csv4, sdv_recall_csv4, sdv_f_score_csv4 = get_metrices_multi(sdv_csv4)

    ds_accuracy_csv1, ds_precision_csv1, ds_recall_csv1, ds_f_score_csv1 = get_metrices_multi(ds_csv1)
    ds_accuracy_csv2, ds_precision_csv2, ds_recall_csv2, ds_f_score_csv2 = get_metrices_multi(ds_csv2)
    ds_accuracy_csv3, ds_precision_csv3, ds_recall_csv3, ds_f_score_csv3 = get_metrices_multi(ds_csv3)
    ds_accuracy_csv4, ds_precision_csv4, ds_recall_csv4, ds_f_score_csv4 = get_metrices_multi(ds_csv4)

    synthpop_accuracy_csv1, synthpop_precision_csv1, synthpop_recall_csv1, synthpop_f_score_csv1 = get_metrices_multi(synthpop_csv1)
    synthpop_accuracy_csv2, synthpop_precision_csv2, synthpop_recall_csv2, synthpop_f_score_csv2 = get_metrices_multi(synthpop_csv2)
    synthpop_accuracy_csv3, synthpop_precision_csv3, synthpop_recall_csv3, synthpop_f_score_csv3 = get_metrices_multi(synthpop_csv3)
    synthpop_accuracy_csv4, synthpop_precision_csv4, synthpop_recall_csv4, synthpop_f_score_csv4 = get_metrices_multi(synthpop_csv4)
  else:
    print("Invalid dataset name")


  print("Random Forest Chart")
  fig = go.Figure(data=[
    go.Bar(name='Original data Accuracy', x=['Accuracy'], y=[accuracy_csv1]),
    go.Bar(name='SDV_Accuracy', x=['Accuracy'],y=[sdv_accuracy_csv1]),
    go.Bar(name='DS_Accuracy', x=['Accuracy'], y=[ds_accuracy_csv1]),
    go.Bar(name='Synthpop Accuracy', x=['Accuracy'], y=[synthpop_accuracy_csv1]),

    go.Bar(name='Original data Precision', x=['Precision'], y=[precision_csv1]),
    go.Bar(name='SDV_Precision', x=['Precision'],y= [sdv_precision_csv1]),
    go.Bar(name='DS_Precision', x=['Precision'], y=[ds_precision_csv1]),
    go.Bar(name='Synthpop Precision', x=['Precision'], y=[synthpop_precision_csv1]),

    go.Bar(name='Original data Recall', x=['Recall'], y=[recall_csv1]),
    go.Bar(name='SDV_Recall', x=['Recall'],y=[sdv_recall_csv1]),
    go.Bar(name='DS_Recall', x=['Recall'], y=[ds_recall_csv1]),
    go.Bar(name='Synthpop Recall', x=['Recall'], y=[synthpop_recall_csv1]),

    go.Bar(name='Original data F_Measure', x=['F measure'], y=[f_score_csv1]),
    go.Bar(name='SDV_F_Measure', x=['F measure'],y= [sdv_f_score_csv1]),
    go.Bar(name='DS_F_Measure', x=['F measure'], y=[ds_f_score_csv1]),
    go.Bar(name='Synthpop F_Measure', x=['F measure'], y=[synthpop_f_score_csv1]),
  ])
  # Change the bar mode
  fig.update_layout(barmode='group')
  fig.show()

  print("Logistic Regression Chart")
  fig = go.Figure(data=[
  go.Bar(name='Original data  Accuracy', x=['Accuracy'], y=[accuracy_csv2]),
  go.Bar(name=' SDV_Accuracy', x=['Accuracy'],y=[sdv_accuracy_csv2]),
  go.Bar(name=' DS_Accuracy', x=['Accuracy'], y=[ds_accuracy_csv2]),
  go.Bar(name=' Synthpop Accuracy', x=['Accuracy'], y=[synthpop_accuracy_csv2]),

  go.Bar(name='Original data  Precision', x=['Precision'], y=[precision_csv2]),
  go.Bar(name='SDV_Precision', x=['Precision'],y= [sdv_precision_csv2]),
  go.Bar(name='DS_Precision', x=['Precision'], y=[ds_precision_csv2]),
  go.Bar(name='Synthpop Precision', x=['Precision'], y=[synthpop_precision_csv2]),

  go.Bar(name='Original data Recall', x=['Recall'], y=[recall_csv2]),
  go.Bar(name='SDV_Recall', x=['Recall'],y=[sdv_recall_csv2]),
  go.Bar(name='DS_Recall', x=['Recall'], y=[ds_recall_csv2]),
  go.Bar(name='Synthpop Recall', x=['Recall'], y=[synthpop_recall_csv2]),

  go.Bar(name='Original data F_Measure', x=['F measure'], y=[f_score_csv2]),
  go.Bar(name='SDV_F_Measure', x=['F measure'],y= [sdv_f_score_csv2]),
  go.Bar(name='DS_F_Measure', x=['F measure'], y=[ds_f_score_csv2]),
  go.Bar(name='Synthpop F_Measure', x=['F measure'], y=[synthpop_f_score_csv2]),
  ])
  # Change the bar mode
  fig.update_layout(barmode='group')
  fig.show()

  print("K Nearest Neighbor Chart")

  fig = go.Figure(data=[
  go.Bar(name='Original data Accuracy', x=['Accuracy'], y=[accuracy_csv3]),
  go.Bar(name='SDV_Accuracy', x=['Accuracy'],y=[sdv_accuracy_csv3]),
  go.Bar(name='DS_Accuracy', x=['Accuracy'], y=[ds_accuracy_csv3]),
  go.Bar(name='Synthpop Accuracy', x=['Accuracy'], y=[synthpop_accuracy_csv3]),

  go.Bar(name='Original data Precision', x=['Precision'], y=[precision_csv3]),
  go.Bar(name='SDV_Precision', x=['Precision'],y= [sdv_precision_csv3]),
  go.Bar(name='DS_Precision', x=['Precision'], y=[ds_precision_csv3]),
  go.Bar(name='Synthpop Precision', x=['Precision'], y=[synthpop_precision_csv3]),

  go.Bar(name='Original data Recall', x=['Recall'], y=[recall_csv3]),
  go.Bar(name='SDV_Recall', x=['Recall'],y=[sdv_recall_csv3]),
  go.Bar(name='DS_Recall', x=['Recall'], y=[ds_recall_csv3]),
  go.Bar(name='Synthpop Recall', x=['Recall'], y=[synthpop_recall_csv3]),

  go.Bar(name='Original data F_Measure', x=['F measure'], y=[f_score_csv3]),
  go.Bar(name='SDV_F_Measure', x=['F measure'],y= [sdv_f_score_csv3]),
  go.Bar(name='DS_F_Measure', x=['F measure'], y=[ds_f_score_csv3]),
  go.Bar(name='Synthpop F_Measure', x=['F measure'], y=[synthpop_f_score_csv3]),
  ])
  # Change the bar mode
  fig.update_layout(barmode='group')
  fig.show()

  print("Suppor Vector Machine Chart")
  fig = go.Figure(data=[
  go.Bar(name='Original data Accuracy', x=['Accuracy'], y=[accuracy_csv4]),
  go.Bar(name='SDV_Accuracy', x=['Accuracy'],y=[sdv_accuracy_csv4]),
  go.Bar(name='DS_Accuracy', x=['Accuracy'], y=[ds_accuracy_csv4]),
  go.Bar(name='Synthpop Accuracy', x=['Accuracy'], y=[synthpop_accuracy_csv4]),

  go.Bar(name='Original data Precision', x=['Precision'], y=[precision_csv4]),
  go.Bar(name='SDV_Precision', x=['Precision'],y= [sdv_precision_csv4]),
  go.Bar(name='DS_Precision', x=['Precision'], y=[ds_precision_csv4]),
  go.Bar(name='Synthpop Precision', x=['Precision'], y=[synthpop_precision_csv4]),

  go.Bar(name='Original data Recall', x=['Recall'], y=[recall_csv4]),
  go.Bar(name='SDV_Recall', x=['Recall'],y=[sdv_recall_csv4]),
  go.Bar(name='DS_Recall', x=['Recall'], y=[ds_recall_csv4]),
  go.Bar(name='Synthpop Recall', x=['Recall'], y=[synthpop_recall_csv4]),

  go.Bar(name='Original data F_Measure', x=['F measure'], y=[f_score_csv4]),
  go.Bar(name='SDV_F_Measure', x=['F measure'],y= [sdv_f_score_csv4]),
  go.Bar(name='DS_F_Measure', x=['F measure'], y=[ds_f_score_csv4]),
  go.Bar(name='Synthpop F_Measure', x=['F measure'], y=[synthpop_f_score_csv4]),
  ])
  fig.update_layout(barmode='group')
  fig.show()
  
  # Change the bar mode
  
  # Confusion matrix
  print("Confusion matrices of Original Data")
  cm1=confusion_matrix(csv1['actual labels'],csv1['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(cm1, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()
 
  cm2=confusion_matrix(csv2['actual labels'],csv2['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(cm2, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()

  cm3=confusion_matrix(csv3['actual labels'],csv3['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(cm3, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()

  cm4=confusion_matrix(csv4['actual labels'],csv4['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(cm4, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()
  
  print("Confusion matrices of Synthetic Data Vault")
  sdv_cm1=confusion_matrix(sdv_csv1['actual labels'],sdv_csv1['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(sdv_cm1, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()

 
  sdv_cm2=confusion_matrix(sdv_csv2['actual labels'],sdv_csv2['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(sdv_cm2, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()


  sdv_cm3=confusion_matrix(sdv_csv3['actual labels'],sdv_csv3['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(sdv_cm3, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()


  sdv_cm4=confusion_matrix(sdv_csv4['actual labels'],sdv_csv4['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(sdv_cm4, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()


  print("Confusion matrices of Data Synthesizer")
  ds_cm1=confusion_matrix(ds_csv1['actual labels'],ds_csv1['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(ds_cm1, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()

 
  ds_cm2=confusion_matrix(ds_csv2['actual labels'],ds_csv2['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(ds_cm2, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()


  ds_cm3=confusion_matrix(ds_csv3['actual labels'],ds_csv3['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(ds_cm3, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()


  ds_cm4=confusion_matrix(ds_csv4['actual labels'],ds_csv4['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(ds_cm4, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()


  
  print("Confusion matrices of Synthpop Data")
  synthpop_cm1=confusion_matrix(synthpop_csv1['actual labels'],synthpop_csv1['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(synthpop_cm1, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()
 
  synthpop_cm2=confusion_matrix(synthpop_csv2['actual labels'],synthpop_csv2['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(synthpop_cm2, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()

  synthpop_cm3=confusion_matrix(synthpop_csv3['actual labels'],synthpop_csv3['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(synthpop_cm3, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()

  synthpop_cm4=confusion_matrix(synthpop_csv4['actual labels'],synthpop_csv4['predicted labels'])
  plt.figure(figsize=(10,7))
  sn.heatmap(synthpop_cm4, annot=True,cmap='Blues', fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.show()

  return (accuracy_csv1,sdv_accuracy_csv1,ds_accuracy_csv1,
  synthpop_accuracy_csv1,accuracy_csv2,sdv_accuracy_csv2,
  ds_accuracy_csv2,synthpop_accuracy_csv2,accuracy_csv3,
  sdv_accuracy_csv3,ds_accuracy_csv3,synthpop_accuracy_csv3,
  accuracy_csv4,sdv_accuracy_csv4,ds_accuracy_csv4,synthpop_accuracy_csv4)
  
  

#Plotting charts
def get_census_charts(data,s_data,d_data,sp_data):
  def x_y(data,col):
    temp={}
    for i in list(data[col]):
      if i not in temp.keys():
        temp[i]=1
      else:
        temp[i]=temp[i]+1
    return temp
  for cols in data.columns:
    temp1=x_y(data,cols)
    temp2=x_y(s_data,cols)
    temp3=x_y(d_data,cols)
    temp4=x_y(sp_data,cols)
    fig, ax = plt.subplots(figsize=(15,15))
    ax.bar(temp1.keys(), temp1.values(), color="None",  edgecolor="red", linewidth=2, label='Original Data')
    ax.bar(temp2.keys(), temp2.values(), color="None",  edgecolor="green", linewidth=2, label='Synthetic Data Vault')
    ax.bar(temp3.keys(), temp3.values(), color="None",  edgecolor="yellow", linewidth=2, label='Data Synthesizer')
    ax.bar(temp4.keys(), temp4.values(), color="None",  edgecolor="blue", linewidth=2, label='Synthpop')
    ax.legend()
    plt.show()

#census data csv's
def get_forest_charts(data,s_data,d_data,sp_data):
  #Plotting charts
  def x_y(data,col):
    temp={}
    for i in list(data[col]):
      if i not in temp.keys():
        temp[i]=1
      else:
        temp[i]=temp[i]+1
    return temp
  for cols in data.columns:
    temp1=x_y(data,cols)
    temp2=x_y(s_data,cols)
    temp3=x_y(d_data,cols)
    temp4=x_y(sp_data,cols)
    fig, ax = plt.subplots(figsize=(15,15))
    ax.bar(temp1.keys(), temp1.values(), color="None",  edgecolor="red", linewidth=2, label='Original Data')
    ax.bar(temp2.keys(), temp2.values(), color="None",  edgecolor="green", linewidth=2, label='Synthetic Data Vault')
    ax.bar(temp3.keys(), temp3.values(), color="None",  edgecolor="yellow", linewidth=2, label='Data Synthesizer')
    ax.bar(temp4.keys(), temp4.values(), color="None",  edgecolor="blue", linewidth=2, label='Synthpop')
    ax.legend()
    plt.show()

def get_heat_maps(data,s_data,d_data,sp_data):
  plt.figure(figsize=(15,13))
  plt.title('Heat Map of original Data', y=1.05, size=20)
  sn.heatmap(data.corr(),linewidths=0.1,vmax=1.0,square=False)
  plt.figure(figsize=(15,13))
  plt.title('Heat Map of SDV data', y=1.05, size=20)
  sn.heatmap(s_data.corr(),linewidths=0.1,vmax=1.0,square=False)
  plt.figure(figsize=(15,13))
  plt.title('Heat Map of Data Synthesizer', y=1.05, size=20)
  sn.heatmap(d_data.corr(),linewidths=0.1,vmax=1.0,square=False)
  plt.figure(figsize=(15,13))
  plt.title('Heat Map of Synthpop', y=1.05, size=20)
  sn.heatmap(sp_data.corr(),linewidths=0.1,vmax=1.0,square=False)

def get_box_plots(data,s_data,d_data,sp_data):
  for column in data:
    plt.boxplot([data[column], s_data[column],d_data[column],sp_data[column]],labels=['Original Data ','SDV','Data Synthesizer','Synthpop'], autorange=True)
    plt.show()
 
    
    
  
