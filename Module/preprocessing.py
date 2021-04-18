from sklearn.preprocessing import LabelEncoder 
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split



#Preprocessing
def preprocessing(data,target):

  #Total number of attributes
  print("Number of attributes are",len(data.count())) 
  print(data[target].value_counts())

  #checking balance/imbalance dataset
  target_count = data[target].value_counts()
  target_count.plot(kind='bar', title='Count (target)')

  #Dropping the duplicates
  data=data.drop_duplicates(keep=False)

  #checking null values
  print("Null values are",data.isnull().values.any())

  #Shuffling
  data = data.sample(frac = 1)
  return data

  #Label encoding
def label_encode(df):
    col_names=[i for i in df if df[i].dtypes=='object']
    labelencoder = LabelEncoder()
    for cols in col_names:
        df[cols]=labelencoder.fit_transform(df[cols])
    return df



def Balancing(train_data,train_labels):
  smote=SMOTE(sampling_strategy='minority')
  train_data,train_labels=smote.fit_sample(train_data,train_labels)
  unique, counts = np.unique(train_labels, return_counts=True)
  print("Dataset balanced using SMOTE",unique,counts)  
  return (train_data,train_labels)



def scaling_data(scalingdata):
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(scalingdata)
  return scaled_data



def preprocessing_data(data,target,dataset_name):
  if dataset_name=='Census':
    data=preprocessing(data,target)
    data=label_encode(data)

    #Train Data
    train_data=data.drop(target,axis=1)
    #Train Labels
    train_labels=data[target].values

    train_data,train_labels=Balancing(train_data,train_labels)
    train_data=scaling_data(train_data) 
    return train_data,train_labels

  else:
    data=preprocessing(data,target)
    data=label_encode(data)

    #Train Data
    train_data=data.drop(target,axis=1)
    #Train Labels
    train_labels=data[target].values

    train_data,train_labels=Balancing(train_data,train_labels)
    train_data=scaling_data(train_data)
  
    train_data,test_data,train_labels,test_labels=train_test_split(train_data,train_labels,random_state=42,test_size=0.20)
    print("Train data shape:",train_data.shape,"Train labels shape:",train_labels.shape)
    print("Test  data shape:",test_data.shape," Test labels shape:",test_labels.shape)
    return (train_data,train_labels,test_data,test_labels)
