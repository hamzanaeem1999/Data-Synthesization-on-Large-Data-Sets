from sdv import SDV
import pandas as pd
import yaml
import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder 
from sdv.tabular import CopulaGAN,GaussianCopula,CTGAN,TVAE 
from sdv.metrics.tabular import SVCDetection, KSTest
from sdv.evaluation import evaluate

# Must Install below libraries in order to use Synthetic Data Vault
# !pip install sdv
# !pip install numpy==1.20.0

global _params
def load_config():
    with open('/content/drive/MyDrive/Config.yaml') as file:
        config = yaml.safe_load(file)
    return config
config = load_config()

scores=[]

def HyperParameterTuning(data):
  hyper_params={}
  distributions_list=['univariate','parametric','bounded','semi_bounded','parametric_bounded','parametric_semi_bounded','gaussian','gamma','beta','student_t','truncated_gaussian']
  for i in data.columns.values:
    for distribution in distributions_list:
      model=GaussianCopula(field_distributions={i:distribution})
      model.fit(data)
      new_data = model.sample(len(data))
      kstest=KSTest.compute(data, new_data)
      svcdetection=SVCDetection.compute(data,new_data)
      scores.append([kstest,svcdetection])
      if i not in hyper_params:
        hyper_params[i]={distribution:[set(new_data[i].to_list())==set(data[i].to_list()),kstest,svcdetection]}  
      else:
        hyper_params[i]=[hyper_params[i],{distribution:[set(new_data[i].to_list())==set(data[i].to_list()),kstest,svcdetection]}]
  return hyper_params


def fetch_best(hyper_params):
  final={}
  for i in hyper_params:
    record={}
    for distributions in hyper_params[i]:
      temp=distributions.values()

      if (list(temp)[0][0])==True:
        record[list(distributions.keys())[0]]=list(temp)[0][1]
        # print(record)

    if record:
        sorted_=sorted(record.items(), key=lambda x: x[1], reverse=True)
        final[i]=sorted_[0][0]
    else:
      for i in hyper_params:
        record={}
        for distributions in hyper_params[i]:
          temp=distributions.values()
          record[list(distributions.keys())[0]]=list(temp)[0][1]
          sorted_=sorted(record.items(), key=lambda x: x[1], reverse=True)
          final[i]=sorted_[0][0]
  return final  


# There are four methods for generating data - > TVAE(), CopulaGAN(), GaussianCopula(), CTGAN() . All specified Below
def SyntheticDataVault(data,save_CSV=False,path='/content/drive'):
  
  hyper_params=HyperParameterTuning(data)
  _params=fetch_best(hyper_params)

  models={'GaussianCopula':GaussianCopula(field_distributions=_params),
      'CopulaGAN':CopulaGAN(field_distributions=_params),
      'CTGAN':CTGAN(),
      'TVAE':TVAE()}


  samples=config['SDV']['sample']
  param=config["SDV"]
  model=models[param["model"]]
  print(model)
  model.fit(data)
  if (samples==None):
    samples=len(data)
  
  if (isinstance(model,type(GaussianCopula()))):
    model_name='GaussianCopula.pkl'
  
  elif (isinstance(model,type(CopulaGAN()))):
    model_name='CopulaGAN.pkl'  

  elif (isinstance(model,type(CTGAN()))):
    model_name='CTGAN.pkl'

  else:
    model_name='TVAE.pkl'

  model.save(path+f'{model_name}')
  new_data = model.sample(samples)
  if save_CSV==True:
    new_data.to_csv('/content/drive/SDVCensus.csv')
  
  metric = evaluate(data,new_data,metrics=['KSTest','SVCDetection','ContinuousKLDivergence','KSTestExtended'],aggregate=False)
  return (new_data,metric)