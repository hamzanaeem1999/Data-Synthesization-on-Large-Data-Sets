import pandas as pd
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
import yaml
#Must Install below libraries in order to use Data Synthesizer
# !pip install DataSynthesizer
# !pip install scikit-learn==0.22.1

def load_config():
    with open('/content/drive/MyDrive/Config.yaml') as file:
        config = yaml.safe_load(file)
    return config
config = load_config()

mode=config['DS']['mode']
def DataSynthesizer(input_data):

  input_data.to_csv('temp.csv')
  input_data='temp.csv'

  if mode == 'correlated':    
    synthetic_data = f'/content/drive/MyDrive/synthetic_data.csv' 
    description_file = f'/content/drive/MyDrive/description.json'
  
    epsilon =  config['DS']['epsilon']

    degree_of_bayesian_network = config['DS']['value_of_K']

    num_tuples_to_generate = config['DS']['num_tuples_to_generate'] 
    
    input_df = pd.read_csv(input_data, skipinitialspace=True)

    if (num_tuples_to_generate==None):
      num_tuples_to_generate=len(input_df)
      
    describer = DataDescriber()
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file='temp.csv',k=degree_of_bayesian_network,epsilon=epsilon)
    describer.save_dataset_description_to_file(description_file)

    display_bayesian_network(describer.bayesian_network)

    #Generating the data
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
    generator.save_synthetic_data(synthetic_data)

    synthetic_df = pd.read_csv(synthetic_data)

    # Read attribute description from the dataset description file.
    attribute_description = read_json_file(description_file)['attribute_description']
    inspector = ModelInspector(input_df, synthetic_df, attribute_description)

    for attribute in synthetic_df.columns:
      inspector.compare_histograms(attribute)

    return (synthetic_data)

  elif mode == 'random':

    num_tuples_to_generate = config['DS']['num_tuples_to_generate']

    description_file = f'/content/drive/MyDrive/description.json'
    synthetic_data = f'/content/drive/MyDrive/synthetic_data.csv'

    describer = DataDescriber()
    describer.describe_dataset_in_random_mode('temp.csv')
    describer.save_dataset_description_to_file(description_file)

    generator = DataGenerator()
    generator.generate_dataset_in_random_mode(num_tuples_to_generate, description_file)
    generator.save_synthetic_data(synthetic_data)
    
    # Read both datasets using Pandas.
    input_df = pd.read_csv(input_data, skipinitialspace=True)
    synthetic_df = pd.read_csv(synthetic_data)
    # Read attribute description from the dataset description file.
    attribute_description = read_json_file(description_file)['attribute_description']
    inspector = ModelInspector(input_df, synthetic_df, attribute_description)
    
    for attribute in synthetic_df.columns:
      inspector.compare_histograms(attribute)

    return (synthetic_data)
