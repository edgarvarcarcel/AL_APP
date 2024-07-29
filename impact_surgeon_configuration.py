# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:26:13 2024

@author: Edgar David

This Script is used for see the impact of different values of anastomotic configuration
and surgeon experience in different samples

"""

# Utils
import pandas as pd
import pickle as pkl
import numpy as np
from itertools import product

# Models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import torch as th
from torch import Tensor
from torch.nn.parameter import Parameter


# Data imputation
from sklearn.impute import KNNImputer

print('Libraries loaded')

###############################################################################
# Functions for the Pytorch Model

# Define fully conected model
class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.5):
        super(FullyConnectedModel, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_prob = dropout_prob

        layers = []
        # Create the hidden layers with linear, batch normalization, and dropout
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(p=self.dropout_prob))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer with softmax activation
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim = 1))

        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
# Define function to save pytorch model for early stopping
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
# Define function to load best early stopping pytorch model to continue with the evaluation
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

###############################################################################
# Load original data
input_path = r'data'
input_filename = r'\reduced_op_UK_merged_data_final_21062024.csv'
input_data = pd.read_csv(input_path + input_filename , decimal = '.' , sep = ';')
print('File loaded -->' , input_path + input_filename)

# Load MinMaxScaler
path_scaler = r'models'
filename_scaler = r'\Scaler.sav'
scaler = pkl.load(open(path_scaler + filename_scaler , 'rb'))
print('File loaded -->' , path_scaler + filename_scaler)

# Load KNN Imputer
path_knn = r'models'
filename_knn = r'\KNN_Imputer.sav'
imputer = pkl.load(open(path_knn + filename_knn , 'rb'))
print('File loaded -->' , path_knn + filename_knn)


# Load Pytorch Model
input_size = 26 # Value from the train notebook
hidden_sizes = [100, 100, 50, 100, 100]
output_size = 2
dropout_prob = 0.1
model = FullyConnectedModel(input_size, hidden_sizes, output_size, dropout_prob)
resume(model, r'models\Pytorch_Model.pth')
print('Pytorch Model Loaded')

# Define columns
selected_features = ['age',
 'preoperative_albumin_level',
 'bmi',
 'preoperative_hemoglobin_level',
 'pack_years',
 'sex',
 'neoadjuvant_therapy',
 'preoperative_use_of_immunosuppressive_drugs',
 'tnf_alpha_inhib',
 'emergency_surgery',
 'approach',
 'preoperative_steroid_use',
 'preoperative_nsaids_use',
 'active_smoking',
 'liver_metastasis_at_time_of_anastomosis',
 'BIHistoryOfDiabetes',
 'preoperative_blood_transfusion',
 'perforation',
 'anastomotic_technique',
 'surgeon_experience',
 'conversion',
 'anastomotic_configuration',
 'protective_stomy',
 'BIHistoryOfIschaemicHeartDisease',
 'alcohol_abuse',
 'asa_score',
 'anastomotic_leackage']
cat_columns = ['sex',
  'neoadjuvant_therapy',
  'preoperative_use_of_immunosuppressive_drugs',
  'tnf_alpha_inhib',
  'emergency_surgery',
  'approach',
  'preoperative_steroid_use',
  'preoperative_nsaids_use',
  'active_smoking',
  'liver_metastasis_at_time_of_anastomosis',
  'BIHistoryOfDiabetes',
  'preoperative_blood_transfusion',
  'perforation',
  'anastomotic_technique',
  'surgeon_experience',
  'conversion',
  'anastomotic_configuration',
  'protective_stomy',
  'BIHistoryOfIschaemicHeartDisease',
  'alcohol_abuse',
  'asa_score']
target = ['anastomotic_leackage']

# Exclude 30% of more nans
percent_of_missing = pd.DataFrame((input_data.replace(-1 , np.nan).isnull().sum() / input_data.shape[0]).sort_values(ascending = False))
print(percent_of_missing)
to_drop_missing_columns = percent_of_missing[percent_of_missing[0] > 0.3].index.tolist()
print('Columns deleted by missing values:' , to_drop_missing_columns)
to_drop_columns = ['record_id' , 'data_group']
df = input_data.drop(columns = to_drop_columns)
df = pd.DataFrame(imputer.fit_transform(df) , columns = df.columns.tolist())


# Apply numeric scaler
aux_scaler_columns = scaler.feature_names_in_.tolist()
df_num = df[aux_scaler_columns]
df_num = pd.DataFrame(scaler.transform(df_num) , columns = df_num.columns.tolist())
df[aux_scaler_columns] = df_num.copy()

# Extract posible values of configuration and surgeon experience
posible_values_configuration = df['anastomotic_configuration'].astype(int).unique().tolist() 
posible_values_surgeon = df['surgeon_experience'].astype(int).unique().tolist()
posible_values = list(product(posible_values_surgeon , posible_values_configuration))
df = df[selected_features]

# Create a base of X vectors equals to the total of combinations
data_to_predict = {}
probabilities = {}

for i in range(len(posible_values)):
    print('-' * 50)
    print('Creating data for combination (surgeon , configuration -->)' , posible_values[i])
    data_to_predict[i] = df.drop(columns = target)
    # Set values for surgeon and configuration
    data_to_predict[i]['surgeon_experience'] = posible_values[i][0]
    data_to_predict[i]['anastomotic_configuration'] = posible_values[i][1]
    
    # Convert data to Pytorch tensors
    pytorch_tensor = torch.FloatTensor(data_to_predict[i].values)
    # Predict probability
    model.eval()
    with torch.no_grad():
        prediction = model(pytorch_tensor)
        prediction = prediction.squeeze().numpy()
    probabilities[i] = pd.DataFrame(prediction.copy() , columns = ['No AL' , 'Yes AL'])[['Yes AL']]

# Summary of results for see changing of probabilities
summary_probabilities = pd.DataFrame()
for i in range(len(posible_values)):
    summary_probabilities = pd.concat([summary_probabilities ,
                                       pd.DataFrame(probabilities[i].values , columns = ['surgeon = ' + str(posible_values[i][0]) + ' - Configuration = ' + str(posible_values[i][1])])],
                                      axis = 1)
# Export results
export = pd.concat([input_data,
                    summary_probabilities] , axis = 1)
export.to_excel(r'results\Impact_Surgeon_Configuration_v2.xlsx')




