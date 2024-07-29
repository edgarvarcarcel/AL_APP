# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 23:11:01 2024

"""
###############################################################################
# Load libraries

# App
import streamlit as st
from streamlit_option_menu import option_menu

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

# Format of numbers


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
###############################################################################
# Section when the app initialize and load the required information
@st.cache_data() # We use this decorator so this initialization doesn't run every time the user change into the page
def initialize_app():
    # Load Examples
    example_path = r'data'
    example_filename = r'/examples.xlsx'
    example_data = pd.read_excel(example_path + example_filename)
    print('File loaded -->' , example_path + example_filename)
    
    # Load original data
    input_path = r'data'
    input_filename = r'/reduced_op_UK_merged_data_final_21062024.csv'
    input_data = pd.read_csv(input_path + input_filename , decimal = '.' , sep = ';')
    print('File loaded -->' , input_path + input_filename)
    
    # Load MinMaxScaler
    path_scaler = r'models'
    filename_scaler = r'/Scaler.sav'
    scaler = pkl.load(open(path_scaler + filename_scaler , 'rb'))
    print('File loaded -->' , path_scaler + filename_scaler)
    
    # Load KNN Imputer
    path_knn = r'models'
    filename_knn = r'/KNN_Imputer.sav'
    imputer = pkl.load(open(path_knn + filename_knn , 'rb'))
    print('File loaded -->' , path_knn + filename_knn)
   
    # Load Pytorch Model
    input_size = 26 # Value from the train notebook
    hidden_sizes = [100, 100, 50, 100, 100]
    output_size = 2
    dropout_prob = 0.1
    model = FullyConnectedModel(input_size, hidden_sizes, output_size, dropout_prob)
    resume(model, r'models/Pytorch_Model.pth')
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
     'asa_score']
    cat_columns = ['sex',  'neoadjuvant_therapy',  'preoperative_use_of_immunosuppressive_drugs',  'tnf_alpha_inhib',
                  'emergency_surgery',  'approach',  'preoperative_steroid_use',  'preoperative_nsaids_use',  'active_smoking',
                  'liver_metastasis_at_time_of_anastomosis',  'BIHistoryOfDiabetes',  'preoperative_blood_transfusion',  'perforation',
                  'anastomotic_technique',  'surgeon_experience',  'conversion',  'anastomotic_configuration',  'protective_stomy',
                  'BIHistoryOfIschaemicHeartDisease',  'alcohol_abuse',  'asa_score']
    target = ['anastomotic_leackage']
    
    print('App Initialized correctly!')
    
    return input_data , scaler , imputer , model , selected_features , cat_columns , target , example_data

# Funcion to process user input
def parser_user_input(dataframe_input , scaler , model , selected_features , target):
    # Define dictionary for the categorical features
    dictionary_categorical_features = {'sex' : {'Male' : 2,
                                                'Female' : 1},
                                       'neoadjuvant_therapy' : {'Yes' : 1,
                                                                'No' : 0},
                                       'preoperative_use_of_immunosuppressive_drugs' : {'Yes' : 1,
                                                                    'No' : 0},
                                       'tnf_alpha_inhib' : {'Yes' : 1,
                                                            'No' : 0},
                                       'emergency_surgery' : {'Yes' : 1,
                                                              'No' : 0},
                                       'approach' : {'1: Laparoscopic' : 1 ,
                                                     '2: Robotic' : 2 ,
                                                     '3: Open to open' : 3,
                                                     '4: Conversion to open' : 4,
                                                     '5: Conversion to laparoscopy' : 5,
                                                     '6: Transanal (ta TME , TATA , TAMIS)' : 6},
                                       'preoperative_steroid_use' : {'Yes' : 1,
                                                        'No' : 0},
                                       'preoperative_nsaids_use' : {'Yes' : 1,
                                                                    'No' : 0},
                                       'active_smoking' : {'Yes' : 1,
                                                           'No' : 0},
                                       'liver_metastasis_at_time_of_anastomosis' : {'Yes' : 2,
                                                                                    'No' : 1},
                                       'BIHistoryOfDiabetes' : {'Yes' : 1,
                                                                   'No' : 0},
                                       'preoperative_blood_transfusion' : {'Yes' : 1,
                                                                           'No' : 0},
                                       'perforation' : {'Yes' : 1,
                                                        'No' : 0},
                                       'surgeon_experience' : {'Yes' : 1,
                                                               'No' : 0},
                                       'conversion' : {'Yes' : 1,
                                                       'No' : 0},
                                       'protective_stomy' : {'Yes' : 2,
                                                             'No' : 1},
                                       'BIHistoryOfIschaemicHeartDisease' : {'Yes' : 1,
                                                                                   'No' : 0},
                                       'alcohol_abuse' : {'Yes' : 2,
                                                          'No' : 1},
                                       'asa_score' : {'1: Healthy Person' : 1,
                                                '2: Mild Systemic disease' : 2,
                                                '3: Severe syatemic disease' : 3,
                                                '4: Severe systemic disease that is a constan threat to life' : 4,
                                                '5: Moribund person' : 5,
                                                '6: Unkonw' : 6},
                                       'anastomotic_technique' : {'1: Stapler' : 1,
                                                                  '2: Hand-sewn' : 2,
                                                                  '3: Stapler and Hand-sewn' : 3,
                                                                  '4: Unknown' : 4}}
    
    # Split cat and numeric columns
    aux_cat_columns = [i for i in dataframe_input.columns.tolist() if i in dictionary_categorical_features.keys()]
    aux_num_columns = [i for i in dataframe_input.columns.tolist() if i not in aux_cat_columns]
    df_cat = dataframe_input[aux_cat_columns]
    df_num = dataframe_input[aux_num_columns]
    
    # Create syntetic columns for scaler
    df_num['dosage_of_steroids'] = -1
    df_num['preoperative_crp_level'] = -1
    df_num['preoperative_leukocyte_count'] = -1
    
    aux_scaler_columns = scaler.feature_names_in_.tolist()
    df_num = df_num[aux_scaler_columns]
    

    # Apply scaler to numeric features
    df_num = pd.DataFrame(scaler.transform(df_num) , columns = df_num.columns.tolist())
    
    # Encode categorical features
    df_cat_encoded = pd.DataFrame()
    for i in df_cat.columns.tolist():
        df_cat_encoded[i] = df_cat[i].map(dictionary_categorical_features[i])
    # Add configuration value
    probabilities = {}
    values_of_anastomotic_configuration = {'End to End' : 1,
                                           'Side to End' : 2,
                                           'Side to Side' : 3,
                                           'End to Side' : 4}
    values_of_surgeon_experience = {'Yes' : 2,
                                    'No' : 1}
    posible_values = list(product(values_of_surgeon_experience , values_of_anastomotic_configuration))
    
    for i in range(len(posible_values)):
        print('#' * 50)
        print('Making prediction usung surgeon experience = ' , posible_values[i][0] , 'configuration =' , posible_values[i][1])
        df_cat_encoded['surgeon_experience'] = dictionary_categorical_features['surgeon_experience'][posible_values[i][0]]
        df_cat_encoded['anastomotic_configuration'] = values_of_anastomotic_configuration[posible_values[i][1]]
        df_hasher = pd.concat([df_num,
                               df_cat_encoded] , axis = 1)
        df_hasher = df_hasher[selected_features]
        # Convert data to Pytorch tensors
        pytorch_tensor = torch.FloatTensor(df_hasher.values)
        # Predict probability
        model.eval()
        with torch.no_grad():
            prediction = model(pytorch_tensor)
            prediction = prediction.squeeze().numpy()
        probabilities[i] = pd.DataFrame(prediction.copy() , columns = ['Probabilities']).T
        probabilities[i] = probabilities[i][1]
    # Create a dataframe with the results as a matrix
    df = pd.DataFrame(index=values_of_anastomotic_configuration.keys(), columns=values_of_surgeon_experience.keys())
    # Fill the DataFrame with the product of their corresponding values
    for config_key, config_value in values_of_anastomotic_configuration.items():
        for exp_key, exp_value in values_of_surgeon_experience.items():
            df.at[config_key, exp_key] = (config_value, exp_value)
    
    for i in range(len(posible_values)):
        aux_configuration = posible_values[i][1]
        aux_surgeon = posible_values[i][0]
        df.loc[aux_configuration , aux_surgeon] = probabilities[i].values[0]
    
    # Format of prediction
    formatted_df = df.style.format({"Predictions": "{:.6f}".format})
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')
    
    with col2:
        st.write(formatted_df)
    
    with col3:
        st.write(' ')

    return df


###############################################################################
# Page configuration
st.set_page_config(
    page_title="AL Prediction App"
)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Initialize app
input_data , scaler , imputer , model , selected_features , cat_columns , target , example_data = initialize_app()

# Option Menu configuration
with st.sidebar:
    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['Home' , 'Prediction' , 'Examples'],
        icons = ['house' , 'book' , 'book'],
        menu_icon = 'cast',
        default_index = 0,
        orientation = 'Vertical')
    
######################
# Home page layout
######################
if selected == 'Home':
    st.title('Anastomotic Leackage App')
    st.markdown("""
    This app contains 2 sections which you can access from the horizontal menu above.\n
    The sections are:\n
    Home: The main page of the app.\n
    **Prediction:** On this section you can select the patients information and
    the models iterate over all posible anastomotic configuration and surgeon experience for suggesting
    the best option.\n
    **Examples:** On this section you can use pre-defined values for a patient
    to see the predictions.
    """)
    
###############################################################################
# Prediction page layout
if selected == 'Prediction':
    st.title('Prediction Section')
    #st.markdown("""
    #Please input the following CPVs:.\n
    #•	Age in years.\n
    #•   Preoperative Albumin Lebel.\n
    #•   BMI Preoperative.\n
    #•   Preoperative Hemoglobin level.\n
    #•   Pack per years.\n
    #•   Sex.\n
    #•   Type of neoadjuntvant therapy.\n
    #•   Preoperative use of inmunosuppressive drugs.\n
    #•   TNF alpha inhib.\n
    #•   Emergency Surgery.\n
    #•   Approach.\n
    #•   Preoperative steroid use.\n
    #•   Preoperative nsaids use.\n
    #•   Active Smoking.\n
    #•   Liver metastasis at time of anastomosis.\n
    #•   BI History of Diabetes.\n
    #•   Preoperative blood transfusion.\n
    #•   Perforation.\n
    #•   Surgeon Experience.\n
    #•   Conversion.\n
    #•   Anastomotic technique.\n
    #•   Protective Stomy.\n
    #•   BI History of Ischaemic Hearth Disease.\n
    #•   Alcohol abuse.\n
    #•	ASA Score.\n
    #""")
    st.subheader("Description")
    st.subheader("To predict Anastomotic Leackage, you need to follow the steps below:")
    st.markdown("""
    1. Enter clinical parameters of patient on the left side bar.
    2. Press the "Predict" button and wait for the result.
    """)
    st.markdown("""
    This model predicts the probabilities of AL for each type of configuration and surgeon experience.
    """)
    # Sidebar layout
    st.sidebar.title("Patiens Info")
    st.sidebar.subheader("Please choose parameters")
    
    # Input features
    age = st.sidebar.slider("Age:", min_value = 18, max_value = 100,step = 1)
    preoperative_albumin_level = st.sidebar.slider("Preoperative Albumin Level:", min_value = 0.0, max_value = 30.0,step = 0.1)
    bmi = st.sidebar.slider("Preoperative BMI:", min_value = 18, max_value = 50,step = 1)
    preoperative_hemoglobin_level = st.sidebar.slider("Preoperative Hemoglobin Level:", min_value = 0.0, max_value = 30.0,step = 0.1)
    pack_years = st.sidebar.slider("Pack years:", min_value = 0.0, max_value = 55.0,step = 0.1)
    sex = st.sidebar.selectbox('Gender', ('Male' , 'Female'))
    neoadjuvant_therapy = st.sidebar.selectbox('Neoadjuvant Therapy', ('Yes' , 'No'))
    preoperative_use_of_immunosuppressive_drugs = st.sidebar.selectbox('Use of Inmunosuppressive Drugs', ('Yes' , 'No'))
    tnf_alpha_inhib = st.sidebar.selectbox('TNF Alpha Inhib', ('Yes' , 'No'))
    emergency_surgery = st.sidebar.selectbox('Emergency Surgery', ('Yes' , 'No'))
    approach = st.sidebar.selectbox('Approach', ('1: Laparoscopic' ,
                                                 '2: Robotic' , '3: Open to open',
                                                 '4: Conversion to open',
                                                 '5: Conversion to laparoscopy',
                                                 '6: Transanal (ta TME , TATA , TAMIS)'))
    preoperative_steroid_use = st.sidebar.selectbox('Steroid Use', ('Yes' , 'No'))
    preoperative_nsaids_use = st.sidebar.selectbox('Nsaids Use', ('Yes' , 'No'))
    active_smoking = st.sidebar.selectbox('Active Smoking', ('Yes' , 'No'))
    liver_metastasis_at_time_of_anastomosis = st.sidebar.selectbox('Liver metastasis at time of anastomosis', ('Yes' , 'No'))
    BIHistoryOfDiabetes = st.sidebar.selectbox('BI History of Diabetes', ('Yes' , 'No'))
    preoperative_blood_transfusion = st.sidebar.selectbox('Preoperative Blood Transfusion', ('Yes' , 'No'))
    perforation = st.sidebar.selectbox('Perforation', ('Yes' , 'No'))
    #surgeon_experience = st.sidebar.selectbox('Surgeon Experience', ('Yes' , 'No'))
    conversion = st.sidebar.selectbox('Conversion to laparoscopic', ('Yes' , 'No'))
    anastomotic_technique = st.sidebar.selectbox('Anastomotic Technique', ('1: Stapler' ,
                                                                           '2: Hand-sewn',
                                                                           '3: Stapler and Hand-sewn',
                                                                           '4: Unknown'))
    protective_stomy = st.sidebar.selectbox('Protective Stomy', ('Yes' , 'No'))
    BIHistoryOfIschaemicHeartDisease = st.sidebar.selectbox('BI History of Ischaemic Heart Disease', ('Yes' , 'No'))
    alcohol_abuse = st.sidebar.selectbox('Alcohol Abuse', ('Yes' , 'No'))
    asa_score = st.sidebar.selectbox('ASA Score', ('1: Healthy Person',
                                             '2: Mild Systemic disease',
                                             '3: Severe syatemic disease',
                                             '4: Severe systemic disease that is a constan threat to life',
                                             '5: Moribund person',
                                             '6: Unkonw'))
    
    dataframe_input = pd.DataFrame({'sex' : [sex],
                                    'neoadjuvant_therapy' : [neoadjuvant_therapy],
                                       'preoperative_use_of_immunosuppressive_drugs' : [preoperative_use_of_immunosuppressive_drugs],
                                       'tnf_alpha_inhib' : [tnf_alpha_inhib],
                                       'emergency_surgery' : [emergency_surgery],
                                       'approach' : [approach],
                                       'preoperative_steroid_use' : [preoperative_steroid_use],
                                       'preoperative_nsaids_use' : [preoperative_nsaids_use],
                                       'active_smoking' : [active_smoking],
                                       'liver_metastasis_at_time_of_anastomosis' : [liver_metastasis_at_time_of_anastomosis],
                                       'BIHistoryOfDiabetes' : [BIHistoryOfDiabetes],
                                       'preoperative_blood_transfusion' : [preoperative_blood_transfusion],
                                       'perforation' :[perforation],
                                       #'surgeon_experience' : [surgeon_experience],
                                       'conversion' : [conversion],
                                       'protective_stomy' : [protective_stomy],
                                       'BIHistoryOfIschaemicHeartDisease' : [BIHistoryOfIschaemicHeartDisease],
                                       'alcohol_abuse' : [alcohol_abuse],
                                       'anastomotic_technique' : [anastomotic_technique],
                                       'asa_score' : [asa_score],
                                       'age' : [age],
                                       'preoperative_albumin_level' : [preoperative_albumin_level],
                                       'bmi' : [bmi],
                                       'preoperative_hemoglobin_level' : [preoperative_hemoglobin_level],
                                       'pack_years' : [pack_years]})
    # Parser input and make predictions
    predict_button = st.button('Predict')
    if predict_button:
        predictions = parser_user_input(dataframe_input , scaler , model , selected_features , target)
        #st.dataframe(predictions)
        
##############################################################################
# Example page layout
# Prediction page layout
if selected == 'Examples':
    # Define difcionary from values to options
    dictionary_categorical_features = {'sex' : {'Male' : 2,
                                                'Female' : 1},
                                       'neoadjuvant_therapy' : {'Yes' : 1,
                                                                'No' : 0},
                                       'preoperative_use_of_immunosuppressive_drugs' : {'Yes' : 1,
                                                                    'No' : 0},
                                       'tnf_alpha_inhib' : {'Yes' : 1,
                                                            'No' : 0},
                                       'emergency_surgery' : {'Yes' : 1,
                                                              'No' : 0},
                                       'approach' : {'1: Laparoscopic' : 1 ,
                                                     '2: Robotic' : 2 ,
                                                     '3: Open to open' : 3,
                                                     '4: Conversion to open' : 4,
                                                     '5: Conversion to laparoscopy' : 5,
                                                     '6: Transanal (ta TME , TATA , TAMIS)' : 6},
                                       'preoperative_steroid_use' : {'Yes' : 1,
                                                        'No' : 0},
                                       'preoperative_nsaids_use' : {'Yes' : 1,
                                                                    'No' : 0},
                                       'active_smoking' : {'Yes' : 1,
                                                           'No' : 0},
                                       'liver_metastasis_at_time_of_anastomosis' : {'Yes' : 2,
                                                                                    'No' : 1},
                                       'BIHistoryOfDiabetes' : {'Yes' : 1,
                                                                   'No' : 0},
                                       'preoperative_blood_transfusion' : {'Yes' : 1,
                                                                           'No' : 0},
                                       'perforation' : {'Yes' : 1,
                                                        'No' : 0},
                                       'surgeon_experience' : {'Yes' : 1,
                                                               'No' : 0},
                                       'conversion' : {'Yes' : 1,
                                                       'No' : 0},
                                       'protective_stomy' : {'Yes' : 2,
                                                             'No' : 1},
                                       'BIHistoryOfIschaemicHeartDisease' : {'Yes' : 1,
                                                                                   'No' : 0},
                                       'alcohol_abuse' : {'Yes' : 2,
                                                          'No' : 1},
                                       'asa_score' : {'1: Healthy Person' : 1,
                                                '2: Mild Systemic disease' : 2,
                                                '3: Severe syatemic disease' : 3,
                                                '4: Severe systemic disease that is a constan threat to life' : 4,
                                                '5: Moribund person' : 5,
                                                '6: Unkonw' : 6},
                                       'anastomotic_technique' : {'1: Stapler' : 1,
                                                                  '2: Hand-sewn' : 2,
                                                                  '3: Stapler and Hand-sewn' : 3,
                                                                  '4: Unknown' : 4}}
    inverted_dictionary = {}

    for key, value in dictionary_categorical_features.items():
        inverted_dictionary[key] = {v: k for k, v in value.items()}
    
    st.title('Examples Section')
    #st.markdown("""
    #Please input the following CPVs:.\n
    #•	Age in years.\n
    #•   Preoperative Albumin Lebel.\n
    #•   BMI Preoperative.\n
    #•   Preoperative Hemoglobin level.\n
    #•   Pack per years.\n
    #•   Sex.\n
    #•   Type of neoadjuntvant therapy.\n
    #•   Preoperative use of inmunosuppressive drugs.\n
    #•   TNF alpha inhib.\n
    #•   Emergency Surgery.\n
    #•   Approach.\n
    #•   Preoperative steroid use.\n
    #•   Preoperative nsaids use.\n
    #•   Active Smoking.\n
    #•   Liver metastasis at time of anastomosis.\n
    #•   BI History of Diabetes.\n
    #•   Preoperative blood transfusion.\n
    #•   Perforation.\n
    #•   Surgeon Experience.\n
    #•   Conversion.\n
    #•   Anastomotic technique.\n
    #•   Protective Stomy.\n
    #•   BI History of Ischaemic Hearth Disease.\n
    #•   Alcohol abuse.\n
    #•	ASA Score.\n
    #""")
    st.subheader("Description")
    st.subheader("This Section Show different examples for prediction, follow the instructions:")
    st.markdown("""
    1. Select the example row you want to use.
    2. Press the "Predict" button and wait for the result.
    """)
    st.markdown("""
    This model predicts the probabilities of AL for each type of configuration and surgeon experience.
    """)
    
    patient_row = st.slider("Patient's row:", min_value = 0, max_value = example_data.shape[0] - 1,step = 1)
    aux_patient = pd.DataFrame(example_data.iloc[patient_row]).T
    # Sidebar layout
    st.sidebar.title("Patiens Info")
    st.sidebar.subheader("Parameters are setting based on example you choose")
    
    # Input features
    age = st.sidebar.slider("Age:", 
                            min_value = 18, 
                            max_value = 100,
                            step = 1 , 
                            value = aux_patient['age'].values[0])
    preoperative_albumin_level = st.sidebar.slider("Preoperative Albumin Level:",
                                                   min_value = 0.0,
                                                   max_value = 30.0,
                                                   step = 0.1 ,
                                                   value = float(aux_patient['preoperative_albumin_level'].values[0]))
    bmi = st.sidebar.slider("Preoperative BMI:",
                            min_value = 18.0,
                            max_value = 50.0,
                            step = 0.1,
                            value = float(aux_patient['bmi'].values[0]))
    preoperative_hemoglobin_level = st.sidebar.slider("Preoperative Hemoglobin Level:",
                                                      min_value = 0.0,
                                                      max_value = 30.0,
                                                      step = 0.1,
                                                      value = float(aux_patient['preoperative_hemoglobin_level'].values[0]))
    pack_years = st.sidebar.slider("Pack years:",
                                   min_value = 0.0,
                                   max_value = 55.0,
                                   step = 0.1,
                                   value = float(aux_patient['pack_years'].values[0]))
    sex = st.sidebar.selectbox('Gender',
                               ('Male' , 'Female'),
                               index = ('Male' , 'Female').index(inverted_dictionary['sex'][aux_patient['sex'].values[0]]))
    neoadjuvant_therapy = st.sidebar.selectbox('Neoadjuvant Therapy', ('Yes' , 'No'))
    preoperative_use_of_immunosuppressive_drugs = st.sidebar.selectbox('Use of Inmunosuppressive Drugs',
                                                                       ('Yes' , 'No'),
                                                                       index = ('Yes' , 'No').index(inverted_dictionary['preoperative_use_of_immunosuppressive_drugs'][aux_patient['preoperative_use_of_immunosuppressive_drugs'].values[0]]))
    tnf_alpha_inhib = st.sidebar.selectbox('TNF Alpha Inhib', 
                                           ('Yes' , 'No'),
                                           index = ('Yes' , 'No').index(inverted_dictionary['tnf_alpha_inhib'][aux_patient['tnf_alpha_inhib'].values[0]]))
    emergency_surgery = st.sidebar.selectbox('Emergency Surgery',
                                             ('Yes' , 'No'),
                                             index = ('Yes' , 'No').index(inverted_dictionary['emergency_surgery'][aux_patient['emergency_surgery'].values[0]]))
    approach = st.sidebar.selectbox('Approach',
                                    ('1: Laparoscopic' ,
                                    '2: Robotic' , '3: Open to open',
                                    '4: Conversion to open',
                                    '5: Conversion to laparoscopy',
                                    '6: Transanal (ta TME , TATA , TAMIS)'),
                                    index = ('1: Laparoscopic' ,
                                    '2: Robotic' , '3: Open to open',
                                    '4: Conversion to open',
                                    '5: Conversion to laparoscopy',
                                    '6: Transanal (ta TME , TATA , TAMIS)').index(inverted_dictionary['approach'][aux_patient['approach'].values[0]]))
    preoperative_steroid_use = st.sidebar.selectbox('Steroid Use',
                                                    ('Yes' , 'No'),
                                                    index = ('Yes' , 'No').index(inverted_dictionary['preoperative_steroid_use'][aux_patient['preoperative_steroid_use'].values[0]]))
    preoperative_nsaids_use = st.sidebar.selectbox('Nsaids Use',
                                                   ('Yes' , 'No'),
                                                   index = ('Yes' , 'No').index(inverted_dictionary['preoperative_nsaids_use'][aux_patient['preoperative_nsaids_use'].values[0]]))
    active_smoking = st.sidebar.selectbox('Active Smoking',
                                          ('Yes' , 'No'),
                                          index = ('Yes' , 'No').index(inverted_dictionary['active_smoking'][aux_patient['active_smoking'].values[0]]))
    liver_metastasis_at_time_of_anastomosis = st.sidebar.selectbox('Liver metastasis at time of anastomosis',
                                                                   ('Yes' , 'No'),
                                                                   index = ('Yes' , 'No').index(inverted_dictionary['liver_metastasis_at_time_of_anastomosis'][aux_patient['liver_metastasis_at_time_of_anastomosis'].values[0]]))
    BIHistoryOfDiabetes = st.sidebar.selectbox('BI History of Diabetes',
                                               ('Yes' , 'No'),
                                               index = ('Yes' , 'No').index(inverted_dictionary['BIHistoryOfDiabetes'][aux_patient['BIHistoryOfDiabetes'].values[0]]))
    preoperative_blood_transfusion = st.sidebar.selectbox('Preoperative Blood Transfusion', 
                                                          ('Yes' , 'No'),
                                                          index = ('Yes' , 'No').index(inverted_dictionary['preoperative_blood_transfusion'][aux_patient['preoperative_blood_transfusion'].values[0]]))
    perforation = st.sidebar.selectbox('Perforation',
                                       ('Yes' , 'No'),
                                       index = ('Yes' , 'No').index(inverted_dictionary['perforation'][aux_patient['perforation'].values[0]]))
    #surgeon_experience = st.sidebar.selectbox('Surgeon Experience', ('Yes' , 'No'))
    conversion = st.sidebar.selectbox('Conversion to laparoscopic',
                                      ('Yes' , 'No'),
                                      index = ('Yes' , 'No').index(inverted_dictionary['conversion'][aux_patient['conversion'].values[0]]))
    anastomotic_technique = st.sidebar.selectbox('Anastomotic Technique',
                                                 ('1: Stapler' ,
                                                '2: Hand-sewn',
                                                '3: Stapler and Hand-sewn',
                                                '4: Unknown'),
                                                index = ('1: Stapler' ,
                                               '2: Hand-sewn',
                                               '3: Stapler and Hand-sewn',
                                               '4: Unknown').index(inverted_dictionary['anastomotic_technique'][aux_patient['anastomotic_technique'].values[0]]))
    protective_stomy = st.sidebar.selectbox('Protective Stomy',
                                            ('Yes' , 'No'),
                                            index = ('Yes' , 'No').index(inverted_dictionary['protective_stomy'][aux_patient['protective_stomy'].values[0]]))
    BIHistoryOfIschaemicHeartDisease = st.sidebar.selectbox('BI History of Ischaemic Heart Disease',
                                                            ('Yes' , 'No'),
                                                            index = ('Yes' , 'No').index(inverted_dictionary['BIHistoryOfIschaemicHeartDisease'][aux_patient['BIHistoryOfIschaemicHeartDisease'].values[0]]))
    alcohol_abuse = st.sidebar.selectbox('Alcohol Abuse',
                                         ('Yes' , 'No'),
                                         index = ('Yes' , 'No').index(inverted_dictionary['alcohol_abuse'][aux_patient['alcohol_abuse'].values[0]]))
    asa_score = st.sidebar.selectbox('ASA Score',
                                     ('1: Healthy Person',
                                    '2: Mild Systemic disease',
                                    '3: Severe syatemic disease',
                                    '4: Severe systemic disease that is a constan threat to life',
                                    '5: Moribund person',
                                    '6: Unkonw'),
                                    index = ('1: Healthy Person',
                                   '2: Mild Systemic disease',
                                   '3: Severe syatemic disease',
                                   '4: Severe systemic disease that is a constan threat to life',
                                   '5: Moribund person',
                                   '6: Unkonw').index(inverted_dictionary['asa_score'][aux_patient['asa_score'].values[0]]))
    
    dataframe_input = pd.DataFrame({'sex' : [sex],
                                    'neoadjuvant_therapy' : [neoadjuvant_therapy],
                                       'preoperative_use_of_immunosuppressive_drugs' : [preoperative_use_of_immunosuppressive_drugs],
                                       'tnf_alpha_inhib' : [tnf_alpha_inhib],
                                       'emergency_surgery' : [emergency_surgery],
                                       'approach' : [approach],
                                       'preoperative_steroid_use' : [preoperative_steroid_use],
                                       'preoperative_nsaids_use' : [preoperative_nsaids_use],
                                       'active_smoking' : [active_smoking],
                                       'liver_metastasis_at_time_of_anastomosis' : [liver_metastasis_at_time_of_anastomosis],
                                       'BIHistoryOfDiabetes' : [BIHistoryOfDiabetes],
                                       'preoperative_blood_transfusion' : [preoperative_blood_transfusion],
                                       'perforation' :[perforation],
                                       #'surgeon_experience' : [surgeon_experience],
                                       'conversion' : [conversion],
                                       'protective_stomy' : [protective_stomy],
                                       'BIHistoryOfIschaemicHeartDisease' : [BIHistoryOfIschaemicHeartDisease],
                                       'alcohol_abuse' : [alcohol_abuse],
                                       'anastomotic_technique' : [anastomotic_technique],
                                       'asa_score' : [asa_score],
                                       'age' : [age],
                                       'preoperative_albumin_level' : [preoperative_albumin_level],
                                       'bmi' : [bmi],
                                       'preoperative_hemoglobin_level' : [preoperative_hemoglobin_level],
                                       'pack_years' : [pack_years]})
    # Parser input and make predictions
    predict_button = st.button('Predict')
    if predict_button:
        predictions = parser_user_input(dataframe_input , scaler , model , selected_features , target)
        #st.dataframe(predictions)
        