#!/usr/bin/env python
# coding: utf-8

# ## Supervised Learning (Sec. 001)
# #### Group 2</br>
# 
# - Mehreen Abdul Rahman</br>
# - Bruno Cantanhede Morgado</br>
# - Ankit Mehra</br>
# - Ayesha Mohammed Azim Shaikh</br>
# - Prashant Sharma</br>

# In[270]:


# Initial imports
import numpy as np
import pandas as pd
import time
import math
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os


# In[271]:


# Helper Function to plot confusion Matrix
def plot_confusion_matrix(confusion_matrix, y_limit: list, model:str, color_map: str):
    #Plot the confusion Matrix
    fig, ax = plt.subplots(figsize=(10,6))
    title = f'Confusion matrix: {model}'
    # create heatmap
    sns.heatmap(confusion_matrix, annot = True, cmap = color_map ,fmt='g')
    ax.xaxis.set_label_position("top")
    ax.set_ylim(y_limit)
    ax.xaxis.set_ticklabels(['0','1'])
    ax.yaxis.set_ticklabels(['0','1'])
    plt.title(title, fontsize=20, pad=10.0)
    plt.ylabel('Actual label', fontsize='large')
    plt.xlabel('Predicted label', fontsize='large')
    plt.tight_layout()


# In[272]:


# Load the KSi Dataset
df_ksi = pd.read_csv('KSI.csv')


# In[273]:


# Display all the columns
pd.set_option('display.max_columns', None)


# In[274]:


# Inspect the initial five rows of the dataset
df_ksi.head()


# In[275]:


# Inspect the last 5 rows
df_ksi.tail()


# In[276]:


# Get a summary of the dataset
df_ksi.info()


# `The info() method suggests that the dataset is integral and that there are no missing values
# We see below that i isn't exactly the case`

# In[277]:


# Null values are informed as strings
df_ksi['OFFSET'][0]


# In[278]:


type(df_ksi['OFFSET'][0])


# In[279]:


# Set all "<Null>" strings to NaN
df_ksi = df_ksi.replace('<Null>', np.nan, regex = True)


# In[280]:


df_ksi.info()


# In[281]:


sns.set(rc={"figure.figsize":(28, 15)})
sns.heatmap(df_ksi.isna(),yticklabels=False,cbar=False,cmap='viridis')
plt.xticks(rotation = 45)
plt.show()


# In[282]:


# Move target class to the last position in the dataframe
new_cols = ['X', 'Y', 'INDEX_', 'ACCNUM', 'YEAR', 'DATE', 'TIME', 'HOUR', 'STREET1',
       'STREET2', 'OFFSET', 'ROAD_CLASS', 'DISTRICT', 'WARDNUM', 'DIVISION',
       'LATITUDE', 'LONGITUDE', 'LOCCOORD', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY',
       'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE', 'INVAGE',
       'INJURY', 'FATAL_NO', 'INITDIR', 'VEHTYPE', 'MANOEUVER', 'DRIVACT',
       'DRIVCOND', 'PEDTYPE', 'PEDACT', 'PEDCOND', 'CYCLISTYPE', 'CYCACT',
       'CYCCOND', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
       'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV',
       'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'POLICE_DIVISION', 'HOOD_ID',
       'NEIGHBOURHOOD', 'ObjectId', 'ACCLASS']
df_ksi = df_ksi.reindex(columns = new_cols)


# In[283]:


df_ksi.columns


# In[284]:


# Number of columns in the dataset
num_columns = len(df_ksi.columns)


# In[285]:


# Number of columns that contain at least one missing value
num_missing_val_columns = len(df_ksi.isna().sum()[df_ksi.isna().sum()>0])


# ### Columns and their respective numbers of missing values

# In[286]:


df_ksi.isna().sum()[df_ksi.isna().sum()>0]


# In[289]:


# Proportion of columns with ate least one missing value
print(f'{round((num_missing_val_columns / num_columns)*100,2)}% have at least one missing value')


# In[290]:


df_ksi.drop(['X', 'Y', 'INDEX_'], axis = 1, inplace = True)


# `Drop X and Y since they are a different scale for latitude and longitude
# INDEX_ is also dropped for it's lack of statistical value.`

# In[291]:


df_ksi.head()


# `The DATE column is split into DAY, MONTH, AND YEAR. The latter is dropped`

# In[292]:


df_ksi['DATE'] = pd.to_datetime(df_ksi['DATE'], format = '%Y/%m/%d %H:%M:%S')


# In[293]:


df_ksi.insert(1, 'MONTH', df_ksi['DATE'].dt.month)


# In[294]:


df_ksi.insert(2, 'DAY', df_ksi['DATE'].dt.day)


# In[295]:


df_ksi.drop(['YEAR', 'DATE', 'HOUR'], axis = 1, inplace = True)


# In[297]:


df_ksi.OFFSET.value_counts()


# In[298]:


df_ksi.OFFSET.isna().sum() /len(df_ksi.OFFSET)


# In[299]:


# Drop ACCNUM and OFFSET to reduce model complexity. 
df_ksi.drop(['ACCNUM','OFFSET'], axis = 1, inplace = True)


# In[300]:


# Switch all NaN values in Road_Class
df_ksi.ROAD_CLASS.replace(to_replace = np.nan, value = 'Road Type Unavailable', inplace = True)


# In[301]:


df_ksi.ROAD_CLASS.value_counts()


# In[302]:


df_ksi.DISTRICT.value_counts()


# In[303]:


# Merge 'Toronto East York' with 'Toronto and East York' 
df_ksi.DISTRICT.replace(to_replace = 'Toronto East York', value = 'Toronto and East York', inplace = True)


# In[304]:


# Replace NaN
df_ksi.DISTRICT.replace(to_replace = np.nan, value = 'District_Not_Informed', inplace = True)


# In[305]:


# Drop 'WARDNUM' and 'DIVISION,' since their intrinsic values overlap with
# other columns that better inidcate location of the accident
df_ksi.drop(['WARDNUM', 'DIVISION'], axis = 1, inplace = True)


# In[306]:


df_ksi.POLICE_DIVISION.unique()


# In[307]:


# Pearson's Correlation
sns.heatmap(df_ksi.corr(method = 'pearson'), annot = True, cmap = 'viridis')
plt.show()


# In[37]:


df_ksi.columns


# In[308]:


# Analyze the similarites between these two columns
df_ksi[['ACCLOC', 'LOCCOORD']]


# In[309]:


# ACCLOC has a much larger number of NaN than LOCCOORD
df_ksi[['ACCLOC', 'LOCCOORD']].isna().sum()


# `Whenever LOCCORD has an NaN value and ACCLOC does note, we'll fill
# the first column with the values from the latter`

# In[310]:


#Fill NaN values in LOCCOORD with correspondent values in ACCLOC
df_ksi['LOCCOORD'].fillna(df_ksi.ACCLOC[df_ksi['LOCCOORD'].isna()], inplace=True)


# In[311]:


# 16 LOCCORD's NaN values filled with values from ACCLOC
df_ksi.LOCCOORD.isna().sum()


# In[312]:


df_ksi.LOCCOORD.value_counts()


# In[313]:


# Replace NaN values
df_ksi.LOCCOORD.replace(to_replace = np.nan, value = 'Loccoord_Not_Informed', inplace = True)


# In[314]:


df_ksi.drop('ACCLOC', axis = 1, inplace = True)


# In[315]:


df_ksi.TRAFFCTL.isna().sum()


# In[316]:


# Traffic control seems to be relevant for our predictive model
df_ksi.TRAFFCTL.value_counts()


# In[317]:


# Replace NaN values
df_ksi.TRAFFCTL.replace(to_replace = np.nan, value = 'Traffctl_Not_Informed', inplace = True)


# In[318]:


df_ksi.VISIBILITY.isna().sum()


# In[319]:


# Merge 'Other and NaN into a new category'
df_ksi.VISIBILITY.replace(to_replace = ['Other', np.nan], value = 'Other_Visibility', inplace = True)


# In[321]:


df_ksi.VISIBILITY.value_counts()


# In[322]:


df_ksi.RDSFCOND.value_counts()


# In[323]:


df_ksi.RDSFCOND.isna().sum()


# In[324]:


# Merge 'Other' and NaN into 'Other_Road_Conditions'
df_ksi.RDSFCOND.replace(to_replace = ['Other', np.nan], value = 'Other_Road_Conditions', inplace = True)


# In[325]:


df_ksi.RDSFCOND.value_counts()


# In[326]:


# Merge 'Other' and NaN into 'Other_Impact_Type'
df_ksi.IMPACTYPE.replace(to_replace = ['Other', np.nan], value = 'Other_Impact_Type', inplace = True)


# In[327]:


df_ksi.IMPACTYPE.value_counts()


# In[328]:


df_ksi.INVTYPE.value_counts()


# In[329]:


# Drop Involvement type, since it seems to overlap with other features' information 
df_ksi.drop('INVTYPE', axis = 1, inplace = True)


# In[330]:


df_ksi.INJURY.value_counts()


# In[331]:


df_ksi.INJURY.isna().sum()


# In[332]:


# Inkury also seems to be relevant
df_ksi.INJURY.replace(to_replace = np.nan, value = 'Injury_Not_Disclosed', inplace = True)


# In[333]:


df_ksi.FATAL_NO.isna().sum()


# In[334]:


# Fatal number is merely an identifier
df_ksi.drop('FATAL_NO', axis = 1, inplace = True)


# In[335]:


df_ksi.INITDIR.value_counts()


# In[336]:


df_ksi.INITDIR.isna().sum()


# In[337]:


# Drop 'INITDIR' to reduce dimension and model complexity
df_ksi.drop('INITDIR', axis = 1, inplace = True)


# In[338]:


df_ksi.VEHTYPE.value_counts()


# In[339]:


# Drop 'VEHTYPE' to reduce dimensions and model complexity
# The most relevant information contained here is replicated in other columns
df_ksi.drop('VEHTYPE', axis = 1, inplace = True)


# In[340]:


df_ksi.MANOEUVER.isna().sum()


# In[341]:


df_ksi.MANOEUVER.value_counts()


# In[342]:


# Drop 'MANOUVER' to reduce dimensions and model complexity
df_ksi.drop('MANOEUVER', axis = 1, inplace = True)


# In[343]:


# Driver Action seems relevant
df_ksi.DRIVACT.replace(to_replace = ['Other', np.nan], value = 'Other_Driver_Action', inplace = True)


# In[344]:


# Driver Condition also seems relevant
df_ksi.DRIVCOND.value_counts()


# In[345]:


df_ksi.DRIVCOND.isna().sum()


# In[346]:


df_ksi.DRIVCOND.replace(to_replace = ['Unknown', 'Other', np.nan], value = 'Driver_Condition_Unkown', inplace = True)


# In[347]:


df_ksi.PEDTYPE.isna().sum()


# In[348]:


# Drop 'PEDTYPE' to reduce dimensions and model complexity
df_ksi.drop('PEDTYPE', axis = 1, inplace = True)


# In[82]:


df_ksi.PEDACT.isna().sum()


# In[83]:


df_ksi.PEDACT.value_counts()


# In[349]:


# Drop 'PEDACT' to reduce dimensions and model complexity
df_ksi.drop('PEDACT', axis = 1, inplace = True)


# In[85]:


df_ksi.PEDCOND.value_counts()


# In[86]:


df_ksi.PEDCOND.isna().sum()


# In[350]:


# Drop 'PEDCOND' to reduce dimensions and model complexity
df_ksi.drop('PEDCOND', axis = 1, inplace = True)


# In[88]:


df_ksi.CYCLISTYPE.value_counts()


# In[89]:


df_ksi.CYCLISTYPE.isna().sum()


# In[351]:


# Drop 'CYCLISTYPE' to reduce dimensions and model complexity
df_ksi.drop('CYCLISTYPE', axis = 1, inplace = True)


# In[91]:


df_ksi.CYCACT.value_counts()


# In[352]:


df_ksi.CYCACT.isna().sum()


# In[353]:


# Drop 'CYCACT' to reduce dimensions and model complexity
df_ksi.drop('CYCACT', axis = 1, inplace = True)


# In[95]:


df_ksi.CYCCOND.isna().sum()


# In[354]:


df_ksi.CYCCOND.value_counts()


# In[355]:


# Drop 'CYCCOND' due to the number of NaN values and to reduce model complexity
df_ksi.drop('CYCCOND', axis = 1, inplace = True)


# ### Keep the categories of those involved in the accident

# In[97]:


df_ksi.PEDESTRIAN.isna().sum()


# In[98]:


df_ksi.PEDESTRIAN.value_counts()


# In[99]:


df_ksi.PEDESTRIAN = df_ksi.PEDESTRIAN.map({'Yes': 1,
                                          np.nan: 0})


# In[100]:


df_ksi.CYCLIST.isna().sum()


# In[101]:


df_ksi.CYCLIST = df_ksi.CYCLIST.map({'Yes': 1,
                                          np.nan: 0})


# In[102]:


df_ksi.AUTOMOBILE.isna().sum()


# In[103]:


df_ksi.AUTOMOBILE.value_counts()


# In[104]:


df_ksi.AUTOMOBILE = df_ksi.AUTOMOBILE.map({'Yes': 1,
                                          np.nan: 0})


# In[105]:


df_ksi.MOTORCYCLE.value_counts()


# In[106]:


df_ksi.MOTORCYCLE = df_ksi.MOTORCYCLE.map({'Yes': 1,
                                          np.nan: 0})


# In[107]:


df_ksi.TRUCK.isna().sum()


# In[108]:


df_ksi.TRUCK.value_counts()


# In[109]:


df_ksi.TRUCK = df_ksi.TRUCK.map({'Yes': 1,
                                np.nan: 0})


# In[111]:


df_ksi.TRSN_CITY_VEH.isna().sum()


# In[112]:


df_ksi.TRSN_CITY_VEH.value_counts()


# In[113]:


df_ksi.TRSN_CITY_VEH = df_ksi.TRSN_CITY_VEH.map({'Yes': 1,
                                                np.nan: 0})


# In[114]:


df_ksi.EMERG_VEH.isna().sum()


# In[115]:


df_ksi.EMERG_VEH.value_counts()


# In[116]:


df_ksi.EMERG_VEH = df_ksi.EMERG_VEH.map({'Yes': 1,
                                        np.nan: 0})


# In[117]:


df_ksi.PASSENGER.value_counts()


# In[118]:


df_ksi.PASSENGER.isna().sum()


# In[119]:


df_ksi.PASSENGER = df_ksi.PASSENGER.map({'Yes': 1,
                                        np.nan: 0})


# In[120]:


df_ksi.PASSENGER.value_counts()


# ### Speeding, Agressive, red light, and alcohol driving seem empirically important for our model

# In[121]:


df_ksi.SPEEDING.value_counts()


# In[122]:


df_ksi.SPEEDING = df_ksi.SPEEDING.map({'Yes': 1,
                                        np.nan: 0})


# In[123]:


df_ksi.AG_DRIV.value_counts()


# In[124]:


df_ksi.AG_DRIV = df_ksi.AG_DRIV.map({'Yes': 1,
                                    np.nan: 0})


# In[125]:


df_ksi.AG_DRIV.value_counts()


# In[126]:


df_ksi.REDLIGHT.value_counts()


# In[127]:


df_ksi.REDLIGHT = df_ksi.REDLIGHT.map({'Yes': 1,
                                        np.nan: 0})


# In[128]:


df_ksi.ALCOHOL.value_counts()


# In[129]:


df_ksi.ALCOHOL = df_ksi.ALCOHOL.map({'Yes': 1,
                                    np.nan: 0})


# In[130]:


df_ksi.DISABILITY.value_counts()


# In[131]:


df_ksi.DISABILITY = df_ksi.DISABILITY.map({'Yes': 1,
                                           np.nan: 0})


# In[132]:


df_ksi.POLICE_DIVISION.value_counts()


# In[133]:


df_ksi.HOOD_ID.value_counts()


# In[134]:


df_ksi.NEIGHBOURHOOD.value_counts()


# In[356]:


# Drop these location columns and IDs to simplify the model
df_ksi.drop(['HOOD_ID','ObjectId', 'POLICE_DIVISION', 'NEIGHBOURHOOD'], axis = 1, inplace = True)


# In[357]:


# All columns are now integral
df_ksi.info()


# In[358]:


# Create a copy of the dataset to prepare for transformation
df_pipeline = df_ksi.copy()


# In[359]:


df_pipeline.head()


# In[360]:


# Categorical Features
df_categorical = df_pipeline.select_dtypes(include = ['object']).drop('ACCLASS', axis = 1)


# In[361]:


# Numeric Features
df_numeric = df_pipeline[['MONTH', 'DAY', 'TIME', 'LATITUDE', 'LONGITUDE']]


# In[199]:


df_categorical.head()


# In[200]:


df_numeric.head()


# In[362]:


# Reduce the target variable to two classes
df_pipeline.ACCLASS.replace(to_replace = ['Property Damage Only', 'Non-Fatal Injury'], value = 'Non-Fatal', inplace = True)


# In[363]:


df_pipeline.ACCLASS.value_counts()


# In[364]:


# Convert the dependent variable to numeric
classification = pd.get_dummies(df_pipeline['ACCLASS'])
df_pipeline = pd.concat([df_pipeline, classification], axis = 1)
df_pipeline.drop('ACCLASS', axis = 1, inplace = True)


# In[365]:


df_pipeline.drop('Non-Fatal', axis = 1, inplace = True)


# In[366]:


sns.heatmap(df_ksi.corr(method = 'pearson'), annot = True, cmap = 'viridis')
plt.xticks(rotation = 45)
plt.show()


# In[367]:


# Instantiate the encoder
encoder = OneHotEncoder(drop = 'first', handle_unknown='ignore')


# In[368]:


# ColumnTransformer
num_attributes = df_numeric.columns
cat_attributes = df_categorical.columns
transformer = ColumnTransformer([
    ('encoder', OneHotEncoder(drop = 'first', handle_unknown='ignore'), cat_attributes),
    ('standardizer', StandardScaler(), num_attributes)],
    remainder='passthrough',
    verbose_feature_names_out=False)


# In[369]:


transformer.transformers


# In[370]:


# Features and Target
features = df_pipeline.drop('Fatal', axis = 1)
target = df_pipeline.Fatal


# In[371]:


# Split into Training and Test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state = 98)


# In[372]:


# Instantiate a default Support Vector Classifier
svc = SVC()


# In[373]:


# Pipeline Object to streamline the process
pipeline_svc = Pipeline([
    ('col_transformer', transformer),
    ('svc', svc)
    ])


# In[374]:


pipeline_svc.fit(X_train, y_train)


# In[375]:


scores = cross_val_score(pipeline_svc,
                        X_train,
                        y_train,
                        cv=10,
                        n_jobs=-1,
                        verbose=1)


# In[376]:


print(scores)


# In[377]:


print(scores.mean())


# In[378]:


# Predictions
y_pred_svc = pipeline_svc.predict(X_test)


# In[379]:


cm = confusion_matrix(y_test, y_pred_svc)


# In[380]:


plot_confusion_matrix(cm, [0,2], 'SVC - rbf', 'PuBu')


# In[381]:


print(classification_report(y_test, y_pred_svc))


# precision means what percentage of the positive predictions made were actually correct.
# 
# `TP/(TP+FP)`
# 
# Recall in simple terms means, what percentage of actual positive predictions were correctly classified by the classifier.
# 
# `TP/(TP+FN)`
# 
# F1 score can also be described as the harmonic mean or weighted average of precision and recall.
# 
# `2x((precision x recall) / (precision + recall))`

# In[382]:


# Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
param_grid = {'svc__kernel': ['linear', 'rbf', 'poly'],
              'svc__C': [0.01, 0.1, 1, 10, 100],
              'svc__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
              'svc__degree': [2, 3]}


# In[383]:


# Create a GridSearchCV object
grid_search_svc = GridSearchCV(estimator = pipeline_svc,
                                 param_grid = param_grid,
                                 scoring = 'accuracy',
                                 refit = True,
                                 n_jobs = -1,
                                 verbose = 3)


# In[384]:


# Get the start time
start = time.perf_counter()


# In[ ]:


grid_search_svc.fit(X_train, y_train)


# In[ ]:


# The the final time of processing
end = time.perf_counter()


# In[ ]:


# Best hyperparameters
print("tuned hpyerparameters :(best parameters) ", grid_search_svc.best_params_)
print("Best Estimator :", grid_search_svc.best_estimator_)


# In[ ]:


# Store the best model into a variable
best_model_svc = grid_search_svc.best_estimator_


# In[ ]:


# Make predictions with the best SVC model
pred_svc = best_model_svc.predict(X_test)


# In[ ]:


# Get the Score
best_model_svc.score(X_test, y_test)


# In[ ]:


# Import joblib to save the model
import joblib


# In[ ]:


joblib.dump(best_model_svc, "SVC_model.pkl")


# In[ ]:


joblib.dump(pipeline_svc, "pipeline_svc.pkl")


# In[ ]:


import dill


# In[ ]:


dill.dump_session('notebook_env_SVC.db')


# In[ ]:


# Print the classification Report
print('\t\tClassification Report\n\n',classification_report(y_test, pred_svc))


# In[ ]:


# Total time to run GridSearchCV
print(f'GridSearchCV processing time: {round((end-start), 2)} s')


# ## Logistic Regression Model

# In[ ]:


logmodel = LogisticRegression(max_iter=1000)
pipeline_log = Pipeline([
    ('col_transformer', transformer),
    ('log', logmodel)
    ])


# In[ ]:


pipeline_log.fit(X_train, y_train)


# In[ ]:


scores_log = cross_val_score(pipeline_log,
                        X_train,
                        y_train,
                        cv=10,
                        n_jobs=-1,
                        verbose=1)


# In[ ]:


# Logistic Regression Scores
print(scores_log)


# In[ ]:


# Average Scores
print(scores_log.mean())


# In[ ]:


# Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
param_grid_log = {'log__penalty': ['l1', 'l2', 'elasticnet'],
              'log__C': [0.01, 0.1, 1, 10, 100],
              'log__solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga']}


# In[ ]:


# Create a GridSearchCV object
grid_search_logistic = GridSearchCV(estimator = pipeline_log,
                                 param_grid = param_grid_log,
                                 scoring = 'accuracy',
                                 refit = True,
                                 n_jobs = -1,
                                 verbose = 3)


# In[ ]:


# Get the start time
start = time.perf_counter()


# In[ ]:


# Run fit with all sets of parameters
grid_search_logistic.fit(X_train, y_train)


# In[ ]:


# The the final time of processing
end = time.perf_counter()


# In[ ]:


# Total time to run GridSearchCV
print(f'GridSearchCV processing time: {round((end-start), 2)} s')


# In[ ]:


# Store the best model into a variable
best_model_log = grid_search_logistic.best_estimator_


# In[ ]:


joblib.dump(best_model_log, "LOGISTIC_model.pkl")


# In[ ]:


joblib.dump(pipeline_log, "pipeline_logistic.pkl")


# In[ ]:


dill.dump_session('notebook_env_LOGISTIC.db')


# In[ ]:


best_model_log.score(X_test, y_test)


# ## Random Forest Classifier

# In[ ]:


rf = RandomForestClassifier()
pipeline_rf = Pipeline([
    ('col_transformer', transformer),
    ('rf', rf)
    ])


# In[ ]:


pipeline_rf.fit(X_train, y_train)


# In[ ]:


scores_rf = cross_val_score(pipeline_rf,
                        X_train,
                        y_train,
                        cv=10,
                        n_jobs=-1,
                        verbose=1)


# In[ ]:


print(scores_rf)


# In[ ]:


print(scores_rf.mean())


# In[ ]:


# Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
param_grid_rf = {'rf__n_estimators': [100, 150, 200],
                 'rf__criterion': ['gini', 'entropi', 'log_loss'],
                 'rf__max_features': ['auto', 'sqrt', 'log2'],
                 'rf__max_depth': [4, 5, 6, 7, 8]}


# In[ ]:


# Create a GridSearchCV object
grid_search_rf = GridSearchCV(estimator = pipeline_rf,
                                 param_grid = param_grid_rf,
                                 scoring = 'accuracy',
                                 refit = True,
                                 n_jobs = -1,
                                 verbose = 3)


# In[ ]:


# Get the start time
start = time.perf_counter()


# In[ ]:


# Run fit with all sets of parameters
grid_search_rf.fit(X_train, y_train)


# In[ ]:


# The the final time of processing
end = time.perf_counter()


# In[ ]:


# Total time to run GridSearchCV
print(f'GridSearchCV processing time: {round((end-start), 2)} s')


# In[ ]:


# Store the best model into a variable
best_model_rf = grid_search_logistic.best_estimator_


# In[ ]:


joblib.dump(best_model_rf, "RANDOM_FOREST_model.pkl")


# In[ ]:


dill.dump_session('notebook_env_RANDOM_FOREST.db')


# In[ ]:


best_model_rf.score(X_test, y_test)


# ##### END
