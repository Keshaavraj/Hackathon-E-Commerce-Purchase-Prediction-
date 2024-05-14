#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[52]:


data = 'D:/Datascience/train_wn75k28.csv'


# In[53]:


df = pd.read_csv(data)


# # Understanding the Data for Logistic Regression

# In[54]:


df.shape


# In[55]:


df.head()


# In[56]:


df.info()


# In[57]:


df.isnull().sum()


# In[58]:


#Checking null percentage
df.isnull().mean()*100


# In[59]:


df.describe()


# In[60]:


print(df.columns.tolist())


# In[61]:


#Checking the unique value counts in columns
featureValues={}
for d in df.columns.tolist():
    count=df[d].nunique()
    if count==1:
        featureValues[d]=count
# List of columns having same 1 unique value        
cols_to_drop= list(featureValues.keys())
print("Columns having 1 unique value are :",cols_to_drop)


# In[62]:


df.nunique()


# In[63]:


df.drop(['id','signup_date'],inplace=True,axis=1)


# In[64]:


df.isnull().mean()*100


# In[65]:


df.isna().mean()*100


# In[66]:


#fill missing value
df["products_purchased"]= df["products_purchased"].fillna(0)


# # Exploratory Data Analysis before creating a Logistic Regression Model

# In[67]:


# cchart for distribution of target variable
fig= plt.figure(figsize=(10,3) )
fig.add_subplot(1,2,1)
a= df["buy"].value_counts(normalize=True).plot.pie()
fig.add_subplot(1,2,2)
churnchart=sns.countplot(x=df["buy"])
plt.tight_layout()
plt.show()


# In[76]:


# Visualize relationship between promoted and other features
fig= plt.figure(figsize=(10,5) )
fig.add_subplot(1,3,1)
ar_6=sns.boxplot(x=df["buy"],y=df["campaign_var_1"])
fig.add_subplot(1,3,2)
ar_6=sns.boxplot(x=df["buy"],y=df["campaign_var_2"])
plt.tight_layout()
plt.show()


# In[77]:


# Visualize relationship between promoted and other features
fig= plt.figure(figsize=(10,5) )
fig.add_subplot(1,3,1)
ar_6=sns.boxplot(x=df["products_purchased"],y=df["campaign_var_1"])
fig.add_subplot(1,3,2)
ar_6=sns.boxplot(x=df["products_purchased"],y=df["campaign_var_2"])
plt.tight_layout()
plt.show()


# In[69]:


# Visualize relationship between promoted and other features
fig= plt.figure(figsize=(20,10) )
fig.add_subplot(1,6,1)
ar_6=sns.boxplot(x=df["buy"],y=df["user_activity_var_1"])
fig.add_subplot(1,6,2)
ar_6=sns.boxplot(x=df["buy"],y=df["user_activity_var_2"])
fig.add_subplot(1,6,3)
ar_6=sns.boxplot(x=df["buy"],y=df["user_activity_var_3"])
fig.add_subplot(1,6,4)
ar_6=sns.boxplot(x=df["buy"],y=df["user_activity_var_4"])
fig.add_subplot(1,6,5)
ar_6=sns.boxplot(x=df["buy"],y=df["user_activity_var_5"])
fig.add_subplot(1,6,6)
ar_6=sns.boxplot(x=df["buy"],y=df["user_activity_var_6"])
fig.add_subplot(2,6,1)
ar_6=sns.boxplot(x=df["buy"],y=df["user_activity_var_7"])
fig.add_subplot(2,6,2)
ar_6=sns.boxplot(x=df["buy"],y=df["user_activity_var_8"])
fig.add_subplot(2,6,3)
ar_6=sns.boxplot(x=df["buy"],y=df["user_activity_var_9"])
fig.add_subplot(2,6,4)
ar_6=sns.boxplot(x=df["buy"],y=df["user_activity_var_10"])
fig.add_subplot(2,6,5)
ar_6=sns.boxplot(x=df["buy"],y=df["user_activity_var_11"])
fig.add_subplot(2,6,6)
ar_6=sns.boxplot(x=df["buy"],y=df["user_activity_var_12"])

plt.tight_layout()
plt.show()


# In[70]:


#correlation between features
fig= plt.figure(figsize=(20,10) )
corr_plot = sns.heatmap(df.corr(),annot = True,linewidths=3 )
plt.title("Correlation plot")
plt.show()


# In[72]:


df.head()


# In[73]:


df.drop(['created_at'],inplace=True,axis=1)


# In[74]:


df.head()


# # Train-Test Split

# In[78]:


from sklearn.model_selection import train_test_split
#split data into dependent variables(X) and independent variable(y) that we would predict
y = df.pop("buy")
X = df
#Let’s split X and y using Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,train_size=0.8)
#get shape of train and test data
print("train size X : ",X_train.shape)
print("train size y : ",y_train.shape)
print("test size X : ",X_test.shape)
print("test size y : ",y_test.shape)


# In[79]:


X.head()


# In[80]:


y.head()


# # Feature Scaling/Normalization

# In[81]:


#Feature scaling
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


# # Class Imbalance

# In[82]:


#check for distribution of labels
y_train.value_counts(normalize=True)


# # Build and Train Logistic Regression model 

# # Creating our base model

# In[83]:


#import library
from sklearn.linear_model import LogisticRegression
#make instance of model with default parameters except class weight
#as we will add class weights due to class imbalance problem
lr_basemodel =LogisticRegression(class_weight={0:0.1,1:0.9})
# train model to learn relationships between input and output variables
lr_basemodel.fit(X_train,y_train)


# In[84]:


y_pred_basemodel = lr_basemodel.predict(X_test)


# In[85]:


y_pred_basemodel


# # Model Evaluation Metrics

# In[86]:


from sklearn.metrics import f1_score
print("f1 score for base model is : " , f1_score(y_test,y_pred_basemodel))


# # Hyperparameter Optimization for the Logistic Regression Model

# In[90]:


#Hyperparameter tuning
# define model/create instance
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
lr=LogisticRegression()
#tuning weight for minority class then weight for majority class will be 1-weight of minority class
#Setting the range for class weights
weights = np.linspace(0.0,0.99,500)
#specifying all hyperparameters with possible values
param= {'C': [0.1, 0.5, 1,10,15,20], 'penalty': ['l1', 'l2'],"class_weight":[{0:x ,1:1.0 -x} for x in weights]}
# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
#Gridsearch for hyperparam tuning
model= GridSearchCV(estimator= lr,param_grid=param,scoring="f1",cv=folds,return_train_score=True)
#train model to learn relationships between x and y
model.fit(X_train,y_train)


# In[91]:


# print best hyperparameters
print("Best F1 score: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)


# # Build Model using optimal values of Hyperparameters

# In[92]:


#Building Model again with best params
lr2=LogisticRegression(class_weight={0:0.32,1:0.67},C=1,penalty="l2")
lr2.fit(X_train,y_train)


# # Model Evaluation

# In[103]:


# predict probabilities on Test and take probability for class 1([:1])
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

y_pred_prob_test = lr2.predict_proba(X_test)[:, 1]
#predict labels on test dataset
y_pred_test = lr2.predict(X_test)
# create onfusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print("confusion Matrix is :\n",cm)
# print("n")
# ROC- AUC score
print("ROC-AUC score  test dataset:  t", roc_auc_score(y_test,y_pred_prob_test))
#Precision score
print("precision score  test dataset:  t", precision_score(y_test,y_pred_test))
#Recall Score
print("Recall score  test dataset:  t", recall_score(y_test,y_pred_test))
#f1 score
print("f1 score  test dataset :  t", f1_score(y_test,y_pred_test))


# In[104]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, lr2.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, lr2.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # Applying test data to this tuned model

# In[105]:


test_data = 'D:/Datascience/test_Wf7sxXF.csv'


# In[106]:


df_test = pd.read_csv(test_data)


# In[107]:


df_test.shape


# In[108]:


df_test.head()


# In[110]:


df_test.isnull().sum()


# In[111]:


df_test.drop(['id','signup_date'],inplace=True,axis=1)
#fill missing value
df_test["products_purchased"]= df["products_purchased"].fillna(0)
df_test.drop(['created_at'],inplace=True,axis=1)


# In[112]:


df_test.head()


# # Feature Scaling/Normalization

# In[114]:


#Feature scaling
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
df_test = scale.fit_transform(df_test)
# X_test = scale.transform(X_test)


# In[116]:


df_test


# In[117]:


df_test_pred_ = lr2.predict(df_test)


# In[118]:


df_test_pred_


# In[120]:


import pandas as pd 
pd.DataFrame(df_test_pred_).to_csv('D:/Datascience/solution_Kesavan.csv', encoding='utf-8')


# In[ ]:




