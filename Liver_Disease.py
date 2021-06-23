#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing all the libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')

# import models

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# model Evaluations
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import cross_val_score 


# # Loading Data

# In[2]:


df=pd.read_csv('indian_liver_patient.csv')
df


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df['Dataset'].value_counts()


# # EDA

# In[6]:


fig,ax =plt.subplots(figsize=(10,5))
df['Dataset'].value_counts().plot(kind='bar',ax=ax,color=['orange','lightgreen'])
plt.xticks(rotation=0);


# In[7]:


la= LabelEncoder()
df["Gender"]=la.fit_transform(df["Gender"])


# In[8]:


df


# In[9]:


df.isna().sum()


# In[10]:


df.dropna(inplace=True)


# In[11]:


df.isna().sum()


# In[12]:


over50=df[df['Age']>50]
over50


# In[13]:


plt.style.use('seaborn-whitegrid')
fig , (ax0,ax1)=plt.subplots(nrows = 2,ncols=1,figsize=(10,10),sharex=True)
scatter=ax0.scatter(x=over50['Age'],y=over50['Total_Protiens'],c=over50['Dataset'],cmap='winter')
ax0.set(title='Liver disease and proteins',ylabel='Proteins')
ax0.legend(*scatter.legend_elements(),title='Dataset')
ax0.axhline(y=over50['Total_Protiens'].mean(),linestyle='--')
ax0.set_xlim([50,80])

scatter2=ax1.scatter(x=over50['Age'],y=over50['Albumin'],c=over50['Dataset'],cmap='summer')
ax1.set(title='Liver disease and albumin',ylabel='Albumin')
ax1.legend(*scatter.legend_elements(),title='Dataset')
ax1.axhline(y=over50['Albumin'].mean(),linestyle='--')
ax1.set_xlim([50,80])
ax1.set_ylim([0,10])

fig.suptitle('Liver Disease Analysis',fontsize=16,fontweight='bold')


# In[14]:


df['Total_Bilirubin'].plot.hist()


# In[15]:


df['Age'].plot.hist(figsize=(10,5))


# In[16]:


sns.jointplot(x='Total_Protiens',y='Albumin',data=df,kind='reg')


# In[17]:


corr=df.corr()


# In[18]:


corr


# In[19]:


plt.figure(figsize=(16,16))
sns.heatmap(corr,cbar=True,square=True,annot=True,cmap='winter')


# # Modeling

# In[20]:


df.columns


# In[21]:


df.rename(columns={'Dataset':'Target'},inplace=True)


# In[22]:


X=df[['Total_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
      'Total_Protiens','Age','Gender',
       'Albumin_and_Globulin_Ratio']]
'''scaler = MinMaxScaler()
X = scaler.fit_transform(X)'''


# In[23]:


X


# In[24]:


X.columns


# In[25]:


y=df['Target']


# In[26]:


np.random.seed(42)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)


# In[27]:



models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector' : SVC(),
    'Decission Tree' : DecisionTreeClassifier()
}

# create a function to fit and score models
def fit_and_score(models,X_train,X_test,y_train,y_test):
    """
    X_train : Training data (no labels)
    X_test : Testing data(no labels)
    y_train : training data(no lables)
    y_test : test lables
    """
    np.random.seed(42)

    model_scores = {}
    for name, model in models.items():
        model.fit(X_train,y_train)
        model_scores[name] = model.score(X_test,y_test)

    return model_scores


# In[28]:


model_score = fit_and_score(models = models,
                            X_train=X_train,
                            X_test=X_test,
                            y_train=y_train,
                            y_test=y_test)

model_score


# In[29]:


model_compare = pd.DataFrame(model_score, index=['accuracy'])
model_compare.T.plot.bar()


# In[30]:


model_compare


# # Hyperparamater tuning

# In[31]:


train_score = []
test_score = []

neighbours = range(1,21)
knn = KNeighborsClassifier()

for i in neighbours:
    knn.set_params(n_neighbors = i)

    #fit the algorithm

    knn.fit(X_train,y_train)

    train_score.append(knn.score(X_train,y_train))

    test_score.append(knn.score(X_test,y_test))


# In[32]:


plt.plot(neighbours,train_score, label='Train score')
plt.plot(neighbours,test_score,label='Test score')
plt.xticks(np.arange(1,21,1))
plt.xlabel('Number of Neighbours')
plt.ylabel('Model score')
plt.legend()


# # Hyperparameter tuning with  RandomizedSearchCV
# 
# We will tune:
# * RandomForestClassifier
# * Logistic Regression

# In[33]:


log_reg_grid = {
    'C':np.logspace(-4,4,20),
    'solver':['liblinear']
}

rf_grid = {
    'n_estimators':np.arange(10,1000,50),
    'max_depth':[None,3,5,10],
    'min_samples_split':np.arange(2,20,20),
    'min_samples_leaf':np.arange(1,20,2)
}


# In[34]:


np.random.seed(42)
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
            param_distributions=log_reg_grid,
            cv = 5,
            n_iter=20,
            verbose=True,
            n_jobs=-1)

rs_log_reg.fit(X_train,y_train)            


# In[35]:


rs_log_reg.best_params_


# In[36]:


rs_log_reg.score(X_test,y_test)


# In[37]:


# tune RandomForest
np.random.seed(42)
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
        param_distributions=rf_grid,
        cv=5,
        verbose=True,
        n_iter=20,
        n_jobs=-1)

rs_rf.fit(X_train,y_train)


# In[38]:


#find best hyperparameters
rs_rf.best_params_


# In[39]:


rs_rf.score(X_test,y_test)


# # Hyperparamater tuning for LogisticRegressionn using GridSearchCV

# In[40]:


log_reg_grid = {'C': np.logspace(-4,4,30),
                'solver': ['liblinear']}

gs_log_reg = GridSearchCV(LogisticRegression(),
                        param_grid = log_reg_grid,
                        cv = 5,
                        verbose = True,
                        n_jobs = -1)
        
gs_log_reg.fit(X_train,y_train)


# In[41]:


gs_log_reg.best_params_


# In[42]:


gs_log_reg.score(X_test,y_test) 


# In[43]:


rf_grid = {
    'n_estimators':np.arange(10,1000,50),
    'max_depth':[None,3,5,10],
    'min_samples_split':np.arange(2,20,20),
    'min_samples_leaf':np.arange(1,20,2)
}

gs_rf = GridSearchCV(RandomForestClassifier(),
                        param_grid = rf_grid,
                        cv = 5,
                        verbose = True,
                        n_jobs=-1)
        
gs_rf.fit(X_train,y_train)


# 

# In[44]:


gs_rf.best_params_


# In[45]:


gs_rf.score(X_test,y_test) 


# In[49]:


y_preds = gs_rf.predict(X_test)


# In[50]:


print(classification_report(y_test,y_preds))


# In[53]:


import joblib,pickle
pickle.dump(gs_rf, open('liver_model_predict.pkl','wb'))


# In[54]:


mod = pickle.load(open('liver_model_predict.pkl','rb'))
print(mod.predict([[0.7,187,16,6.8,65,0,0.90]]))


# In[ ]:




