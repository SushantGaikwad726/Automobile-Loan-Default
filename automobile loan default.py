#!/usr/bin/env python
# coding: utf-8

# <p style="background-color:purple; font-family:newtimeroman; color:#FFF9ED; font-size:200%; text-align:center; border-radius:20px; padding:20px;"><strong>Automobile Loan Default Dataset.</strong></p>
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tabulate import tabulate


# <a id="3"></a>
# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;"> Loading and Reading Data <p>

# In[4]:


am=pd.read_csv(r'D:\Train_Dataset.csv')


# In[144]:


am.head()


# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
# <span style="font-size: 20px;">Data Analysis</span></a>

# In[145]:


am.shape


# In[146]:


am.columns


# In[147]:


am.isnull().sum()[am.isnull().sum()>0]


# In[148]:


numerical_data = ["ID", "Client_Income", "Car_Owned", "Bike_Owned", "Active_Loan", "House_Own", "Child_Count",
                  "Credit_Amount", "Loan_Annuity", "Population_Region_Relative", "Age_Days", "Employed_Days", "Registration_Days",
                  "ID_Days", "Own_House_Age", "Mobile_Tag","Homephone_Tag", "Workphone_Working", "Client_Family_Members", 
                   "Cleint_City_Rating", "Application_Process_Day", "Application_Process_Hour", "Score_Source_1",
                   "Score_Source_2", "Score_Source_3", "Social_Circle_Default", "Phone_Change", "Credit_Bureau", "Default"]

# List of column names representing categorical data features
categorical_data = ["Accompany_Client", "Client_Income_Type", "Client_Education", "Client_Marital_Status", "Client_Gender",
                    "Loan_Contract_Type", "Client_Housing_Type", "Client_Occupation", "Client_Permanent_Match_Tag", 
                     "Client_Contact_Work_Tag", 
                     "Type_Organization"]


# In[149]:


am.Social_Circle_Default.unique()


# In[150]:


#Checking if the column has only numerical data
#if not converting it into null
l=list(am.Social_Circle_Default.unique())
for i in l:
    try:
        i=float(i)
    except:
        print(i)


# In[151]:


am.Client_Income=am.Client_Income.replace({'$':np.NaN,'nan':np.NaN})
am.Credit_Amount=am.Credit_Amount.replace({'$':np.NaN,'nan':np.NaN})
am.Loan_Annuity=am.Loan_Annuity.replace({'$':np.NaN,'#VALUE!':np.NaN,'nan':np.NaN})
am.Age_Days=am.Age_Days.replace({'x':np.NaN,'nan':np.NaN})
am.Employed_Days=am.Employed_Days.replace({'x':np.NaN,'nan':np.NaN})
am.Registration_Days=am.Registration_Days.replace({'x':np.NaN,'nan':np.NaN})
am.ID_Days=am.ID_Days.replace({'x':np.NaN,'nan':np.NaN})
am.Client_Family_Members=am.Client_Family_Members.replace({'nan':np.NaN})
am.Cleint_City_Rating=am.Cleint_City_Rating.replace({'nan':np.NaN})
am.Application_Process_Day=am.Application_Process_Day.replace({'nan':np.NaN})
am.Score_Source_2=am.Score_Source_2.replace({'nan':np.NaN})
am.Phone_Change=am.Phone_Change.replace({'nan':np.NaN})
am.Credit_Bureau=am.Credit_Bureau.replace({'nan':np.NaN})
am.Type_Organization=am.Type_Organization.replace({'nan':np.NaN,'XNA':np.NaN})
am.Score_Source_3=am.Score_Source_3.replace({'&':np.NaN})
am.Social_Circle_Default=am.Social_Circle_Default.replace({'nan':np.NaN})

am.Accompany_Client=am.Accompany_Client.replace({'##':np.NaN,'nan':np.NaN})
am.Client_Income_Type=am.Client_Income_Type.replace({'nan':np.NaN})
am.Client_Education=am.Client_Education.replace({'nan':np.NaN})
am.Client_Marital_Status=am.Client_Marital_Status.replace({'nan':np.NaN})
am.Car_Owned=am.Car_Owned.replace({'nan':np.NaN})
am.Bike_Owned=am.Bike_Owned.replace({'nan':np.NaN})
am.Active_Loan=am.Active_Loan.replace({'nan':np.NaN})
am.House_Own=am.House_Own.replace({'nan':np.NaN})
am.Child_Count=am.Child_Count.replace({'nan':np.NaN})
am.Client_Gender=am.Client_Gender.replace({'nan':np.NaN,'XNA':np.NaN})
am.Loan_Contract_Type=am.Loan_Contract_Type.replace({'nan':np.NaN})
am.Client_Housing_Type=am.Client_Housing_Type.replace({'nan':np.NaN})
am.Population_Region_Relative=am.Population_Region_Relative.replace({'nan':np.NaN,'@':np.NaN,'#':np.NaN})


# In[152]:


#Converted whole column to float dtype
for i in numerical_data:
    am[i]=am[i].astype(float)
    


# In[153]:


#nulls in these columns were more than 30% of total records so we dropped those columns
am=am.drop(['Own_House_Age','Client_Occupation','Score_Source_1','Score_Source_3','Social_Circle_Default'],axis=1)


# In[154]:


am.dtypes


# In[155]:


am.isnull().sum()[am.isnull().sum()>0]


# In[156]:


am.Population_Region_Relative.value_counts()


# In[157]:


am.Population_Region_Relative.median()


# In[158]:


numerical_data = [["ID", "Client_Income", "Car_Owned", "Bike_Owned", "Active_Loan", "House_Own", "Child_Count",
                  "Credit_Amount", "Loan_Annuity", "Population_Region_Relative", "Age_Days", "Employed_Days", "Registration_Days",
                  "ID_Days", "Own_House_Age", "Mobile_Tag","Homephone_Tag", "Workphone_Working", "Client_Family_Members", 
                   "Cleint_City_Rating", "Application_Process_Day", "Application_Process_Hour", "Score_Source_1",
                   "Score_Source_2", "Score_Source_3", "Social_Circle_Default", "Phone_Change", "Credit_Bureau", "Default"]]


# In[ ]:





# In[159]:


am.Client_Income.fillna(am.Client_Income.median(),inplace=True)
am.Car_Owned.fillna(0.0,inplace=True)
am.Bike_Owned.fillna(0.0,inplace=True)
am.Active_Loan.fillna(am.Active_Loan.median(),inplace=True)
am.House_Own.fillna(0.0,inplace=True)
am.Child_Count.fillna(am.Child_Count.median(),inplace=True)
am.Credit_Amount=am.Credit_Amount.astype(float)
am.Credit_Amount.fillna(am.Credit_Amount.mean(),inplace=True)
am.Loan_Annuity=am.Loan_Annuity.astype(float)

am.Loan_Annuity.fillna(am.Loan_Annuity.mean(),inplace=True)

am.Population_Region_Relative.fillna(am.Population_Region_Relative.median(),inplace=True)
am.Age_Days.fillna(am.Age_Days.median(),inplace=True)
am.Employed_Days.fillna(am.Employed_Days.median(),inplace=True)
am.Registration_Days.fillna(am.Registration_Days.median(),inplace=True)
am.ID_Days.fillna(am.ID_Days.median(),inplace=True)
am.Client_Family_Members.fillna(am.Client_Family_Members.median(),inplace=True)
am.Cleint_City_Rating.fillna(am.Cleint_City_Rating.median(),inplace=True)
am.Application_Process_Day.fillna(am.Application_Process_Day.median(),inplace=True)
am.Phone_Change.fillna(am.Phone_Change.median(),inplace=True)
am.Credit_Bureau.fillna(am.Credit_Bureau.median(),inplace=True)


# In[160]:


#columns with dtype object are still pending
am.isnull().sum()[am.isnull().sum()>0]


# In[161]:


am.Type_Organization.replace({'Business Entity Type 3':'Business',
           'Business Entity Type 2':'Business',
           'Business Entity Type 1':'Business',
           'Trade: type 7':'Trade',
            'Trade: type 3':'Trade',
            'Transport: type 4':'Transport',
            'Industry: type 9':'Industry',
            'Industry: type 3':'Industry',
            'Industry: type 11':'Industry',
            'Transport: type 2 ':'Transport',
            'Trade: type 2':'Trade',
            'Transport: type 3':'Transport',
            'Industry: type 7':'Industry',
            'Industry: type 4':'Industry',
            'Industry: type 1':'Industry',
            'Industry: type 5':'Industry',
            'Trade: type 6':'Trade',
            'Industry: type 2':'Industry',
            'Industry: type 12':'Industry',
            'Trade: type 1':'Trade',
            'Transport: type 1':'Transport',
            'Industry: type 10':'Industry',
            'Industry: type 6':'Industry',
            'Industry: type 13':'Industry',
            'Trade: type 4':'Trade',
            'Trade: type 5':'Trade',
            'Industry: type 8':'Industry',
            'Transport: type 2':'Industry'},inplace=True)
            
            


# In[162]:


am.Score_Source_2.value_counts()


# In[163]:


am.Accompany_Client.fillna('Alone',inplace=True)
am.Client_Income_Type.fillna('Service',inplace=True)
am.Client_Education.fillna('Secondary',inplace=True)
am.Client_Marital_Status.fillna('M',inplace=True)
am.Client_Gender.fillna('Male',inplace=True)
am.Loan_Contract_Type.fillna('RL',inplace=True)
am.Client_Housing_Type.fillna('Family',inplace=True)
am.Application_Process_Hour.fillna(am.Application_Process_Hour.median(),inplace=True)
am.Type_Organization.fillna('Self-employed',inplace=True)
am.Score_Source_2.fillna(am.Score_Source_2.mean(),inplace=True)


# In[164]:


am.isnull().sum()[am.isnull().sum()>0]


# In[165]:


# Count the total number of duplicate rows in the DataFrame
am.duplicated().sum()


# In[166]:


am1=am


# In[167]:


am=am.drop(['ID'],axis=1)


# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;">EDA  <p> 

# In[168]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[169]:


# Distribution of Categorical Variables
categorical_cols = [ "Client_Income_Type", "Client_Education", "Client_Marital_Status", "Client_Gender",
                     "Client_Housing_Type",  "Type_Organization","Accompany_Client"]

print("\nDistribution of Categorical Variables")
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(y=col, data=am)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[170]:


from sklearn.preprocessing import LabelEncoder 
le= LabelEncoder()

am[am.select_dtypes(include='object').columns]=am[am.select_dtypes(include='object').columns].apply(le.fit_transform)


# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Spliting Data into Train and Test</span>
# </a>
# 

# In[171]:


from sklearn.model_selection import train_test_split


# In[172]:


am_train,am_test=train_test_split(am,test_size=.2)


# In[173]:


am_train.Default.value_counts()


# In[174]:


#undersampling
a00,a000=train_test_split(am_train[am_train.Default==0.0],test_size=0.75)


# In[175]:


am_train=am_train.drop(a000.index,axis=0)


# In[176]:


am_train.Default.value_counts()


# In[177]:


a1=am_train[am_train.Default==1.0]


# In[178]:


#oversampling
am_train=pd.concat([am_train,a1,a1],axis=0)


# In[179]:


am_train.Default.value_counts()


# In[180]:


am_trainx=am_train.iloc[::,0:-1]
am_trainy=am_train.iloc[::,-1]

am_testx=am_test.iloc[::,0:-1]
am_testy=am_test.iloc[::,-1]


# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;"> Logistic Regression  <p>
# 
# - Logistic regression models the relationship between a binary outcome variable and independent variables by estimating the probability of occurrence. It's favored for binary classification tasks due to its simplicity and effectiveness. Using a sigmoid function, it transforms the equation, enabling the modeling of probabilities and binary decisions.

# In[181]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(class_weight='balanced', penalty='l2', solver='liblinear', random_state=42)


# In[182]:


logreg.fit(am_trainx , am_trainy)


# In[183]:


logregpred_test = logreg.predict(am_testx)


# In[184]:


from sklearn.metrics import confusion_matrix ,classification_report


# In[185]:


confusion_matrix(am_testy , logregpred_test)


# In[186]:


print(classification_report(am_testy , logregpred_test))


# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;">Decision Tree   <p> 
# 
# - Decision tree models the relationship between input features and a target variable by partitioning the feature space into segments, making binary decisions at each node. It's favored for its simplicity and interpretability, breaking down complex decision-making processes into a series of straightforward if-else conditions. By recursively splitting the data based on feature thresholds, it creates a tree-like structure, enabling clear visualization of decision paths and feature importance.

# In[187]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(class_weight='balanced')


# In[188]:


dt.fit(am_trainx , am_trainy)


# In[189]:


pred_test_dt=dt.predict(am_testx)


# In[190]:


from sklearn.metrics import confusion_matrix ,classification_report


# In[191]:


confusion_matrix(am_testy , pred_test_dt)


# In[192]:


print(classification_report(am_testy , pred_test_dt))


# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;"> GridSearchCV <p>
# 
# - 
# GridSearchCV is a technique in machine learning that systematically searches through a specified grid of hyperparameters to find the optimal configuration for a model. It uses cross-validation to evaluate each combination, aiming to maximize model performance by automating the process of hyperparameter tuning.

# In[193]:


from sklearn.model_selection import GridSearchCV
search_dict={"criterion":["gini","entropy"],
            "max_depth":range(3,8),
            "min_samples_split":range(25,50)}
from sklearn.tree import DecisionTreeClassifier
dt_mp=DecisionTreeClassifier()
grid=GridSearchCV(dt_mp,param_grid=search_dict)

#search dict---> dict created by the user
#grid search---> best combination of hyper parameters(hyper parameter tunning)


# In[194]:


grid.fit(am_trainx,am_trainy)


# In[195]:


pred_test_grid=grid.predict(am_testx)


# In[196]:


confusion_matrix(am_testy , pred_test_grid)


# In[197]:


print(classification_report(am_testy , pred_test_grid))


# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;">Random Forest   <p> 
# - Random Forest, an ensemble learning method, builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting. By aggregating the results of individual trees, it enhances robustness and provides superior performance for classification and regression tasks.

# In[198]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=200 , criterion='entropy')


# In[199]:


rfc.fit(am_trainx , am_trainy)


# In[200]:


# 0 MEANS CUSTOMER CAN REPAY THE LOAN
#1 MEANS CANNOT REPAY THE LOAN


# In[201]:


pred_test_rfc=rfc.predict(am_testx)


# In[202]:


from sklearn.metrics import confusion_matrix ,classification_report


# In[203]:


confusion_matrix(am_testy , pred_test_rfc)


# In[204]:


print(classification_report(am_testy , pred_test_rfc))


# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;">Feature Importance</span>

# In[205]:


x=dt.feature_importances_     ###  feature importance


# In[206]:


fi=pd.DataFrame()


# In[207]:


fi['Imp']=x


# In[208]:


fi['fe']=am.columns[0:-1]


# In[209]:


fi=fi.sort_values('Imp',ascending=False)


# In[210]:


fi['Imp'][0:27].sum()


# In[211]:


d=list(fi.fe[0:27])
d


# In[212]:


#from sklearn.svm import SVC


# In[213]:


#svc_am = SVC(kernel="linear")


# In[214]:


#svc_am.fit(am_trainx , am_trainy)


# In[215]:


#pred_svc_am = svc_am.predict(am_testx)


# In[216]:


#from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score,f1_score,classification_report


# In[217]:


#confusion_matrix(am_testy,pred_svc_am)


# In[218]:


#print(classification_report(am_testy,pred_svc_am))


# 
# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;"> GradientBoostingClassifier  <p>
# 
# - 
# Gradient Boosting is a machine learning technique that builds an ensemble of weak learners (typically decision trees) sequentially. It trains each new model to correct errors made by the previous ones, optimizing a differentiable loss function using gradients. Gradient Boosting enhances predictive accuracy and handles complex data relationships effectively
# 

# In[219]:


from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()
model.fit(am_trainx,am_trainy)


# In[220]:


pred_test_am = model.predict(am_testx)

tab_am = confusion_matrix(am_testy, pred_test_am)
tab_am


# In[221]:


print(classification_report(am_testy, pred_test_am))


# 
# <a href="#toc" class="btn btn-primary btn-sm" role="button" aria-pressed="true" 
# style="color:black; background-color:#dfa8e4; font-size: 16px; border-width: 2px; font-weight: bold;" data-toggle="popover">
#     <span style="font-size: 20px;"> Changing Optimum Threshold </span>

# In[222]:


#WE TOOK THE MOST SUITED MODEL


# In[223]:


#


# In[224]:


pred_prob=rfc.predict_proba(am_testx)
pred_prob=pd.DataFrame(pred_prob)

pred_prob.rename(columns={pred_prob.columns[0]:'Prob0',pred_prob.columns[1]:'Prob1'},inplace=True)
rang=[i / 100 for i in range(10, 101)]
rang


# In[225]:


from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score,f1_score,classification_report


# In[226]:


tpr=[]
precision=[]
fpr=[]
f1=[]
acc=[]
for i in rang:
    pred_prob.loc [pred_prob.Prob1 >= i, 'Outcome'] = 1 # outcome is the new column added which conatins CLASS 1

    pred_prob.loc[pred_prob.Prob1<i, 'Outcome'] = 0 # values based on prob
    tab1 =confusion_matrix(am_testy, pred_prob.Outcome)
    tpr1 = recall_score(am_testy, pred_prob.Outcome)
    fpr1 = tab1[0][1]/(tab1[0][1] + tab1[0][0]) 
    precison1 = precision_score(am_testy, pred_prob.Outcome)
    f11= (2 *tpr1* precison1) /(tpr1+ precison1)

    tpr.append(tpr1)

    precision.append(precison1)

    fpr.append(fpr1)

    f1.append(f11)

    acc.append(tab1.diagonal().sum()/tab1.sum())


# In[227]:


import matplotlib.pyplot as plt 
plt.figure(figsize=(15,8))
plt.plot(rang,tpr,color='red',marker='*',label='Tpr-Recall')
plt.plot(rang,precision,color='green',marker='*',label='Precisiom')
plt.plot(rang,fpr,color='blue',marker='*',label='Fpr')
plt.plot(rang,f1,color='black',marker='*',label='F1')
plt.plot(rang,acc,color='yellow',marker='*',label='Accuracy')
plt.legend()
plt.ylabel('Fpr, Tpr, Precision')
plt.xlabel('Prob Cutoff for Class One')
plt.grid()



# In[ ]:





# In[93]:


'''def remove_outliers (df,col,k):
    mean= df[col].mean()
    global df1
    sd= df[col].std()
    final_list= [x for x in df[col] if (x > mean - k * sd)]
    final_list= [x for x in final_list if (x < mean + k * sd)]
    df1= df.loc[df[col].isin(final_list)]; print(df1.shape)
    print('Number of outliers removed ==>', df.shape[0]- df1.shape[0])
'''


# In[94]:


#remove_outliers(am2,'Employed_Days',2)


# ## <p style="background-color:#B61151; font-family:newtimeroman; color:#FFF9ED; font-size:150%; text-align:center; border-radius:10px 10px;"> Compairing Models  <p> 

# In[231]:


models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', penalty='l2', solver='liblinear', random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}


results = []

for model_name, model in models.items():
    model.fit(am_trainx, am_trainy)
    pred_test = model.predict(am_testx)
    recall = recall_score(am_testy, pred_test)
    precision = precision_score(am_testy, pred_test)
    f1 = f1_score(am_testy, pred_test)
    accuracy = accuracy_score(am_testy, pred_test)
    results.append({
        'Model': model_name,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1,
        'Accuracy': accuracy
    })


# In[232]:


print(tabulate(results_df, headers='keys', tablefmt='grid'))


# <p style="background-color:purple; font-family:newtimeroman; color:#FFF9ED; font-size:200%; text-align:center; border-radius:20px; padding:20px;"><strong>Thank You :)</strong></p>
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[168]:


print("Basic Information")
print(am.info())


# In[169]:


#  Missing Values
print("\nMissing Values")
missing_values = am.isnull().sum()
print(missing_values[missing_values > 0])


# In[170]:


#  Summary Statistics
print("\nSummary Statistics")
print(list(am.describe(include='all')))


# In[ ]:





# In[ ]:


results = []

for model_name, model in models.items():
    model.fit(am_trainx, am_trainy)
    pred_test = model.predict(am_testx)
    recall = recall_score(am_testy, pred_test)
    precision = precision_score(am_testy, pred_test)
    f1 = f1_score(am_testy, pred_test)
    accuracy = accuracy_score(am_testy, pred_test)
    results.append({
        'Model': model_name,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1,
        'Accuracy': accuracy
    })




# In[ ]:


# Convert results to DataFrame
results_df = pd.DataFrame(results)


# In[ ]:


# Display results
print(results_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




