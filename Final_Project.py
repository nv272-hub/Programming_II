#!/usr/bin/env python
# coding: utf-8

# # Programing ll Final Project
# ## Nathan Varghese
# 

# # 1

# ***

# In[5]:


import pandas as pd # data frame to store the data

#read in data use panda alias to save it to data frame s
s = pd.read_csv('social_media_usage.csv')
print(s.shape)  # show the dataframe shape
print(s.head())


# ***

# # 2

# In[11]:


import numpy as np

#create funciton that takes in a value x, if x =1 set it to 1, otherwise set it to 0
def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x


# In[15]:


#testing clean_sm using this created toy data frame
toy_data = {'col_A': [9, 1, 4], 
            'col_B': [1, 8, 3]}
toy_df = pd.DataFrame(toy_data)

cleaned_df = toy_df.apply(clean_sm)
print(cleaned_df)


# ***

# # 3

# In[20]:


#display list of all column names to use for the linkedin target variable 
s.columns.tolist()


# In[26]:


#column name for linked in is 'web1h'

# Create the target column 'sm_li'
s['sm_li'] = s['web1h'].apply(clean_sm)

# Select the required features and target
features = ['income', 'educ2', 'par', 'marital', 'sex', 'age']
target = ['sm_li']
all_cols = target + features

ss = s[all_cols].copy()


# In[27]:


#Clean feature columns for missing values (using the correct column names)

# income: 1 to 9, > 9 is missing
ss['income'] = np.where(ss['income'] > 9, np.nan, ss['income'])

# education (educ2): 1 to 8, > 8 is missing
ss['educ2'] = np.where(ss['educ2'] > 8, np.nan, ss['educ2'])

# age: > 98 is missing
ss['age'] = np.where(ss['age'] > 98, np.nan, ss['age'])


# In[28]:


# Binary features: parent (par), married (marital), female (sex)
# 1 = "Yes" for par/marital, and 1="Male", 2="Female" for sex

for col in ['par', 'marital']:
    ss[col] = np.where(ss[col] == 1, 1, 0)

# Re-code sex to create a 'female' binary column (1=Female, 0=Male/Missing)
# need a 'female' column because the data is 'sex' where 2 is Female.
ss['female'] = np.where(ss['sex'] == 2, 1, 0)

ss = ss.drop(columns=['sex'])

ss = ss.rename(columns={'educ2': 'education', 'par': 'parent', 'marital': 'married'})

ss = ss.dropna()

for col in ['sm_li', 'parent', 'married', 'female']:
    ss[col] = ss[col].astype(int)
    
print(f"Original size of s: {s.shape}")
print(f"Cleaned size of ss: {ss.shape}")
print(ss.head())


# In[34]:


#EDA

#provide summary statistics
ss.describe()


# In[41]:


pd.crosstab(ss['sm_li'], columns = "count")


# In[33]:



#visualizations
import seaborn as sns # for plotting pairs plot
import matplotlib.pyplot as plt # for plotting

sns.pairplot(ss, diag_kind='kde')
plt.show()


# ***

# # 4

# In[36]:


#Feature Set (X)
X = ss.drop('sm_li', axis=1)

#Target Vector (y)
y = ss['sm_li']


# ***

# # 5

# In[55]:


from sklearn.model_selection import train_test_split

# X is the Feature Set (independent variables)
# y is the Target Vector (dependent variable)

# Split the data, holding 20% for testing (test_size=0.2)
# random_state ensures the split is the same every time you run the code
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42 # Use a fixed number like 42 for reproducibility
)


# - X_train contains the features set excluding sm_li for 80% of the dataset ss, this training set is used for trying to fit a regression model that learns a relationship between key predictor variables and the response variable 
# 
# - Y_train contains the response variable sm_li, for linkedin usage, this variable is used to predict in response to features from X_train while learning a regression fit through analysis
# 
# - X_test contains the same features as X_train except it accounts for 20% of the data set. The test set is used for making predictions after training the data set to reveal a fitted regression model. 
# 
# - Y_test holds the response variable, linkedusage; this is applied to make final predictions and measure key performance metrics off the model such as accuracy, precision and recall, etc. 
# 

# ***

# # 6

# In[66]:


from sklearn.linear_model import LogisticRegression

#Instantiate the model
logreg = LogisticRegression(
    class_weight='balanced', 
    random_state=42, # Use a fixed random state for reproducibility
    solver='liblinear'
)

# Fit the model with the training data
logregfit = logreg.fit(X_train, y_train)

# Get the coefficients (slopes) for each feature
print(logregfit.coef_)

# Get the intercept (bias)
print(logregfit.intercept_)


# ***

# # 7

# In[67]:


from sklearn.metrics import confusion_matrix, accuracy_score

#Use the model to make predictions on the test feature set (X_test) 1 (LinkedIn User) or 0 (Non-User)
y_pred = logreg.predict(X_test)

#Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on the Test Data: {accuracy:.4f}")

#Generate the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Actual vs. Predicted):\n", cm)


# The columns indicate the predicted results of the model against the response variable sm_li. The rows are the actural results of sm_li. 97 is a 'True-negative' meaning this number 97 times the model correctly predicted users dont use linkedin against the actual result. 64 represents a 'false postive', meaning 64 instaces were predicted as number of user who do use linkedin, but the actual number isnt. 23 denotes a 'false negaitive' meaning the model preditcs that 23 users do not use linkedin the actual results suggest otherwise. And lastly, 68 is a 'true postive' which means both the model and actual results denoted the correct number of linkedin users. 

# ***

# # 8

# In[46]:


class_names = ['Non-User (0)', 'User (1)']
df_cm = pd.DataFrame(cm, 
                     index=class_names, 
                     columns=class_names)

df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
print("\nDataFrame from Confusion Matrix:\n")
print(df_cm)


# ***

# # 9

# In[52]:


TN, FP, FN, TP = cm.ravel() 
# Calculate Precision
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
print(f"Precision: {precision:.2f}")

# Calculate Recall
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
print(f"Recall: {recall:.2f}")

# Calculate F1-score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f"F1-score: {f1_score:.2f}")


# In[51]:


from sklearn.metrics import classification_report, confusion_matrix

report = classification_report(y_test, y_pred, target_names=class_names)
print("\nClassification Report:\n", report)


# Precision measures the accuracy of predictions by taking the ratio of true positive to total number of positive predictions. An example of where this method is preferred is in Fraud or spam detection when questioning the legitimacy of something like an email. 
# Recall measures the ability of a model to find all relevant cases in a data set, also known as the true positive rate(TPR). This is taken as the ratio of true positives over the sum of True positives and False Negative. An example where this is preferred is in medical diagnosis when using model screening to detect if a person has cancer or not. 
# F1 score is the harmonic mean of precision and recall, it provides a single score balance between these 2 metrics. Where this is preferred is in class imbalance when a dataset has an uneven distribution of classes such that accuracy can be misleading. 
# 

# ***

# # 10

# In[49]:


new_users_data = pd.DataFrame({
    'income': [8, 8],
    'education': [7, 7],
    'parent': [0, 0],
    'married': [1, 1],
    'age': [42, 82],
    'female': [1, 1]
})

print("New Data Points:\n", new_users_data)


# In[54]:


probabilities = logreg.predict_proba(new_users_data)

# Extract the probability of using LinkedIn (Class 1)
prob_user_a = probabilities[0, 1]
prob_user_b = probabilities[1, 1]

print(f"Probability of User A (42 y/o) using LinkedIn: {prob_user_a:.4f}")
print(f"Probability of User B (82 y/o) using LinkedIn: {prob_user_b:.4f}")


# When the age varibale is almost doubled while keepiong all other paramters the same, the probability that that user uses linkedin drops signifcantlty. This would imply that users from the older demographic tend not rely on technology and digitial profiles as much because they are perhaps retired and not seeking for a job. Meanwhile, the younger demographic is more reliant on linkedin to connect with professionals, expand their netowrk, and draw attention to new employers/recruiters who are looking to fill roles. 
