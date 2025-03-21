#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES

# In[120]:





# In[121]:


import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# # Loading the Data

# In[123]:

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Read the uploaded file
    st.write("Preview of uploaded file:")
    st.write(df.head()) 
    st.write("Dataset columns:", df.columns)




# In[124]:


print("Dataset columns:", df.columns)
print("Dataset shape:", df.shape)
print("Dataset preview:", df.head())


# In[125]:


if df.shape[1] > 1:  # Ensure there's more than 1 column
    X = df.iloc[:, :-1].values
    print("Shape of X:", X.shape)
else:
    print("Not enough columns to slice features.")


# In[126]:


X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values


# In[127]:


df['Login Time'] = pd.to_datetime(df['Login Time'])
df['Logout Time'] = pd.to_datetime(df['Logout Time'])
df['Login Hour'] = df['Login Time'].dt.hour
df['Logout Hour'] = df['Logout Time'].dt.hour
df['Session Duration (mins)'] = (df['Logout Time'] - df['Login Time']).dt.total_seconds() /60
features = ['Login Hour','Logout Hour', 'Session Duration (mins)', 'Count of Survey Attempts']
x = df[features].values
y = df['Count of Survey Attempts']


# In[128]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Convert categorical columns to numeric
categorical_columns = ['State', 'Region', 'Speciality']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print(df.head())


# In[129]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[1,5,6])], remainder = 'passthrough')


# In[130]:


print(df[['Login Hour', 'Logout Hour', 'Usage Time (mins)']].dtypes)


# In[131]:


features = ['Login Hour', 'Logout Hour', 'Session Duration (mins)', 'Count of Survey Attempts']
X = df[features]

# Ensure X is fully numeric
print(X.dtypes)


# In[132]:


try:
    X_transformed = ct.fit_transform(X)
except Exception as e:
    print("Error during transformation:", e)
    print("Problematic data:", X)


# In[133]:


# Ensure datetime columns are properly converted
df['Login Hour'] = pd.to_datetime(df['Login Time'], errors='coerce').dt.hour
df['Logout Hour'] = pd.to_datetime(df['Logout Time'], errors='coerce').dt.hour
df['Session Duration (mins)'] = (df['Logout Time'] - df['Login Time']).dt.total_seconds() / 60

# Check if there are any null values left
print(df[['Login Hour', 'Logout Hour', 'Session Duration (mins)']].isnull().sum())


# In[134]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)


# In[135]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)


# In[136]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)
accuracy = accuracy_score(Y_test, model.predict(X_test))


# In[137]:


def predict_doctors(model, time, dataset):
    hour = pd.to_datetime(time).hour
    filtered_dataset = df[df['Hour'] == hour]
    X = filtered_dataset[['Hour', 'Session Duration (mins)', 'Count of Survey Attempts']]
    predictions = model.predict(X)
    responsive_doctors = filtered_dataset[predictions == 1]['NPI']


# In[138]:


import streamlit as st

st.title("🩺Doctor Survey Response Predictor")
st.sidebar.header("Predict Doctor Responses")
time_input = st.sidebar.text_input("Enter Time (HH:MM)", "14:00")
st.write(f"*Model Accuracy: {accuracy:.2f}*")


st.markdown("""
This app predicts which doctors are most likely to respond to survey invitations at a given time.
Simply enter a time, and download a CSV of responsive doctors!
""")

if st.sidebar.button("Predict"):
    # Convert user input time to hour
    hour = pd.to_datetime(time_input).hour
    
    # Filter dataset for that hour
    filtered_data = df[df['Login Hour'] == hour]
    
    # Prepare X for prediction
    X_predict = filtered_data[features]
    predictions = model.predict(X_predict)
    
    responsive_doctors = filtered_data[predictions == 1]['NPI']
    
    st.write("Doctors likely to respond at", time_input)
    st.dataframe(responsive_doctors)
    
    # CSV download
    csv_data = responsive_doctors.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="responsive_doctors.csv",
        mime="text/csv"
    )


# In[160]:


dataset.head()


# In[162]:


predictions = model.predict(X)
print(predictions)


# In[164]:


dataset['Predicted'] = predictions
dataset.head()


# In[166]:


dataset.to_csv("my_output.csv", index=False)


# In[168]:





# In[ ]:




