#!/usr/bin/env python
# coding: utf-8

# In[480]:




# In[481]:


import numpy as np
import pandas as pd


# In[482]:


import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[483]:


dataset = pd.read_csv(r'C:\Users\user\Downloads\dummy_npi_data.xlsx - Dataset.csv')


# In[484]:


dataset


# In[485]:


print("Dataset columns:", dataset.columns)
print("Dataset shape:", dataset.shape)
print("Dataset preview:", dataset.head())


# In[486]:


print("Shape of X (features):", dataset.iloc[:, :-1].shape)


# In[487]:


if dataset.shape[1] > 1:  # Ensure there's more than 1 column
    X = dataset.iloc[:, :-1].values
    print("Shape of X:", X.shape)
else:
    print("Not enough columns to slice features.")


# In[488]:


X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


# In[489]:


dataset['Login Time'] = pd.to_datetime(dataset['Login Time'])
dataset['Logout Time'] = pd.to_datetime(dataset['Logout Time'])
dataset['Login Hour'] = dataset['Login Time'].dt.hour
dataset['Logout Hour'] = dataset['Logout Time'].dt.hour
dataset['Session Duration (mins)'] = (dataset['Logout Time'] - dataset['Login Time']).dt.total_seconds() /60
features = ['Login Hour','Logout Hour', 'Usage Time (mins)', 'Count of Survey Attempts']
x = dataset[features].values
y = dataset['Count of Survey Attempts']


# In[510]:


print(dataset[['State', 'Region', 'Speciality']].isnull().sum())


# In[522]:


dataset['State'] = dataset['State'].astype(str)
dataset['Region'] = dataset['Region'].astype(str)
dataset['Speciality'] = dataset['Speciality'].astype(str)


# In[532]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Convert categorical columns to numeric
categorical_columns = ['State', 'Region', 'Speciality']
for col in categorical_columns:
    dataset[col] = label_encoder.fit_transform(dataset[col])

print(dataset.head())


# In[534]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[1,5,6])], remainder = 'passthrough')


# In[540]:


print(dataset[['Login Hour', 'Logout Hour', 'Usage Time (mins)']].dtypes)


# In[542]:


features = ['Login Hour', 'Logout Hour', 'Session Duration (mins)', 'Count of Survey Attempts']
X = dataset[features]
print(X.dtypes)


# In[544]:


try:
    X_transformed = ct.fit_transform(X)
except Exception as e:
    print("Error during transformation:", e)
    print("Problematic data:", X)


# In[546]:

dataset['Login Hour'] = pd.to_datetime(dataset['Login Time'], errors='coerce').dt.hour
dataset['Logout Hour'] = pd.to_datetime(dataset['Logout Time'], errors='coerce').dt.hour
dataset['Session Duration (mins)'] = (dataset['Logout Time'] - dataset['Login Time']).dt.total_seconds() / 60

print(dataset[['Login Hour', 'Logout Hour', 'Session Duration (mins)']].isnull().sum())


# In[548]:


print(X.dtypes)
print(X.head())


# In[550]:


X


# In[552]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)


# In[554]:


Y


# In[556]:


print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)


# In[558]:


print(dataset['Login Time'].dtype)
print(dataset['Logout Time'].dtype)


# In[560]:


print(dataset['Login Time'].isnull().sum())
print(dataset['Logout Time'].isnull().sum())


# In[562]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)


# In[564]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)
accuracy = accuracy_score(Y_test, model.predict(X_test))


# In[566]:


def predict_doctors(model, time, dataset):
    hour = pd.to_datetime(time).hour
    filtered_dataset = dataset[dataset['Hour'] == hour]
    X = filtered_dataset[['Hour', 'Session Duration (mins)', 'Count of Survey Attempts']]
    predictions = model.predict(X)
    responsive_doctors = filtered_dataset[predictions == 1]['NPI']


# In[568]:


with open("app.py", "w") as f:
    f.write('''
import streamlit as st

st.title('Doctor Survey Response Predictor')

st.sidebar.header('Enter a time to predict (HH:MM)')
time_input = st.sidebar.text_input('Time (e.g., 14:00)', '14:00')

st.write('You entered:', time_input)
    ''')


# In[ ]:


import streamlit as st
st.title('Doctor Survey Response Predictor')
time_input = st.sidebar.text_input('Enter a time to predict (HH:MM)', '14:00')
st.write('You entered:', time_input)


# In[ ]:


def load_data():
    df = load_data()
    X, y, NPIs = preprocess_data(df)
    model, accuracy = train_model(X, y)

st.sidebar.subheader(f'Model Accuracy: {accuracy:.2f}')


# In[ ]:


if st.sidebar.button('Predict'):
    results = predict_doctors(model, time_input, df)
    st.write('Doctors likely to respond at this time:')
    st.dataframe(results)

    csv = results.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='responsive_doctors.csv',
        mime='text/csv',)


# In[ ]:


data = {'Doctor NPI': ['12345', '67890'], 'Response Probability': [0.85, 0.72]}
df = pd.DataFrame(data)

st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False),
    file_name='responsive_doctors.csv',
    mime='text/csv',
)


# In[ ]:


dataset['Login Time'] = pd.to_datetime(dataset['Login Time'])
dataset['Count of Survey Attempts'] = pd.to_datetime(dataset['Count of Survey Attempts'])

dataset['Response Time (mins)'] = (dataset['Count of Survey Attempts'] - dataset['Login Time']).dt.total_seconds() / 60

print(dataset[['NPI', 'Login Time', 'Count of Survey Attempts', 'Response Time (mins)']])


# In[ ]:


dataset['Usage Time (mins)'] = pd.to_datetime(dataset['Usage Time (mins)'], errors='coerce')

time_input = st.text_input("Enter a time to predict (HH:MM):", "14:00")

selected_time = pd.to_datetime(time_input, format='%H:%M').time()

responsive_doctors = dataset[dataset['Usage Time (mins)'].dt.time == selected_time]

st.write("Doctors likely to respond at", time_input)
st.dataframe(responsive_doctors[['NPI', 'State', 'Usage Time (mins)', 'Speciality']])


# In[ ]:


dataset.columns = dataset.columns.str.strip()

print("Cleaned columns:", dataset.columns)


# In[ ]:


if 'Response' in dataset.columns:
    print("Found 'Response' column!")
else:
    print("Check for hidden characters or typos in column names.")


# In[ ]:


if not responsive_doctors.empty:
    st.success(f"Found {len(responsive_doctors)} responsive doctors at {time_input}.")
    st.table(responsive_doctors[['Doctor NPI', 'Doctor Name', 'response_probability']])
else:
    st.warning(f"No doctors found for the time {time_input}.")


# In[ ]:

if not responsive_doctors.empty:
    csv = responsive_doctors.to_csv(index=False)
    st.download_button(
        label="Download Doctor Predictions",
        data=csv,
        file_name=f"responsive_doctors_{time_input.replace(':', '-')}.csv",
        mime='text/csv'
    )


# In[ ]:


st.title("🩺 Doctor Survey Response Predictor")

st.sidebar.header("Predict Doctor Responses")
time_input = st.sidebar.text_input("Enter Time (HH:MM)", "14:00")

st.markdown("""
This app predicts which doctors are most likely to respond to survey invitations at a given time.
Simply enter a time, and download a CSV of responsive doctors!
""")

st.subheader("Prediction Results")
st.dataframe(responsive_doctors)

if not responsive_doctors.empty:
    st.download_button("Download CSV", csv, "responsive_doctors.csv", "text/csv")

