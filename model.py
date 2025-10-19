import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv('dataset.csv')

# Separate features and target variable
x = df.iloc[:, :-1]  # All columns except the last one (features)
y = df.iloc[:, -1]   # The last column (target variable)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)

# Preprocessing step: One-hot encode categorical columns
trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])  # Update with actual column names
], remainder='passthrough')

# Create a pipeline: Apply preprocessing and then train a RandomForestClassifier
ra_pipe = Pipeline([
    ('step1', trf),
    ('step2', RandomForestClassifier())  # You can adjust the classifier parameters here if needed
])

# Train the model
ra_pipe.fit(x_train, y_train)

# Predict on the test set
ra_y_pred = ra_pipe.predict(x_test)

# Save the trained model to a file
pickle.dump(ra_pipe, open('ra_pipe.pkl', 'wb'))

# Print the accuracy of the model
print(f'Accuracy: {accuracy_score(y_test, ra_y_pred)}')
