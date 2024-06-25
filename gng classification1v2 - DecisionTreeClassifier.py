import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import csv
import os
import pickle

exist = 0
le = LabelEncoder()
tfidf = TfidfVectorizer(stop_words="english")
model = DecisionTreeClassifier()

def create_model(model):
    with open('model.h5', 'wb') as file:
        pickle.dump(model, file)

def load_data():
    #load dataset
    df = pd.read_csv("E:\\in\\Task1\\vehicle_queries.csv")
    return df

def convert_data2num(df):
    # Seperating feature and target
    X = df['query']
    y = df['generalized']

    #Encoding target column
    #le = LabelEncoder()
    y = le.fit_transform(y)

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #tfidf = TfidfVectorizer(stop_words="english")
    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf= tfidf.transform(X_test)
    return [X_train_tf, X_test_tf, y_train, y_test]

def train_model():
    df = load_data()
    X_train_tf, X_test_tf, y_train, y_test = convert_data2num(df)
    #model = DecisionTreeClassifier()
    model.fit(X_train_tf, y_train)
    create_model(model)

def test_model():
    
    global exist
    
    train_model()
    
    # Query
    query = input("Enter your query: ")
    #print(query)

    # Transform data
    tf_query= tfidf.transform([query.lower()])
    pred = model.predict(tf_query)

    # Display
    print({"Query Type":le.inverse_transform(pred)[0]})
    response_data = le.inverse_transform(pred)[0]

    # Write query to there respective file
    def write_to_file(file_path, query):
        
        global exist
        
        # Check if the file exists
        file_exists = os.path.isfile(file_path)

        # If the file does not exist, create it and write the header
        if not file_exists:
            with open(file_path, mode='w', newline='\n') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(['Query'])
        
        df = pd.read_csv(file_path)
        
        # Check if query already exists in any row of the dataframe
        if query in df['Query'].values:
            print(f'already exists - {query}')
            exist = 1
        else:
            # Appending to the CSV file
            with open(file_path, mode='a', newline='\n') as file:
                writer = csv.writer(file)
                writer.writerow([query])
            
            # Update the dataframe with the new query
            df = df.append({'Query': query}, ignore_index=True)
            # Write the updated dataframe back to the CSV file
            df.to_csv(file_path, index=False)

    if response_data == 'yes':
        # File path for the CSV file
        file_path = "generalized.csv"

        # Add query to generalized.csv file
        write_to_file(file_path, query)
        
        # Display message
        if exist == 1:
            print(f"Data not appended to {file_path}")
            exist=0
        else:
            print(f"Data appended to {file_path}")
    else:
        # File path for the CSV file
        file_path = "non_generalized.csv"
        
        # Add query to generalized.csv file
        write_to_file(file_path, query)
        
        # Display message
        if exist == 1:
            print(f"Data not appended to {file_path}")
            exist=0
        else:
            print(f"Data appended to {file_path}")


test_model()