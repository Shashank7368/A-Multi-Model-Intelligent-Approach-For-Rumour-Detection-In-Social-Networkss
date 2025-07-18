from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import RomerSourceIdentification

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report


# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})



def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'balanced_rumor_dataset.csv'
    df = pd.read_csv(path, nrows=100)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
path = settings.MEDIA_ROOT + "//" + 'balanced_rumor_dataset.csv'
df = pd.read_csv(path)
value_counts = df['is_rumor'].value_counts()
# print(value_counts)
# import matplotlib.pyplot as plt
# # Plotting the bar plot
# value_counts.plot(kind='bar')
# # Adding labels and title
# plt.xlabel('Categories')
# plt.ylabel('Counts')
# plt.title('Value Counts of Your Column')
# # Displaying the plot
# plt.show()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['is_rumor'], test_size=0.2, random_state=42)
# Vectorize the tweets using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

def training(request):  
    # Adding labels and title
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title('Value Counts of Your Column')
    # Displaying the plot
    plt.show()    
    predictions = nb_classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")      
    nb = classification_report(y_test,predictions,output_dict=True)
    nb = pd.DataFrame(nb).transpose()    
    nb = pd.DataFrame(nb)
    return render(request,"users/training.html",{'nb':nb.to_html,'acc':accuracy})

def prediction(request):
    if request.method == 'POST':
        single_tweet = request.POST.get('tweets') 
        print(single_tweet)      
        single_tweet_tfidf = tfidf_vectorizer.transform([single_tweet])
        print('manohar',single_tweet_tfidf)
        # Make prediction
        single_prediction = nb_classifier.predict(single_tweet_tfidf)
        print(single_prediction)
        # Print prediction
        print(f'Tweet: {single_tweet} - Predicted Emotion: {single_prediction[0]}')
        if single_prediction[0] == 0:
            single_prediction='its not a Rumor'
        elif single_prediction[0] == 1:
            single_prediction='Its a Rumor'
        return render(request, 'users/predictForm.html', {'output':single_prediction})
    return render(request, 'users/predictForm.html', {})
        
        
