import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB  
import streamlit as st
data = pd.read_csv('spam.csv')
#data cleaning
data['Category'] = data['Category'].replace("ham","Not Spam")
data.drop_duplicates(inplace=True)
data.drop_duplicates((data.isnull()))

#split train and test data  
mess = data['Message'] #input column
cat = data['Category'] #output column

(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess,cat,test_size=0.2)


cv = CountVectorizer(stop_words="english")
features = cv.fit_transform(mess_train)

#create and train model
model = MultinomialNB()
model.fit(features,cat_train)

#test our model
features_test = cv.transform(mess_test)
print(model.score(features_test,cat_test)) #how our trained model performed or use our model for test datas

#predict 
def answer(message):
    input = cv.transform([message]).toarray()
    res = model.predict(input)
    return res 

st.header('Spam Detection')
input_msg = st.text_input("Enter the message")
if st.button('Validate'):
    output = answer(input_msg)
    st.markdown((output))
