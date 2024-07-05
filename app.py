import streamlit as st
import pickle
import pandas as pd
import nltk
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        ps=PorterStemmer() 
        y.append(ps.stem(i))
        
    return " ".join(y)
    
    

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier:-")
input_sms=st.text_area('Enter your SMS/Email here')

#display
if st.button('Predict'):
    #preprocess
    transformed_text=transform_text(input_sms) 
    
    #vectorize
    vector_input=pd.DataFrame(tfidf.transform([transformed_text]).toarray())
    
    #prediction
    result=model.predict(vector_input)[0] 
    
    if result==0:
        st.header('Not Spam')
        
    else:
        st.header('Spam')
