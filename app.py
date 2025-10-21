import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    y = []
    for i in message:
        if i.isalnum():
            y.append(i)

    message = y[:]
    y.clear()

    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    message = y[:]
    y.clear()

    for i in message:
        y.append(ps.stem(i))

    return " ".join(y)


Tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter your SMS")

if st.button("Predict"):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize
    vector_input = Tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)

    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")




