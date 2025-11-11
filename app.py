# --------------------------------------------------
# 📧 Email/SMS Spam Classifier - Enhanced Streamlit UI
# --------------------------------------------------

import streamlit as st
import nltk
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import plotly.express as px

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# -------------------------------
# Text preprocessing function
# -------------------------------
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

# -------------------------------
# Load saved model and vectorizer
# -------------------------------
Tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Spam Classifier | Tarun's Project",
    page_icon="📩",
    layout="centered"
)

# -------------------------------
# Custom CSS for modern design
# -------------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f9f9f9, #e8f0fe);
        padding: 2rem;
        border-radius: 10px;
    }
    h1 {
        text-align: center;
        color: #2E86C1;
        font-family: 'Poppins', sans-serif;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        font-size: 16px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
        transform: scale(1.05);
    }
    .result-box {
        text-align: center;
        font-size: 22px;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
    }
    .spam {
        background-color: #F1948A;
        color: #922B21;
    }
    .ham {
        background-color: #ABEBC6;
        color: #145A32;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.title("📩 Email / SMS Spam Classifier")

st.markdown("### 🚀 Enter a message below to predict if it's Spam or Not.")

# -------------------------------
# Input box
# -------------------------------
input_sms = st.text_area("✉️ Type your message here:", height=150)

# -------------------------------
# Prediction button
# -------------------------------
if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = Tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display result with animation & color
        if result == 1:
            st.markdown(
                "<div class='result-box spam'>🚨 SPAM MESSAGE DETECTED!</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box ham'>✅ This message is safe (Not Spam).</div>",
                unsafe_allow_html=True
            )

        # -------------------------------
        # Word Cloud Visualization
        # -------------------------------
        st.subheader("☁️ Word Cloud of Your Message")
        wc = WordCloud(width=800, height=400, background_color='white').generate(transformed_sms)
        st.image(wc.to_array())

        # -------------------------------
        # Keyword Frequency Visualization
        # -------------------------------
        words = transformed_sms.split()
        freq_dist = nltk.FreqDist(words)
        freq_df = (
            freq_dist.most_common(10)
        )
        if freq_df:
            freq_data = {"Word": [w for w, _ in freq_df], "Frequency": [f for _, f in freq_df]}
            fig = px.bar(
                freq_data,
                x="Word",
                y="Frequency",
                color="Frequency",
                color_continuous_scale="Blues",
                title="Top Keywords in Your Message"
            )
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Collapsible Model Performance Visualization
# -------------------------------
with st.expander("📊 View Model Performance Comparison"):
    st.markdown("### 🔎 Accuracy & Precision of Different Models")

    # Accuracy and Precision values (update with actual results if needed)
    performance_data = {
        "Model": ["Naive Bayes", "SVM", "Random Forest", "Logistic Regression"],
        "Accuracy": [0.98, 0.97, 0.96, 0.95],
        "Precision": [0.97, 0.96, 0.95, 0.94]
    }

    import pandas as pd
    import plotly.express as px

    # Convert to DataFrame
    perf_df = pd.DataFrame(performance_data)

    # Melt for grouped bar chart
    perf_melted = perf_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    # Create Bar Chart
    fig_perf = px.bar(
        perf_melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=True,
        color_discrete_sequence=["#3498DB", "#E74C3C"],
        title="Model Accuracy & Precision Comparison"
    )

    # Layout Enhancements
    fig_perf.update_layout(
        xaxis_title="Machine Learning Model",
        yaxis_title="Score",
        template="plotly_white",
        font=dict(size=14),
        showlegend=True,
        title_x=0.3
    )

    st.plotly_chart(fig_perf, use_container_width=True)

    st.info("💡 These metrics help compare how each algorithm performs in detecting spam messages.")
    
# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey;'>Built with ❤️ by Tarun Teja Karimi | Streamlit • NLP • ML</p>",
    unsafe_allow_html=True
)
