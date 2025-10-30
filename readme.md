# 🧠 Intelligent Customer Feedback Analysis System using AI

> An AI-driven solution that analyzes, summarizes, and predicts customer sentiment from feedback collected through emails, chat logs, and social media comments — transforming raw feedback into actionable insights for better decision-making.

---

## 🎯 Objective

To **design and develop an AI-based system** capable of:
- Analyzing and classifying customer sentiments (Positive, Negative, Neutral)
- Summarizing lengthy feedback into concise, meaningful overviews
- Predicting future customer satisfaction trends
- Providing an interactive interface for real-time visualization and exploration

---

## 🧩 Project Workflow

### **1️⃣ Data Handling & Preprocessing (25 Marks)**
- Collected/simulated **1,000+ customer feedback records** in `.csv` format  
- Cleaned and prepared data using:
  - Duplicate & noise removal  
  - Tokenization, lemmatization, and stopword removal  
  - Special character handling and normalization  
- **Deliverable:** `data_preprocessing.ipynb` / `data_preprocessing.py`

---

### **2️⃣ Sentiment Classification Model (30 Marks)**
- Implemented advanced NLP-based classification using:
  - **BERT / DistilBERT / LSTM with GloVe embeddings**
- Classified customer feedback into three categories:
  - **Positive**, **Negative**, and **Neutral**
- Evaluated with **Accuracy**, **Precision**, **Recall**, and **F1-Score**
- **Deliverable:** `sentiment_model.ipynb` and saved model `sentiment_model.pkl`

---

### **3️⃣ Text Summarization (20 Marks)**
- Built summarization pipeline using:
  - **Transformer-based models** (T5 / BART) for abstractive summaries  
  - **Extractive method** using TF-IDF + cosine similarity
- Generated:
  - **Short summaries** for quick overviews  
  - **Detailed summaries** for in-depth analysis
- **Deliverable:** `text_summarization.ipynb`

---

### **4️⃣ Predictive Insight Generation (15 Marks)**
- Identified recurring issues and key sentiment patterns  
- Applied **time-series forecasting** (Prophet / ARIMA) to predict:
  - Future sentiment distribution
  - Monthly satisfaction trend
- Visualized insights using **Matplotlib**, **Seaborn**, and **Plotly**
- **Deliverable:** `AI_insights_report.pdf`

---

### **5️⃣ Deployment with Streamlit/Flask (10 Marks)**
- Developed an **interactive web app** allowing users to:
  - Upload feedback datasets
  - View real-time sentiment results
  - Display summarization and predictive graphs
- Built using **Streamlit / Flask**, **Plotly**, and **Matplotlib**
- **Deliverable:** Running web app demo and GitHub repository link

---

### **⭐ Bonus Feature (Optional – 10 Marks)**
- Integrated an **AI Chatbot** (OpenAI / Hugging Face) to:
  - Respond to user queries based on summarized feedback  
  - Suggest actionable improvements for customer satisfaction

---

## 🧰 Tech Stack

| Category | Tools & Libraries |
|-----------|------------------|
| Programming | Python |
| NLP & ML | NLTK, spaCy, Hugging Face Transformers |
| Deep Learning | PyTorch / TensorFlow / Keras |
| Models | BERT, DistilBERT, T5, BART, LSTM |
| Data Handling | pandas, NumPy, scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Forecasting | Prophet, ARIMA |
| Deployment | Streamlit / Flask |

---

## 📂 Project Structure
---
```
sentiment-suite/
├── data/ # Raw & processed datasets
├── src/
│ ├── preprocessing/ # Data cleaning scripts
│ ├── sentiment_model/ # Model training & evaluation
│ ├── summarization/ # Summarization modules
│ ├── insights/ # Predictive analysis & visualization
│ └── app/ # Streamlit or Flask app
├── notebooks/ # EDA & model development notebooks
├── reports/
│ └── AI_insights_report.pdf
├── requirements.txt
├── README.md
└── .gitignore

```
---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|--------------|
| Accuracy | Overall correctness of predictions |
| Precision | True positive rate for each sentiment |
| Recall | Sensitivity across sentiment categories |
| F1-Score | Harmonic mean of Precision & Recall |
| ROUGE Score | Evaluation for text summarization |

---

## 🚀 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/guptaraghav81/sentiment-suite.git
cd sentiment-suite

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python src/preprocessing/data_preprocessing.py

# Train and evaluate sentiment model
python src/sentiment_model/train_model.py

# Launch web app
streamlit run src/app/app.py
```
📈 Deliverables Summary <br>
Part	Description	File <br>
1	Cleaned Dataset & Preprocessing Code	data_preprocessing.py / .ipynb <br>
2	Trained Sentiment Model	sentiment_model.pkl <br>
3	Summarization Notebook	text_summarization.ipynb <br>
4	Insight Report	AI_insights_report.pdf <br>
5	Deployment App	app.py / Streamlit <br>
🔮 Future Enhancements 

Add multilingual sentiment support

Integrate topic modeling (LDA / BERTopic)

Build real-time monitoring dashboard

Expand chatbot capabilities for deeper conversational analytics

# ==========================================
# 📊 Sentiment Analysis EDA Visualization Code
# ==========================================

# ✅ Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

# Load your processed dataset (change path accordingly)
df = pd.read_csv("customer_feedback.csv")

# Ensure columns exist
# Assume columns: ['feedback', 'reason', 'label']
print(df.head())

# ==============================
# 1️⃣ Label Distribution
# ==============================
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df, palette='coolwarm')
plt.title("Distribution of Sentiment Labels")
plt.xlabel("Sentiment Label (0 = Negative, 1 = Positive)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("label_distribution.png", dpi=300)
plt.show()

# ==============================
# 2️⃣ Word Cloud
# ==============================
text = " ".join(df['feedback'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud Visualization of Customer Feedback")
plt.tight_layout()
plt.savefig("wordcloud.png", dpi=300)
plt.show()

# ==============================
# 3️⃣ Top 10 Most Frequent Words in 'reason' Column
# ==============================
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['reason'].astype(str))
word_freq = dict(zip(vectorizer.get_feature_names_out(), np.asarray(X.sum(axis=0)).ravel()))
sorted_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10])

plt.figure(figsize=(8, 4))
sns.barplot(x=list(sorted_words.keys()), y=list(sorted_words.values()), palette="viridis")
plt.title("Top 10 Most Frequent Words in 'reason' Column")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top10_words.png", dpi=300)
plt.show()

# ==============================
# 4️⃣ Sentiment Histogram (Distribution)
# ==============================
plt.figure(figsize=(6, 4))
sns.histplot(df['label'], kde=True, color='teal', bins=3)
plt.title("Histogram of Sentiment Distribution")
plt.xlabel("Sentiment (0=Negative, 1=Positive)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("sentiment_histogram.png", dpi=300)
plt.show()

# ==============================
# 5️⃣ Co-occurrence Heatmap (Top 30 Words)
# ==============================
# Create co-occurrence matrix
cv = CountVectorizer(max_features=30, stop_words='english')
X = cv.fit_transform(df['feedback'].astype(str))
Xc = (X.T * X)
Xc.setdiag(0)
co_occurrence_df = pd.DataFrame(Xc.toarray(), columns=cv.get_feature_names_out(), index=cv.get_feature_names_out())

plt.figure(figsize=(12, 8))
sns.heatmap(co_occurrence_df, cmap="YlGnBu")
plt.title("Co-occurrence Heatmap of Top 30 Words")
plt.tight_layout()
plt.savefig("cooccurrence_heatmap.png", dpi=300)
plt.show()

# ==============================
# 6️⃣ Pie Chart of Sentiment Distribution
# ==============================
label_counts = df['label'].value_counts().reset_index()
label_counts.columns = ['label', 'count']

fig = px.pie(label_counts, values='count', names='label',
             title='Distribution of Sentiment Labels (Pie Chart)',
             color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()


💼 AI & NLP Enthusiast | Data Science | Machine Learning | Deep Learning
