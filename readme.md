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

<h3>### Visualization of Label Distribution After Sentiment Analysis </h3>
![Label Distribution](https://github.com/Aravinth-Megnath/NLP-Project/assets/120720408/0fcaa757-dd55-4491-8d44-9b2ae3284371)
*Caption: Visualizing the distribution of sentiment labels after performing sentiment analysis on the customer feedback.*

<h3>### Word Cloud </h3>
![Word Cloud Visualization](https://github.com/Aravinth-Megnath/NLP-Project/assets/120720408/fa862685-fc74-4f8e-a07b-cd41aef7c424)

<h3>### Top 10 Most Frequent Words in "reason" Column after Stopword Removal</h3>
![Top 10 Words](https://github.com/Aravinth-Megnath/NLP-Project/assets/120720408/3097dc06-704d-4cdd-9a93-09c87ef6a092)
*Caption: Bar chart visualization showing the top 10 most frequent words in the "reason" column after removing stopwords.*

<h3>### Sentiment Analysis of Customer Feedback with Histogram Visualization</h3>
![Sentiment Analysis](https://github.com/Aravinth-Megnath/NLP-Project/assets/120720408/0a461287-cf39-42eb-8a37-89805f8070ef)
*Caption: Histogram visualization of the sentiment scores after performing sentiment analysis on the customer feedback.*

<h3>### Co-occurrence of Top 30 Most Frequent Words in Customer Feedback Dataset with Heatmap Visualization</h3>
![Co-occurrence Heatmap](https://github.com/Aravinth-Megnath/NLP-Project/assets/120720408/501fa775-0883-4c28-9280-f0fe3bdecd44)
*Caption: Heatmap visualization showing the co-occurrence of the top 30 most frequent words in the "reason" column.*

<h3>### Distribution of Sentiment Labels in Customer Feedback Dataset as a Pie Chart</h3>
![Sentiment Distribution](https://github.com/Aravinth-Megnath/NLP-Project/assets/120720408/cc65b18a-4c1e-4431-966f-f4479c7b6b4f) <br>
*Caption: Pie chart visualization showing the distribution of sentiment labels in the customer feedback dataset.*



💼 AI & NLP Enthusiast | Data Science | Machine Learning | Deep Learning
