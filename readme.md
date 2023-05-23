# Improving Customer Satisfaction: Sentiment Analysis on Customer Feedback for an App Designed for Online Classes and Video Conferencing Using BERT



## Objective
The objective of this project is to improve customer satisfaction by performing sentiment analysis on customer feedback for an app designed for online classes and video conferencing. The analysis will be conducted using BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art natural language processing model.

## Data
The initial dataset consisted of customer feedback collected from users of the app. However, all the rows were initially labeled as 1, indicating positive sentiment, and there were no rows labeled as 0, indicating negative sentiment. To address this issue, sentiment analysis was performed on the dataset, and the rows were labeled as 1 and 0 based on the sentiment expressed in the text. To improve the representation of negative sentiment, negation generation techniques were applied. 

Due to the initial class imbalance, where positive labels were more prevalent than negative labels, oversampling techniques were employed to balance the dataset. This involved creating additional instances of the positive class to match the number of instances in the negative class.

## Methodology
The project utilizes BERT, a pre-trained language model known for its contextual understanding of text. BERT was fine-tuned using the sentiment-labeled dataset to train a sentiment analysis model. Fine-tuning involves updating the model's parameters using the specific task of sentiment analysis on the customer feedback data.

## Analysis
The complete analysis can be viewed [here.](https://github.com/Aravinth-Megnath/NLP-Project/blob/main/Customer_satisfaction.ipynb)


## Key Features
- Performed sentiment analysis on the initial dataset, labeling rows as 1 and 0 based on the sentiment expressed in the text.
- Utilized NLP techniques including Spacy, NLTK, and TensorFlow for sentiment analysis and machine learning.
- Utilized Plotly and Word Cloud for exploratory data analysis (EDA) to gain insights into the customer feedback dataset.
- Applied negation generation techniques to enhance the representation of negative sentiment in the dataset.
- Addressed class imbalance by oversampling the positive class to achieve a balanced dataset.
- Tuned the model for high precision in identifying positive sentiments and high recall in identifying negative sentiments.
- Developed a BERT-based model for sentiment analysis on a balanced dataset, achieving an accuracy of 84% with a focus on high precision for positive sentiments and high recall for negative sentiments.

## Exploratory Data Analysis

### Visualization of Label Distribution After Sentiment Analysis

![Label Distribution](https://github.com/Aravinth-Megnath/NLP-Project/assets/120720408/0fcaa757-dd55-4491-8d44-9b2ae3284371)
  
Caption: Visualizing the distribution of sentiment labels after performing sentiment analysis on the customer feedback.

### Word Cloud
![Word Cloud Visualization](https://github.com/Aravinth-Megnath/NLP-Project/assets/120720408/fa862685-fc74-4f8e-a07b-cd41aef7c424)



### Top 10 Most Frequent Words in "reason" Column after Stopword Removal

![Top 10 Words](https://github.com/Aravinth-Megnath/NLP-Project/assets/120720408/3097dc06-704d-4cdd-9a93-09c87ef6a092)

Caption: Bar chart visualization showing the top 10 most frequent words in the "reason" column after removing stopwords.

### Sentiment Analysis of Customer Feedback with Histogram Visualization

![Sentiment Analysis](https://github.com/Aravinth-Megnath/NLP-Project/assets/120720408/0a461287-cf39-42eb-8a37-89805f8070ef)
 
Caption: Histogram visualization of the sentiment scores after performing sentiment analysis on the customer feedback.


### Co-occurrence of Top 30 Most Frequent Words in Customer Feedback Dataset with Heatmap Visualization

![Co-occurrence Heatmap](https://github.com/Aravinth-Megnath/NLP-Project/assets/120720408/501fa775-0883-4c28-9280-f0fe3bdecd44)
 
Caption: Heatmap visualization showing the co-occurrence of the top 30 most frequent words in the "reason" column.

### Distribution of Sentiment Labels in Customer Feedback Dataset as a Pie Chart

![Sentiment Distribution](https://github.com/Aravinth-Megnath/NLP-Project/assets/120720408/cc65b18a-4c1e-4431-966f-f4479c7b6b4f)

Caption: Pie chart visualization showing the distribution of sentiment labels in the customer feedback dataset.


## Classification Results

The sentiment analysis model achieved the following performance metrics on the test dataset:

          precision    recall  f1-score   support

       0       0.95      0.70      0.80       659
       1       0.77      0.97      0.86       702
          
          accuracy                 0.84      1361


The precision, recall, and F1-score are commonly used metrics to evaluate the performance of a binary classification model. In this case, class 0 represents negative sentiments and class 1 represents positive sentiments.

The model achieved an accuracy of 84%, indicating the percentage of correctly classified instances out of the total test dataset. The precision for class 0 is 95%, meaning that when the model predicted a sentiment as negative, it was correct 95% of the time. The recall for class 1 is 97%, indicating that the model successfully identified 97% of the positive sentiments in the dataset. The F1-score combines precision and recall, providing a single metric to assess the model's overall performance.

These results suggest that the sentiment analysis model has a good ability to classify customer feedback accurately, with a focus on correctly identifying positive sentiments.



## Usage
To replicate the project, follow these steps:

1. Install the required dependencies specified in the `requirements.txt` file.
2. Prepare the initial dataset by collecting customer feedback data from the app and organizing it in a suitable format.
3. Perform exploratory data analysis (EDA) using Plotly and WordCloud to gain insights into the dataset.
4. Perform sentiment analysis on the initial dataset, labeling the rows as 1 and 0 based on the sentiment expressed in the text.
5. Apply negation generation techniques to enhance the representation of negative sentiment in the dataset.
6. Address class imbalance by oversampling the positive class to achieve a balanced dataset.
7. Develop a BERT-based model using TensorFlow for sentiment analysis, fine-tuning it on the sentiment-labeled dataset.
8. Evaluate the performance of the model using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score.
9. Utilize the trained model to perform sentiment analysis on new customer feedback, predicting the sentiment expressed in the text.
10. Analyze the results and gain insights into customer satisfaction levels, identifying areas for improvement based on the sentiment analysis.


## Future Enhancements
There are several possible future enhancements for this project:

- Incorporate additional features or data sources, such as user demographics, app usage statistics, or user behavior patterns, to enrich the sentiment analysis and gain deeper insights into customer satisfaction.
- Explore other state-of-the-art natural language processing models, such as GPT-3 or Transformer-XL, to compare their performance with BERT and potentially achieve even better sentiment analysis results.
- Implement a real-time sentiment analysis system that continuously processes incoming customer feedback, providing instant insights for effective decision-making and customer satisfaction improvements.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
We would like to express our gratitude to the developers of BERT and the creators of the customer feedback dataset used in this project. Their contributions have been instrumental in the successful execution of this sentiment analysis project.


## Feedback

Your feedback is important to me! If you have any suggestions, questions, or feedback regarding this project, please feel free to reach out to me at:

- Email: [aravinthmegnath@gmail.com]
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://www.linkedin.com/in/aravinth-meganathan-200667a1/)

I appreciate your valuable input and look forward to hearing from you!

