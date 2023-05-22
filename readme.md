# Improving Customer Satisfaction: Sentiment Analysis on Customer Feedback for an App Designed for Online Classes and Video Conferencing Using BERT

## Objective
The objective of this project is to improve customer satisfaction by performing sentiment analysis on customer feedback for an app designed for online classes and video conferencing. The analysis will be conducted using BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art natural language processing model.

## Data
The dataset consists of customer feedback collected from users of the app. The feedback includes text-based reviews and comments, expressing their opinions and experiences with the app's features, functionality, and overall user experience.

## Methodology
The project utilizes BERT, a pre-trained language model known for its contextual understanding of text. BERT will be fine-tuned using the customer feedback dataset to train a sentiment analysis model. Fine-tuning involves updating the model's parameters using the specific task of sentiment analysis on the customer feedback data.

## Key Features
- Conducted extensive exploratory data analysis using Plotly and WordCloud to gain insights into the customer feedback dataset.
- Applied data balancing techniques such as negation generation and oversampling using the Spacy and NLTK frameworks to address class imbalance.
- Tuned the model for high precision in identifying positive sentiments and high recall in identifying negative sentiments.
- Developed a BERT-based model for sentiment analysis on a balanced dataset, achieving an accuracy of 84% with a focus on high precision for positive sentiments and high recall for negative sentiments.

## Usage
To replicate the project, follow these steps:

1. Install the required dependencies specified in the `requirements.txt` file.
2. Prepare the dataset by collecting customer feedback data from the app and organizing it in a suitable format.
3. Conduct exploratory data analysis (EDA) using Plotly and WordCloud to gain insights into the dataset.
4. Apply data balancing techniques such as negation generation and oversampling to address class imbalance.
5. Develop a BERT-based model using TensorFlow for sentiment analysis, fine-tuning it on the customer feedback dataset.
6. Evaluate the performance of the model using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score.
7. Utilize the trained model to perform sentiment analysis on new customer feedback, predicting the sentiment expressed in the text.
8. Analyze the results and gain insights into customer satisfaction levels, identifying areas for improvement based on the sentiment analysis.

## Future Enhancements
There are several possible future enhancements for this project:

- Incorporate additional features or data sources, such as user demographics, app usage statistics, or user behavior patterns, to enrich the sentiment analysis and gain deeper insights into customer satisfaction.
- Explore other state-of-the-art natural language processing models, such as GPT-3 or Transformer-XL, to compare their performance with BERT and potentially achieve even better sentiment analysis results.
- Implement a real-time sentiment analysis system that continuously processes incoming customer feedback, providing instant insights for effective decision-making and customer satisfaction improvements.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements
We would like to express our gratitude to the developers of BERT and the creators of the customer feedback dataset used in this project. Their contributions have been instrumental in the successful execution of this sentiment analysis project.

## Contact
For further information or inquiries
