# Improving Customer Satisfaction: Sentiment Analysis on Customer Feedback for an App Designed for Online Classes and Video Conferencing Using BERT

## Abstract
In this project, we analyzed customer feedback for an app designed for online classes and video conferencing. We performed sentiment analysis on the text data and created a new label column to balance the dataset, as all original labels were positive. We also generated negative samples from positive samples to increase the size of the negative class. The resulting dataset was balanced by oversampling the positive class. We split the dataset into train and test using the train_test_split function and trained a BERT base classifier model on the training data. Finally, we evaluated the performance of the model on the test data using the classification report, which showed an overall accuracy of 84%.

## Problem Statement
The goal of this project is to perform sentiment analysis on customer feedback for an app designed for online classes and video conferencing. The original dataset had only positive labels, which needed to be balanced by generating negative samples from positive ones. We used the resulting dataset to train a BERT base classifier model to predict the sentiment of customer feedback accurately. The model's performance was evaluated on the test data, and we aimed to achieve a high accuracy score.

## Contents
- Sentiment Analysis and Labeling using VADER
- Exploratory Data Analysis
- Sentiment Analysis of Customer Feedback with Histogram Visualization
- Top 10 Most Frequent Words in "reason" Column after Stopword Removal with Bar Chart Visualization
- Co-occurrence of Top 30 Most Frequent Words in Customer Feedback Dataset with Heatmap Visualization
- Distribution of Sentiment Labels in Customer Feedback Dataset as a Pie Chart
- Topic Modeling of "reason" Column in Customer Feedback Dataset with LDA Algorithm
- Negation Generation for Positive Examples
- Oversampling to Address Class Imbalance in a Dataset
- Train Test Split
- BERT Preprocessing and Encoding Layers Initialization
- BERT-based Sentiment Analysis Model Architecture
- Model Compilation and Metrics Selection
- Summary

## Usage
You can follow the code blocks and explanations provided in this repository to understand and replicate the steps involved in performing sentiment analysis on customer feedback. Each code block is accompanied by a description and visualizations to help you analyze and interpret the results.

## Dependencies
- Python (version X.X)
- Pandas (version X.X)
- NLTK (version X.X)
- Seaborn (version X.X)
- TextBlob (version X.X)
- TensorFlow (version X.X)
- TensorFlow Hub (version X.X)
- Other required dependencies

## Installation
1. Clone the repository: `git clone https://github.com/your-username/your-repository.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Contributing
Contributions to this project are welcome. You can contribute by adding new features, fixing bugs, or improving the documentation. Please open a pull request with your changes, and they will be reviewed.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgments
We would like to acknowledge the contributions of [Author's Name] for their valuable insights and guidance in completing this project.

## Contact
For any questions or inquiries, please contact [Your Name] at [your-email@example.com].

