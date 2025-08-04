# Twitter_Sentiment_Analysis

## Project Overview
This project demonstrates an end-to-end machine learning pipeline for sentiment analysis on Twitter data related to airline tweets. The primary goal is to classify tweets as "positive" or "negative" to gain insights into public opinion and customer feedback.

## Key Concepts
Sentiment Analysis: Using Natural Language Processing (NLP) to determine the emotional tone of text.Text Preprocessing: A crucial step to clean raw text data by removing noise (e.g., URLs, symbols) and performing stemming.Feature Extraction: Converting text into a numerical format using a CountVectorizer so a model can process it.Supervised Learning: Training a Logistic Regression model on a labeled dataset to classify new, unlabeled tweets.

## Technical Stack
Language: Python
Libraries: pandas, numpy, re, nltk, scikit-learn, matplotlib, seaborn

## Pipeline & Results

The pipeline follows a logical flow:
1. Data Loading & Filtering: The project uses the "Airline dataset.csv" file, focusing on tweets with "positive" and "negative" sentiment labels.
2. Preprocessing: Raw tweet text is cleaned and transformed.
3. Model Training: A LogisticRegression model is trained on the processed data.
4. Evaluation: The model's performance is evaluated using a confusion matrix and classification report.The model achieved an overall accuracy of 91%. While it performed very well on the negative class (93% precision), it showed a lower recall (70%) for the positive class due to a class imbalance in the dataset (79.5% negative vs. 20.5% positive tweets).
 
## Conclusion & Future Work
The project successfully built an effective sentiment analysis model. However, to improve its performance on the minority positive class, future work could involve techniques like oversampling or exploring different model architectures to create a more balanced and robust classifier.
