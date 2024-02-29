# Capstone Project - NLP Application

## Description
This project utilises Natural Language Processing (NLP) techniques to perform sentiment analysis on product reviews. Sentiment analysis is crucial for businesses to understand customer opinions, satisfaction levels, and overall product perception. By analysing the sentiment expressed in reviews, companies can gain valuable insights into customer preferences and areas for improvement. This application demonstrates the process of sentiment analysis using spaCy and provides functionalities to preprocess text data, predict sentiment polarity and sentiment labels, and calculate the similarity between reviews.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

## Installation
1. Clone the repository:
    ```
    git clone https://github.com/yourusername/yourrepository.git
    ```
2. Navigate to the project directory:
    ```
    cd yourrepository
    ```
3. Install the required libraries:
    ```
    pip install spacy pandas spacytextblob
    ```
4. Download the spaCy English language model:
    ```
    python -m spacy download en_core_web_md # The medium model had been used in this code
    ```

## Usage
1. Import the required libraries:
    ```python
    import spacy
    import pandas as pd
    from spacytextblob.spacytextblob import SpacyTextBlob
    ```
2. Load the language model and add the SpacyTextBlob extension:
    ```python
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("spacytextblob")
    ```
3. Define functions for preprocessing text and predicting sentiment:
    ```python
    def preprocess_text(text):
        ...
    
    def predict_sentiment(review):
        ...
    ```
4. Load the dataset and apply sentiment analysis:
    ```python
    data = pd.read_csv("your_file.csv") # Your file name, .csv file had been used in this code
    data.dropna(subset=["reviews.text"], inplace=True)
    data["polarity"] = data["reviews.text"].apply(lambda x: predict_sentiment(x)[0])
    data["sentiment"] = data["reviews.text"].apply(lambda x: predict_sentiment(x)[1])
    ```
5. Test sentiment analysis on sample reviews:
    ```python
    sample_reviews = [
        "This is the most amazing item. And very easy to use",
        "Just an average Alexa option. Does show a few things on screen but still limited.",
        "Absolutely Awesome! Great product, recommend that Everyone have one!",
        "The screen is too dark, and cannot adjust the brightness"
    ]
    for review in sample_reviews:
        polarity, sentiment = predict_sentiment(review)
        print(f"Review:", review)
        print(f"Polarity:", polarity)
        print(f"Sentiment:", sentiment)
        print()
    ```
6. Calculate similarity between reviews:
    ```python
    review1 = data["reviews.text"].iloc[140] # Adjust index
    review2 = data["reviews.text"].iloc[787] # Adjust index
    clean_review1 = preprocess_text(review1)
    clean_review2 = preprocess_text(review2)
    doc1 = nlp(clean_review1)
    doc2 = nlp(clean_review2)
    similarity_score = doc1.similarity(doc2)
    print(f"Similarity between the two reviews:", similarity_score)
    ```
