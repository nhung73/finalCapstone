# Import libraries: spacy, pandas, spacytextblob
import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob


try:
    # Load the language model (used md for more accuracy, rather than sm)
    nlp = spacy.load("en_core_web_md")

    # Add the SpacyTextBlob extension to the pipeline
    nlp.add_pipe("spacytextblob")

    # Manually register the sentiment extension attribute
    try:
        # Attempt to access the 'sentiment' attribute of the Token class
        spacy.tokens.Token.sentiment
    except AttributeError:
        # If the attribute does not exist, execute this block of code
        # Add the 'sentiment' extension attribute to the Token class
        spacy.tokens.Token.set_extension("sentiment", default=None, force=True)

    # Create function to preprocess text data and remove stopwords
    def preprocess_text(text):
        # Convert text to lowercase and remove leading/trailing whitespaces
        text = text.lower().strip()
        # Process the text using spaCy
        doc = nlp(text)
        # Filter out stopwords and punctuation
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        # Join the tokens back into a single string
        clean_text = " ".join(tokens)
        return clean_text

    # Create function for sentiment analysis, use sentiment and polarity
    def predict_sentiment(review):
        clean_review = preprocess_text(review)
        doc = nlp(clean_review)
        polarity = doc._.blob.polarity
        sentiment = doc._.blob.sentiment
        return polarity, sentiment

    # Load the dataset "amazon_product_reviews.csv"
    data = pd.read_csv("amazon_product_reviews.csv")

    # Drop rows with missing values in the 'reviews.text' column
    data.dropna(subset=["reviews.text"], inplace=True)

    # Condition if there is no data
    if data.empty:
        print(f"No valid data after dropping rows with missing values.")
    else:
        # Apply sentiment analysis to each review in the 'review.text' column
        data["polarity"] = data["reviews.text"].apply(lambda x: predict_sentiment(x)[0])
        data["sentiment"] = data["reviews.text"].apply(lambda x: predict_sentiment(x)[1])

        # Test the sentiment analysis function on a few sample product reviews and print the results
        print("\nTesting sentiment analysis on sample product reviews:")
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

        # Select two reviews
        review1 = data["reviews.text"].iloc[140]
        review2 = data["reviews.text"].iloc[787]

        # Preprocess the selected reviews
        clean_review1 = preprocess_text(review1)
        clean_review2 = preprocess_text(review2)

        # Process the preprocessed reviews using spaCy
        doc1 = nlp(clean_review1)
        doc2 = nlp(clean_review2)

        # Calculate the similarity between the preprocessed reviews
        similarity_score = doc1.similarity(doc2)

        print(f"Polarity of review 1:", {review1}, data["polarity"].iloc[140])
        print(f"Polarity of review 2:", {review2}, data["polarity"].iloc[787])
        print(f"Sentiment of review 1:", {review1}, data["sentiment"].iloc[140])
        print(f"Sentiment of review 2:", {review2}, data["sentiment"].iloc[787])
        print(f"Similarity between the two reviews:", similarity_score)

# Raise exception if error
except Exception as e:
    print(f"Error:", e)
