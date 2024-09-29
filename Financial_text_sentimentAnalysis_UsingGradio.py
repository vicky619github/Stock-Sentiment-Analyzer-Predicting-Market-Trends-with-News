# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:09:50 2024

@author: vickj
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 01:34:41 2024

@author: vickj
"""

import feedparser
from transformers import pipeline
import matplotlib.pyplot as plt
import gradio as gr

# Initialize the sentiment analysis pipeline
pipe = pipeline(task="text-classification", model="ProsusAI/finbert")

# Main function to perform sentiment analysis
def analyze_stock_sentiment(ticker, keyword):
    rss_url = f'https://finance.yahoo.com/rss/headline/?s={ticker}'
    feed = feedparser.parse(rss_url)

    total_score = 0
    num_articles = 0
    pos_count, neg_count, neu_count = 0, 0, 0
    articles = []  # Store the articles

    # Adjusted thresholds for more conservative sentiment classification
    positive_threshold = 0.05  # Lower positive threshold for stricter classification
    negative_threshold = -0.05  # Lower negative threshold for stricter classification

    # Analyze the feed for articles containing the keyword
    for i, entry in enumerate(feed.entries):
        if keyword.lower() not in entry.summary.lower():
            continue

        # Store the relevant article information
        articles.append(f'Title: {entry.title}\nLink: {entry.link}\nPublished: {entry.published}')
        
        # Analyze the entire summary without truncating
        sentiment = pipe(entry.summary)[0]

        # Debugging: Print sentiment result for each article
        print(f"Analyzed Summary: {entry.summary}")
        print(f"Sentiment: {sentiment['label']}, Score: {sentiment['score']}")

        # Adjust the score based on sentiment
        if sentiment['label'] == 'positive':
            total_score += sentiment['score']
            pos_count += 1
        elif sentiment['label'] == 'negative':
            total_score -= sentiment['score']  # Subtract negative sentiment score
            neg_count += 1
        else:
            neu_count += 1
            total_score -= 0.01  # Slight bias towards negative for neutral articles

        num_articles += 1

    # Handle case where no articles match
    if num_articles == 0:
        return f"No relevant articles found for {ticker} with keyword '{keyword}'", None, None

    # Calculate the final sentiment score
    final_score = total_score / num_articles

    # Adjust the thresholds to classify sentiment
    if final_score >= positive_threshold:
        overall_sentiment = "Positive"
    elif final_score <= negative_threshold:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    # Generate the pie chart
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos_count, neg_count, neu_count]
    colors = ['#4CAF50', '#F44336', '#FFEB3B']
    explode = (0.1, 0, 0)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')

    # Return the final sentiment, pie chart, and articles
    return f"Overall Sentiment for {ticker}: {overall_sentiment}", fig, '\n\n'.join(articles)

# Gradio UI setup
iface = gr.Interface(
    fn=analyze_stock_sentiment,
    inputs=[
        gr.Textbox(label="Enter Stock Ticker (e.g., AAPL)", placeholder="AAPL"),
        gr.Textbox(label="Enter Keyword (optional)", placeholder="Keyword to filter articles")
    ],
    outputs=[
        gr.Textbox(label="Sentiment Result"),
        gr.Plot(label="Sentiment Pie Chart"),
        gr.Textbox(label="Articles Used", lines=10)
    ],
    title="Stock Sentiment Analyzer",
    description="Enter the stock ticker and an optional keyword to analyze news articles and determine the overall sentiment."
)

# Launch the Gradio interface
iface.launch()
