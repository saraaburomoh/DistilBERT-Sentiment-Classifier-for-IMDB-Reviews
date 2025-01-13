# Sentiment Analysis on Movie Reviews

This project focuses on performing sentiment analysis on movie reviews using the IMDB dataset. A fine-tuned **DistilBERT** model is used to classify reviews into positive or negative sentiments.

---

## Overview

Sentiment analysis is a natural language processing (NLP) task that involves determining the sentiment of a piece of text. In this project:
- We use the **IMDB dataset** from Hugging Face, containing 50,000 movie reviews.
- The **DistilBERT** model is fine-tuned for binary sentiment classification (positive or negative).

---

## Features

- Fine-tunes the **DistilBERT** model for sentiment analysis.
- Evaluates the model using accuracy, precision, recall, and F1-score.
- Saves the fine-tuned model for future use.

---

## Dataset

The dataset used is the **IMDB dataset**:
- **Training Data**: 20,000 reviews (80% of the training set).
- **Validation Data**: 5,000 reviews (20% of the training set).
- **Test Data**: 25,000 reviews.

Dataset Link: [IMDB Dataset on Hugging Face](https://huggingface.co/datasets/imdb)

---

## Technologies Used

- **Frameworks**:
  - PyTorch
  - Hugging Face Transformers
- **Libraries**:
  - scikit-learn
  - Hugging Face Datasets
- **Platform**: Google Colab or local machine with GPU acceleration.

---

## Steps to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-movie-reviews.git
   cd sentiment-analysis-movie-reviews
