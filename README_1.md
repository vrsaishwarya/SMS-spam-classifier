# Email Spam Classifier

## Overview
This project aims to build a machine learning model that classifies emails into two categories: **Spam** and **Not Spam**. By leveraging Natural Language Processing (NLP) techniques and supervised learning algorithms, the model detects patterns in emails that are typically found in spam messages, enabling automated email filtering.

## Table of Contents
- [Project Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features
- Classifies emails as **Spam** or **Not Spam**.
- Uses NLP for text preprocessing (tokenization, stop word removal, stemming).
- Trains on a labeled dataset of emails.
- Implements a machine learning model (e.g., Naive Bayes, SVM, or logistic regression).
- Provides accuracy, precision, recall, and F1-score for performance evaluation.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/email-spam-classifier.git
    cd email-spam-classifier
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare the dataset**:
   - Download the email dataset (e.g., [SpamAssassin dataset](https://spamassassin.apache.org/old/publiccorpus/)).
   - Place the dataset in the `data/` directory.

2. **Run the training script**:
    ```bash
    python train.py
    ```

3. **Classify new emails**:
    After training, you can use the trained model to classify new emails:
    ```bash
    python classify_email.py --email "Your email content here"
    ```

## Dataset
This project uses a public email dataset that contains labeled examples of both spam and ham (non-spam) emails. The dataset is preprocessed by:
- Converting text to lowercase.
- Removing stop words.
- Tokenizing the text.
- Stemming or lemmatizing words.

**Example Datasets**:
- [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)
- [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)

## Model

### Preprocessing
The emails are preprocessed using the following steps:
1. **Lowercasing**: Converting all text to lowercase.
2. **Tokenization**: Splitting text into individual words or tokens.
3. **Stop Words Removal**: Removing commonly used words that do not carry significant meaning (e.g., "the", "is").
4. **Stemming/Lemmatization**: Reducing words to their root forms to generalize word usage.

### Model Selection
The model is built using one of the following machine learning algorithms:
- **Naive Bayes**: Often used for text classification due to its simplicity and effectiveness.
- **Support Vector Machine (SVM)**: Effective in high-dimensional spaces, suitable for email classification.
- **Logistic Regression**: A baseline linear classifier for binary classification.

The model is trained on the preprocessed dataset and optimized for accuracy and generalization.

## Results
The model's performance is evaluated using:
- **Accuracy**: The ratio of correctly predicted emails to the total emails.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positives.
- **F1-Score**: The harmonic mean of precision and recall, balancing false positives and false negatives.

Example results:
- **Accuracy**: 98.2%
- **Precision**: 96.5%
- **Recall**: 97.3%
- **F1-Score**: 96.9%

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request, or open an issue for any feature requests or bugs you find.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
