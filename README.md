# Video Game Feature Analysis and Extraction

This repository contains the research paper and the Python code for analyzing the influence of various video game features (such as graphics, story, gameplay mechanics, sound, and multiplayer) on player satisfaction. The analysis uses Amazon video game reviews, applying natural language processing (NLP) techniques, sentiment analysis, and machine learning models to identify patterns and correlations between game features and review ratings.

## Project Overview

### Research Paper
The research paper, titled **"The Role of Specific Features in Enhancing Player Satisfaction in Video Games"**, investigates how different game features impact player satisfaction, as reflected in customer reviews on Amazon. It explores the following:

- **Hypotheses**:
  - H1: Specific game features (such as story depth, graphical quality, or ease of controls) are positively correlated with higher ratings.
  - H0: Game features do not have a significant impact on the ratings.
- **Methods**:
  - The study uses **Latent Semantic Analysis (LSA)** and a set of **informative topics** based on game features.
  - The dataset contains Amazon reviews, and the features (such as graphics, gameplay, story) are extracted using NLP techniques like **lemmatization** and **tokenization**.
  - A **sentiment analysis** is performed on the extracted features, and their correlation with overall ratings is evaluated using **Ordinary Least Squares (OLS)** regression and **machine learning models** such as Random Forest and Neural Networks.

### Code
The Python code in this repository processes the Amazon reviews and extracts valuable insights on game features. The key processes include:

- **Data Cleaning**:
  - The dataset is cleaned to remove duplicate reviews, missing data, and short reviews. Only reviews with more than 1000 characters are retained for analysis.
- **Feature Extraction**:
  - Game features (e.g., graphics, sound, story) are extracted from the text reviews using **CountVectorizer** and **Latent Semantic Analysis (LSA)** to identify key topics.
  - Informative topics based on predefined sets of keywords (e.g., words related to graphics or sound) are also extracted.
- **Sentiment Analysis**:
  - Sentiment analysis is performed on the review text to determine the polarity (positive, negative, neutral) of the comments related to each game feature.
- **Machine Learning Models**:
  - **Random Forest**, **Neural Networks**, and **Support Vector Classifier (SVC)** models are used to predict player satisfaction based on the presence and sentiment of the game features in the reviews.
  - Model performance is evaluated using metrics such as **ROC curves**, **confusion matrices**, and **precision-recall curves**.

## Folder Structure

- `Final Paper (5).pdf`: The complete research paper detailing the analysis and findings.
- `Code (1).pdf`: The Python code that processes the reviews, performs sentiment analysis, and applies machine learning models.
- `requirements.txt`: A list of Python libraries required to run the code (to be added).
- `README.md`: This file, which explains the project structure and how to use the code.

## Requirements

To run the Python code, ensure you have the following libraries installed:

- `pandas`
- `nltk`
- `spacy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels`
- `textblob`
- `xgboost`

You can install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1.	Clone the Repository:
```bash
git clone https://github.com/YOUR_USERNAME/Video-Game-Features-Analysis-and-Extraction.git
cd Video-Game-Features-Analysis-and-Extraction
```

# Results

The study found that features such as Graphics and Sound positively correlate with higher player ratings, while the Story feature showed a negative correlation with ratings in certain cases. The machine learning models, particularly Random Forest and Neural Networks, predicted negative reviews more effectively due to class imbalance in the dataset.

## Future Work

	•	Improving the feature extraction process by incorporating more advanced NLP models.
	•	Balancing the dataset to improve the prediction of positive reviews.
	•	Expanding the analysis to include in-game metrics and user behavior data for a more holistic view of player satisfaction.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements.
