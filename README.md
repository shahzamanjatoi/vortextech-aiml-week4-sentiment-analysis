# End-to-End Sentiment Analysis of Airline Customer Tweets

Natural Language Processing capstone project — **Week 4 (Advanced)** of the Vortex Technologies
AI & Machine Learning Internship.

## Project Overview

Airlines receive far more customer feedback on social media than any support team can read
manually. This project builds a complete NLP pipeline that automatically classifies airline
customer tweets as **negative**, **neutral**, or **positive**, so that negative feedback can be
triaged quickly, sentiment trends tracked over time, and positive feedback surfaced for
marketing — going beyond the base assignment's binary positive/negative requirement by modeling
the dataset's full three-class labels.

## Features

- Full exploratory text analysis: class imbalance, tweet-length distributions, word frequency,
  and word clouds by sentiment
- Custom text-cleaning pipeline with explicit negation-word preservation
- TF-IDF feature engineering with tuned `ngram_range`, `min_df`, and `max_df`
- Three trained and compared models: Logistic Regression, Multinomial Naive Bayes, Linear SVM
- Hyperparameter tuning via `GridSearchCV` and 5-fold stratified cross-validation
- Model interpretability: most influential words per class, prediction-probability visualization
- A deliberately adversarial 10-sentence stress test (including sarcasm and mixed opinion)
- Honest, evidence-based limitations discussion
- Persisted, ready-to-deploy inference pipeline (`models/sentiment_pipeline.joblib`) with a
  clean `predict_sentiment()` function

## Dataset

**Twitter US Airline Sentiment** — 14,640 tweets about major U.S. airlines (February 2015),
manually labeled `negative` / `neutral` / `positive`. Included locally at `data/tweets.csv` for
full reproducibility.

## NLP Pipeline

```
Raw tweet text
   → lowercase, strip HTML/URLs/emojis/@mentions/numbers/punctuation
   → tokenize → stopword removal (negations preserved) → lemmatize
   → TF-IDF vectorization (max_features=5000, ngram_range=(1,2), min_df=3, max_df=0.9)
   → classifier (Logistic Regression / Naive Bayes / Linear SVM)
```

## Machine Learning Models

| Model | Role |
|---|---|
| Logistic Regression | Primary production candidate — interpretable, calibrated probabilities |
| Multinomial Naive Bayes | Fast classical baseline |
| Linear SVM | High-dimensional sparse-text specialist |

Evaluated with accuracy, macro-precision, macro-recall, and macro-F1 (macro-averaging chosen
because the dataset is imbalanced, ~63% negative).

## Results

Exact metric values are computed live in Section 8 of the notebook and can shift slightly across
environments/library versions — see the notebook's model comparison table and confusion matrices
for current numbers. Logistic Regression (tuned via `GridSearchCV`) is carried forward as the
production candidate; the full justification is in Section 8 of the notebook.

## Screenshots

*(Add exported PNGs of the word cloud, confusion matrix, and model-comparison chart here for a
GitHub-visible preview — e.g. `screenshots/wordclouds.png`, `screenshots/confusion_matrix.png`.)*

## Installation

```bash
git clone <this-repo-url>
cd vortextech-aiml-week4-sentiment-analysis
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

## Usage

```bash
jupyter notebook Vortex_Week4_Sentiment_Analysis.ipynb
```

Run all cells top to bottom (`Kernel > Restart & Run All`). No external downloads or API keys
required — the dataset ships with the repository.

To use the trained model directly in another script:

```python
import joblib
pipeline = joblib.load("models/sentiment_pipeline.joblib")
pipeline.predict(["My flight was delayed for 4 hours with no updates."])
```

## Folder Structure

```
vortextech-aiml-week4-sentiment-analysis/
├── data/
│   └── tweets.csv
├── models/
│   └── sentiment_pipeline.joblib
├── Vortex_Week4_Sentiment_Analysis.ipynb
├── README.md
├── requirements.txt
└── LICENSE
```

## Future Work

- Fine-tune a lightweight transformer (e.g. DistilBERT) on the same dataset and benchmark it
  against this TF-IDF baseline, specifically on the sarcasm/mixed-opinion cases identified in
  Section 11 of the notebook.
- Add an LSTM/GRU baseline to quantify how much sequence-order information TF-IDF discards.
- Wrap `predict_sentiment()` in a lightweight API (e.g. FastAPI) for real-time scoring.

## Technologies Used

- **Language:** Python 3
- **Core libraries:** pandas, numpy, matplotlib, seaborn, wordcloud
- **NLP:** nltk (tokenization, stopwords, lemmatization)
- **ML:** scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, Linear SVM, GridSearchCV,
  cross-validation), joblib (model persistence)

## Author

**Shahzaman Jatoi**
Mechatronics Engineering, Mehran University of Engineering and Technology (MUET)
[LinkedIn](https://linkedin.com/in/shahzamanjatoi) · [GitHub](https://github.com/shahzamanjatoi)

---
*Built as part of the Vortex Technologies AI & Machine Learning Internship — Week 4 of 4
(Capstone).*
