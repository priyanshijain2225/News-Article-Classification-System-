# News Article Classification & Headline Generation System
<img width="1908" height="811" alt="image" src="https://github.com/user-attachments/assets/9f841d61-0f75-46e4-b52d-94571245a58c" />

## 🔗Live Link: [Streamlit Link](https://news-article-classification-system.streamlit.app/)
A complete end-to-end NLP system designed to classify news articles into target categories and automatically generate concise headlines from article descriptions.

The project uses the BBC News dataset and implements:
1. **News Classification**: A Logistic Regression classifier using TF-IDF features to categorize articles into four classes (**Business**, **Politics**, **Sports**, and **Technology**).
2. **Confidence Estimation**: Probability estimates from Logistic Regression mapped to confidence levels (High, Medium, Low).
3. **Headline Generation**: A fine-tuned Hugging Face `T5-small` model utilizing sequence-to-sequence beam search to generate headlines from description inputs.
4. **Interactive Dashboard**: A premium glassmorphic Streamlit web application with probability progress bars and interactive Plotly charts.
5. **Bidirectional LSTM Baseline**: The notebook also includes a Bidirectional LSTM network baseline using Word2Vec embeddings for evaluation.

---

## Project Structure

```
├── news_classification_himakesh.ipynb  # Core Jupyter Notebook (EDA, training, saving, pipeline)
├── app.py                              # Streamlit Web Application
├── requirements.txt                    # Project Dependencies
├── README.md                           # Project Documentation
├── bbc_news.csv                        # Raw dataset (BBC News RSS feeds)
├── models/                             # Saved classification assets (created automatically)
│   ├── classification_model.pkl        # Saved Logistic Regression Classifier
│   ├── tfidf_vectorizer.pkl            # Pickled TF-IDF Vectorizer
│   ├── label_encoder.pkl               # Pickled Sklearn LabelEncoder
│   └── t5_small/                       # Fine-tuned T5-small model and tokenizer weights
```

---

## Installation & Setup

Ensure you are using Python 3.10+ (tested on Python 3.13) and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1: Train Models & Save Assets (Jupyter Notebook)

The Streamlit web application requires saved model files on disk. To train the classification models and download/save the T5 model weights:

1. Open a terminal and start Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter lab
   ```
2. Open [news_classification_himakesh.ipynb](file:///news_classification_himakesh.ipynb) and click **Run All Cells**.
3. The notebook will automatically train the ML models (Logistic Regression, Naive Bayes, Random Forest, XGBoost), train the Keras LSTM classifier, output performance reports, and serialize all assets to `models/`.

### Step 2: Launch the Streamlit Web App

Once the assets are saved, launch the local interactive web interface:

```bash
streamlit run app.py
```

The app will launch in your default web browser (usually at `http://localhost:8501`).

---

## Technical Details

### Classification Pipeline
* **Text Processing**: Standardizes text via lowercasing, punctuation/number removal, NLTK stopword filtering, and NLTK WordNet lemmatization.
* **Classifier**: Logistic Regression trained on TF-IDF features (unigrams, bigrams, trigrams).
* **Confidence Scores**: Map category probabilities directly from Logistic Regression's probability estimates (`predict_proba`):
  - **High**: $\ge 80\%$
  - **Medium**: $\ge 50\%$ and $< 80\%$
  - **Low**: $< 50\%$

### Headline Generation
* Uses Google's `t5-small` sequence-to-sequence model.
* Generates concise, non-repeating titles using **beam search** with early stopping:
  ```python
  num_beams=4, max_length=30, min_length=5, no_repeat_ngram_size=2, early_stopping=True
  ```

---

## Future Scope

1. **Transformer-based Classifiers**: Upgrade the classification head to BERT, RoBERTa, or DistilBERT for contextualized embeddings.
2. **Generative Fine-Tuning**: Fine-tune `t5-base` or `t5-small` on a larger aligned dataset of BBC title-description pairs.
3. **Multi-Label Classifications**: Support news articles belonging to multiple categories simultaneously.
4. **RSS Feeds Integration**: Add direct API endpoints to ingest real-time world news.
