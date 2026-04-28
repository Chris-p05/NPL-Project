# OptiHire — AI-Powered Resume & Job Matching System

OptiHire is an NLP-based resume screening and ranking system that automatically matches candidate resumes against a job description using semantic similarity and keyword alignment. It is designed to reduce manual recruiter workload and minimize unconscious bias by anonymizing personally identifiable information before any scoring takes place. The system processes resumes in batch, scores each one using a hybrid model combining Sentence-BERT embeddings and spaCy keyword extraction, and outputs a ranked leaderboard of the most qualified candidates.

---

## How It Works

The pipeline runs each resume through six stages:

1. Raw resume text is loaded from the Kaggle Resume Dataset CSV
2. PII redaction removes names, locations, and dates using spaCy NER
3. Text normalization lowercases, lemmatizes, and strips stopwords
4. SBERT encodes the cleaned text into a 384-dimensional embedding vector
5. Keyword extraction pulls nouns and proper nouns using spaCy POS tagging
6. A weighted final score is computed as 70% semantic similarity + 30% keyword overlap

---

## Requirements

### Python Version
Python 3.8 or higher is required.

### Hardware
A GPU is strongly recommended for SBERT encoding. The notebook is configured to run on Google Colab with a T4 GPU. CPU-only execution is supported but will be significantly slower on large batches.

### Python Packages

Install all dependencies by running:

```bash
pip install torch tensorflow transformers nltk spacy pymupdf scrapingbee kaggle sentence-transformers pandas
```

Then download the spaCy language model:

```bash
python -m spacy download en_core_web_md
```

### API Keys Required

The following credentials must be configured as secrets in your Google Colab environment under the Secrets tab:

| Secret Name | Description |
|---|---|
| `KAGGLE_USERNAME` | Your Kaggle account username |
| `KAGGLE_KEY` | Your Kaggle API key, found at kaggle.com/account |
| `SCRAPINGBEE_KEY` | Your ScrapingBee API key for live job scraping (optional) |

To get your Kaggle credentials, go to kaggle.com, click on your profile, select Account, and scroll down to the API section to generate a new token.

---

## Steps to Run

### Step 1 — Open in Google Colab

Upload `OptiHireDataScraper.ipynb` to Google Colab. Before running any cells, go to Runtime > Change runtime type and select GPU as the hardware accelerator.

### Step 2 — Add Your Secrets

In the left sidebar, click the key icon to open the Secrets panel. Add your `KAGGLE_USERNAME`, `KAGGLE_KEY`, and optionally your `SCRAPINGBEE_KEY` as individual secrets.

### Step 3 — Run Cell 1 (Install Libraries)

Run the first cell to install all required packages. This only needs to be done once per Colab session.

### Step 4 — Run Cell 2 (Load Credentials)

This cell reads your Kaggle credentials from the Secrets panel and sets them as environment variables so the Kaggle CLI can authenticate.

### Step 5 — Run Cell 3 (Download Dataset)

This cell downloads and unzips the Kaggle Resume Dataset. It will appear as `Resume/Resume.csv` in your Colab file system.

### Step 6 — Run Cell 4 (Load NLP Tools)

This cell loads the spaCy language model and defines the `clean_and_anonymize()` function used for PII redaction and text normalization.

### Step 7 — Run Cell 5 (Batch Processing)

This cell reads the dataset and processes the first 50 resumes through the anonymization and normalization pipeline. The results are stored in a list called `processed_data`. You can increase the batch size by changing `.head(50)` to a larger number.

### Step 8 — Run Cell 6 (Optional: Live Job Scraping)

If you have a ScrapingBee API key, this cell scrapes a live job posting from SimplyHired and extracts the job description. If you do not have a key or prefer to skip this step, a fallback sample job description is used automatically. You can also upload your own resume PDF as `my_resume.pdf` to the Colab file browser for individual evaluation.

### Step 9 — Run Cell 7 (Semantic Matching)

This cell loads the SBERT model (`all-MiniLM-L6-v2`) and defines the `get_match_score()` function. It runs a sample test against the first 5 processed resumes and prints their match scores.

### Step 10 — Run Cell 8 (Refined Scoring)

This cell adds keyword extraction on top of the base SBERT score. It defines `refined_match_score()` and demonstrates the difference between the baseline semantic score and the improved hybrid score on a sample resume.

### Step 11 — Run Cell 9 (Generate Leaderboard)

This is the final output cell. It ranks all processed resumes against the job description and displays the top 10 candidates in a formatted leaderboard table, sorted by final score descending.

---

## Optional: Evaluating Your Own Resume

To evaluate a specific resume PDF against a job description:

1. Upload your PDF file to the Colab file browser and rename it `my_resume.pdf`
2. Run Cell 6, which will detect the file and encode it as base64 for evaluation
3. Pass the encoded resume into `refined_match_score()` alongside any job description text

---

## Project Structure

```
OptiHireDataScraper.ipynb   Main notebook containing the full pipeline
README.md                   This file
my_resume.pdf               Optional personal resume for individual evaluation
Resume/
    Resume.csv              Downloaded Kaggle dataset (generated at runtime)
```

---

## Notes and Limitations

The system currently has no ground-truth relevance labels, meaning evaluation is heuristic rather than formally measured with precision or recall. The 70/30 weighting between semantic and keyword scores was chosen based on observed ranking quality and can be adjusted in the `refined_match_score()` function. The spaCy NER model may occasionally miss informal name formats or unconventional resume structures, and the keyword component does not account for domain synonyms such as Keras and TensorFlow being related technologies.
