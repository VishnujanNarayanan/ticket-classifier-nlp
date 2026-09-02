<h1 align="center">Customer Support Ticket Classifier</h1>

<p align="center">
  Two-target classification of free-text support tickets — issue type and urgency level —<br>
  with rule-based entity extraction and a Gradio interface for live testing.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white"/>
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.3.2-F7931E?logo=scikitlearn&logoColor=white"/>
  <img alt="NLTK" src="https://img.shields.io/badge/NLTK-3.9-154F5B"/>
  <img alt="pandas" src="https://img.shields.io/badge/pandas-2.0-150458?logo=pandas&logoColor=white"/>
  <img alt="Gradio" src="https://img.shields.io/badge/Gradio-4.44-FF7C00?logo=gradio&logoColor=white"/>
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white"/>
  <img alt="SQLite" src="https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white"/>
  <img alt="CI" src="https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white"/>
  <img alt="License" src="https://img.shields.io/badge/License-MIT-750014"/>
  <br>
  <a href="https://vishnujan-narayanan.vercel.app/"><img alt="Portfolio" src="https://img.shields.io/badge/Portfolio-vishnujan--narayanan.vercel.app-3b5998?logo=googlechrome&logoColor=white&style=for-the-badge"/></a>
  <a href="https://github.com/VishnujanNarayanan"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-VishnujanNarayanan-181717?logo=github&logoColor=white&style=for-the-badge"/></a>
  <a href="https://www.linkedin.com/in/vishnujan-narayanan"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-Vishnujan_Narayanan-0A66C2?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yMC40NDcgMjAuNDUyaC0zLjU1NHYtNS41NjljMC0xLjMyOC0uMDI3LTMuMDM3LTEuODUyLTMuMDM3LTEuODUzIDAtMi4xMzYgMS40NDUtMi4xMzYgMi45Mzl2NS42NjdIOS4zNTFWOWgzLjQxNHYxLjU2MWguMDQ2Yy40NzctLjkgMS42MzctMS44NSAzLjM3LTEuODUgMy42MDEgMCA0LjI2NyAyLjM3IDQuMjY3IDUuNDU1djYuMjg2ek01LjMzNyA3LjQzM2MtMS4xNDQgMC0yLjA2My0uOTI2LTIuMDYzLTIuMDY1IDAtMS4xMzguOTItMi4wNjMgMi4wNjMtMi4wNjMgMS4xNCAwIDIuMDY0LjkyNSAyLjA2NCAyLjA2MyAwIDEuMTM5LS45MjUgMi4wNjUtMi4wNjQgMi4wNjV6bTEuNzgyIDEzLjAxOUgzLjU1NVY5aDMuNTY0djExLjQ1MnpNMjIuMjI1IDBIMS43NzFDLjc5MiAwIDAgLjc3NCAwIDEuNzI5djIwLjU0MkMwIDIzLjIyNy43OTIgMjQgMS43NzEgMjRoMjAuNDUxQzIzLjIgMjQgMjQgMjMuMjI3IDI0IDIyLjI3MVYxLjcyOUMyNCAuNzc0IDIzLjIgMCAyMi4yMjIgMGguMDAzeiIvPjwvc3ZnPg%3D%3D&logoColor=white&style=for-the-badge"/></a>
  <a href="https://substack.com/@vishnujannarayanan"><img alt="Substack" src="https://img.shields.io/badge/Substack-@vishnujannarayanan-FF6719?logo=substack&logoColor=white&style=for-the-badge"/></a>
</p>

<p align="center">
  🎯 <a href="#why-this-project-exists">Why</a> ·
  🧩 <a href="#architecture">Architecture</a> ·
  📊 <a href="#results">Results</a> ·
  ⚡ <a href="#installation">Installation</a> ·
  🧑‍💻 <a href="#usage">Usage</a> ·
  ⚠️ <a href="#limitations">Limitations</a> ·
  🗺️ <a href="#roadmap">Roadmap</a>
</p>

---

## Why this project exists

Support desks receive tickets as unstructured prose. Routing them requires two decisions that
are normally made by hand: *what kind of problem is this* and *how fast does it need attention*.

This project trains both decisions as separate supervised models over the same feature matrix,
and pairs them with a rule-based extractor that pulls the concrete details a human agent would
scan for — the product mentioned, the date of the incident, and the complaint language used.

The result is a single `predict_ticket(text)` call that turns one string into a structured record.

## Features

- A SQLite storage layer: the spreadsheet export is loaded once into a `tickets` table, and the
  training set is selected with SQL (`sql/clean_tickets.sql`) rather than rebuilt in pandas.
- One shared feature pipeline (`pipeline.py`) used by training, inference and the notebook, so a
  served prediction is built by the same code path the models were fitted on.
- Persisted artifacts (`train.py` → `artifacts/*.joblib`) so inference never retrains.
- Text normalisation with POS-aware lemmatisation (NLTK `WordNetLemmatizer` + `pos_tag`).
- Rule-based entity extraction: products, dates, and complaint keywords.
- Handcrafted signal features: VADER sentiment, ticket length, character length, exclamation
  and question counts, and all-caps word counts.
- TF-IDF over unigrams and bigrams, capped at 3,000 features.
- Two independent classifiers: `RandomForestClassifier` for issue type, `KNeighborsClassifier`
  for urgency with a sweep over `k ∈ {3, 5, 7, 10, 15}`.
- A Gradio interface exposing issue type, urgency, and extracted entities as three outputs.

## Architecture

Both targets are predicted from one shared feature matrix. The two model heads are trained
independently — there is no shared representation learning between them.

```mermaid
flowchart TB
    Xlsx["Spreadsheet export"] --> DB[("SQLite<br/>tickets table")]
    DB --> SQL["sql/clean_tickets.sql<br/>drop unlabelled, de-duplicate bodies<br/>ROW_NUMBER() window function"]
    SQL --> Raw["Raw ticket text"]
    Raw --> Clean["Normalise<br/>lowercase, strip punctuation,<br/>stopwords, POS lemmatisation"]
    Raw --> Ents["Rule-based extraction<br/>products / dates / complaints"]
    Raw --> Meta["Signal features<br/>VADER sentiment, lengths, !/?/CAPS"]

    Clean --> Tfidf["TF-IDF<br/>1-2 grams, max 3000"]
    Ents --> Scale["MinMax scaling"]
    Meta --> Scale

    Tfidf --> X["Feature matrix<br/>1,583 columns"]
    Scale --> X

    X --> RF["RandomForest<br/>issue type"]
    X --> KNN["KNN, best k<br/>urgency level"]

    RF --> Out["Structured JSON"]
    KNN --> Out
    Ents --> Out
```

### Feature composition

On the 629-row deduplicated dataset the matrix resolves to:

| Block | Columns |
|---|---|
| TF-IDF (1–2 grams) | 1,563 |
| Entity + signal features | 20 |
| **Total** | **1,583** |

The 20 handcrafted columns are `num_products`, `num_dates`, `num_complaints`, their three
binary `has_*` counterparts, eight per-keyword complaint flags, and `ticket_length`,
`sentiment`, `exclamation_count`, `question_count`, `all_caps_count`, `char_length`.

## Results

Evaluated on a 20% held-out split (126 tickets, `random_state=42`).

### Issue type — RandomForest

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Account Access | 1.00 | 1.00 | 1.00 | 9 |
| Billing Problem | 1.00 | 1.00 | 1.00 | 18 |
| General Inquiry | 1.00 | 1.00 | 1.00 | 16 |
| Installation Issue | 1.00 | 1.00 | 1.00 | 17 |
| Late Delivery | 1.00 | 1.00 | 1.00 | 24 |
| Product Defect | 1.00 | 1.00 | 1.00 | 21 |
| Wrong Item | 1.00 | 1.00 | 1.00 | 21 |
| **Accuracy** | | | **1.00** | **126** |

A perfect score is a finding about the dataset, not the model — see [Limitations](#limitations).

### Urgency level — KNN sweep

| k | Accuracy |
|---|---|
| 3 | 0.3254 |
| 5 | 0.3492 |
| 7 | 0.3254 |
| 10 | 0.2937 |
| **15** | **0.3571** |

Best-of-sweep accuracy is **0.357**, against a majority-class baseline of 0.381 (48 Medium of
126). Urgency is not recoverable from these features; this is reported rather than hidden.

## A correctness fix worth calling out

The original notebook built its feature vector twice: column by column during training, and
again by hand inside `predict_ticket`. The two constructions disagreed in two ways.

| | Training | Old `predict_ticket` |
|---|---|---|
| Column order | `num_products … has_complaint`, complaint flags, then `ticket_length … char_length` | `ticket_length … char_length` **first**, then `num_products …` |
| `MinMaxScaler` | applied | **never applied** |

So every served prediction — the Gradio demo and the example in this README included — was
computed from a mis-assembled, unscaled vector. The models were fine; the serving path was not.

`pipeline.FEATURE_ORDER` is now the single definition of column order, and `build_matrix()`
applies the scaler on both paths. `tests/test_pipeline.py` guards it, including a regression test
that feeds the columns in reversed order and asserts the matrix is unchanged. Sanity check after
the fix: predictions agree with the stored label on 40 of 40 known tickets.

This does change one documented output — the sample ticket below now returns urgency **High**
rather than **Medium**. The new answer is what the trained model actually says.

## Project Structure

```
ticket-classifier-nlp/
├── .github/workflows/ci.yml              # Lint, tests, and a notebook parse check
├── Task1_Ticket_Classifier_Final.ipynb   # Pipeline: prep, features, training, Gradio app
├── app.py                                 # Gradio app, loads artifacts (Hugging Face entry point)
├── pipeline.py                            # Shared feature pipeline — FEATURE_ORDER lives here
├── predictor.py                           # predict_ticket() over the persisted artifacts
├── train.py                               # Trains both models, writes artifacts/
├── deploy_space.py                        # Publishes the app to a Hugging Face Space
├── tickets_db.py                          # SQLite load + SQL-backed dataset access
├── sql/
│   ├── schema.sql                         # tickets table and its indexes
│   ├── clean_tickets.sql                  # the modelling set, defined in SQL
│   └── class_distribution.sql             # class balance, GROUP BY + window function
├── tests/
│   ├── test_tickets_db.py                 # SQL cleaning rules, over a synthetic export
│   └── test_pipeline.py                   # Feature order, scaling, entity rules
├── pyproject.toml                         # ruff and pytest configuration
├── requirements.txt                       # Pinned dependencies
└── README.md
```

The modelling pipeline is still one notebook — preprocessing, feature construction, both
models, the inference function, and the Gradio launch. What moved out of it is data access:
loading and cleaning now live in `tickets_db.py` and `sql/`, where they can be tested.

### The SQL layer

`sql/clean_tickets.sql` is the single definition of "a usable ticket":

- rows without ticket text, without an issue type, or without an urgency level are dropped;
- duplicate ticket bodies collapse to their lowest `ticket_id`, chosen deterministically with
  a `ROW_NUMBER() OVER (PARTITION BY ticket_text ORDER BY ticket_id)` window function.

Both rules used to be `dropna` and `drop_duplicates` calls inside notebook cell 2. Moving them
into SQL changes no result — the same 629 rows survive — but they are now reusable and covered
by tests.

```python
from tickets_db import build_database, load_clean_tickets, class_distribution

build_database("ai_dev_assignment_tickets_complex_1000.xlsx")   # 1000 raw rows
df = load_clean_tickets()                                        # 629 usable tickets
class_distribution()                                             # counts per issue x urgency
```

## Installation

Clone the repository:

```bash
git clone https://github.com/VishnujanNarayanan/ticket-classifier-nlp.git
cd ticket-classifier-nlp
```

Create a virtual environment:

```bash
python -m venv env
source env/bin/activate      # Linux / macOS
env\Scripts\activate         # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the NLTK corpora used at runtime:

```python
import nltk
for pkg in ("punkt", "stopwords", "wordnet", "vader_lexicon", "averaged_perceptron_tagger"):
    nltk.download(pkg)
```

## Usage

### Notebook

```bash
jupyter notebook Task1_Ticket_Classifier_Final.ipynb
```

Run top to bottom. The final cell launches the Gradio app on a local URL
(`http://127.0.0.1:7862` in the recorded run).

### Inference

```python
result = predict_ticket(
    "My phone crashed on 25/05/2024 and it hasn't worked since. Very frustrated!"
)
```

```json
{
  "issue_type": "Product Defect",
  "urgency_level": "Medium",
  "entities": {
    "products": ["phone"],
    "dates": ["25/05/2024"],
    "complaint_keywords": []
  }
}
```

Note that `crashed` does not match the `crash` keyword — the extractor uses word-boundary
matching on unlemmatised text, which is a known gap.

### Gradio interface

The interface takes one textbox and returns three fields: issue type, urgency level, and the
extracted entity dictionary. It is launched from the notebook's last cell.

## Configuration

The pipeline has no configuration file. The values that control behaviour are literals in the
notebook:

| Setting | Value | Where |
|---|---|---|
| Input dataset | `ai_dev_assignment_tickets_complex_1000.xlsx` | Read via `pd.read_excel`, relative to the notebook |
| TF-IDF max features | 3000 | `TfidfVectorizer` |
| TF-IDF n-gram range | (1, 2) | `TfidfVectorizer` |
| Test split | 0.2 | `train_test_split` |
| Random seed | 42 | `train_test_split`, `RandomForestClassifier` |
| KNN sweep | 3, 5, 7, 10, 15 | Urgency training loop |
| Product vocabulary | laptop, phone, charger, headphones, battery | `product_list` |
| Complaint vocabulary | broken, late, error, issue, crash, not working, damaged, fail | `complaint_keywords` |

## Example Workflow

1. Place the ticket spreadsheet alongside the notebook, named
   `ai_dev_assignment_tickets_complex_1000.xlsx`.
2. Run the preprocessing cells — rows missing `ticket_text`, `issue_type`, or `urgency_level`
   are dropped, then duplicate ticket bodies are removed, leaving 629 rows.
3. Run the feature cells to build the 1,583-column matrix.
4. Run the training cell. The issue-type report prints immediately; the urgency loop prints one
   report per `k` and keeps the best model in `best_model`.
5. Run the inference cell to check a single ticket.
6. Run the final cell to launch Gradio and test interactively.

## Dependencies

| Package | Why |
|---|---|
| `scikit-learn` | TF-IDF vectorisation, both classifiers, metrics, scaling, splitting |
| `nltk` | Tokenisation, stopwords, POS tagging, WordNet lemmatisation, VADER sentiment |
| `pandas` / `numpy` | Dataframe handling and matrix assembly |
| `openpyxl` | Reading the ticket spreadsheet export |
| `gradio` | Browser interface for live prediction |
| `matplotlib` | Plotting during exploration |
| `pytest` | Tests for the SQL layer |

SQLite needs no dependency — it ships with Python as `sqlite3`.

## Development

```bash
jupyter notebook           # edit and run the pipeline
ruff check .               # lint (config in pyproject.toml)
pytest -q                  # tests for the SQL layer
python tickets_db.py       # rebuild the database and print the class distribution
```

### Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`, against Python 3.8
and 3.11:

| Step | What it checks |
|---|---|
| `ruff check .` | Lint across the Python modules and the notebook cells |
| `pytest -q` | 14 tests: the SQL cleaning rules, and the feature pipeline's column order and scaling |
| notebook parse | The `.ipynb` is still valid JSON after an edit |

The tests build their own synthetic spreadsheet in a temporary directory, so CI never needs
the real dataset — which is gitignored and not in the repository.

## Limitations

- **Issue-type accuracy of 1.00 is not a credible generalisation estimate.** Every class scores
  perfectly on all 126 held-out tickets, which points to templated or near-duplicate ticket
  text in the source dataset rather than a genuinely solved task.
- **Urgency prediction does not work.** The best configuration reaches 0.357 accuracy, below the
  0.381 majority-class baseline. TF-IDF plus surface signals do not carry the urgency label.
- **The dataset is not in this repository.** `.gitignore` excludes spreadsheets, so the file
  must be supplied separately; the notebook expects it beside the notebook under its original
  name.
- **Entity extraction is a fixed vocabulary.** Five products and eight complaint keywords,
  matched literally. Inflected forms (`crashed`) and unlisted products are missed.
- **Urgency prediction still does not work.** Fixing the serving path did not make the label
  learnable — it was never a serving problem. Best sweep accuracy is 0.357 against a 0.381
  majority-class baseline.
- **Entity extraction is still a fixed vocabulary.** Inflected forms (`crashed`) and unlisted
  products are missed; lemmatising before keyword matching remains open.

## Roadmap

- ~~Persist the vectoriser, scaler, and models with `joblib`.~~ Done — `train.py`.
- ~~Replace the hand-built inference feature vector with a shared path.~~ Done — `pipeline.py`.
- Investigate the issue-type leakage — inspect near-duplicate ticket bodies across the split.
- Lemmatise before complaint-keyword matching so inflected forms are caught.
- Try stronger urgency models (gradient boosting, linear SVM) and class-aware metrics before
  concluding the label is unlearnable.

## License

Released under the MIT License — free to use, modify and distribute, with attribution and
without warranty.

## Author

<p align="center">
  <strong>Vishnujan Narayanan</strong>
</p>

<p align="center">
  <a href="https://vishnujan-narayanan.vercel.app/"><img alt="Portfolio" src="https://img.shields.io/badge/Portfolio-vishnujan--narayanan.vercel.app-3b5998?logo=googlechrome&logoColor=white&style=for-the-badge"/></a>
  <a href="https://github.com/VishnujanNarayanan"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-VishnujanNarayanan-181717?logo=github&logoColor=white&style=for-the-badge"/></a>
  <a href="https://www.linkedin.com/in/vishnujan-narayanan"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-Vishnujan_Narayanan-0A66C2?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0yMC40NDcgMjAuNDUyaC0zLjU1NHYtNS41NjljMC0xLjMyOC0uMDI3LTMuMDM3LTEuODUyLTMuMDM3LTEuODUzIDAtMi4xMzYgMS40NDUtMi4xMzYgMi45Mzl2NS42NjdIOS4zNTFWOWgzLjQxNHYxLjU2MWguMDQ2Yy40NzctLjkgMS42MzctMS44NSAzLjM3LTEuODUgMy42MDEgMCA0LjI2NyAyLjM3IDQuMjY3IDUuNDU1djYuMjg2ek01LjMzNyA3LjQzM2MtMS4xNDQgMC0yLjA2My0uOTI2LTIuMDYzLTIuMDY1IDAtMS4xMzguOTItMi4wNjMgMi4wNjMtMi4wNjMgMS4xNCAwIDIuMDY0LjkyNSAyLjA2NCAyLjA2MyAwIDEuMTM5LS45MjUgMi4wNjUtMi4wNjQgMi4wNjV6bTEuNzgyIDEzLjAxOUgzLjU1NVY5aDMuNTY0djExLjQ1MnpNMjIuMjI1IDBIMS43NzFDLjc5MiAwIDAgLjc3NCAwIDEuNzI5djIwLjU0MkMwIDIzLjIyNy43OTIgMjQgMS43NzEgMjRoMjAuNDUxQzIzLjIgMjQgMjQgMjMuMjI3IDI0IDIyLjI3MVYxLjcyOUMyNCAuNzc0IDIzLjIgMCAyMi4yMjIgMGguMDAzeiIvPjwvc3ZnPg%3D%3D&logoColor=white&style=for-the-badge"/></a>
  <a href="https://substack.com/@vishnujannarayanan"><img alt="Substack" src="https://img.shields.io/badge/Substack-@vishnujannarayanan-FF6719?logo=substack&logoColor=white&style=for-the-badge"/></a>
</p>

## Deploying

```bash
huggingface-cli login                                        # write token, once
python train.py ai_dev_assignment_tickets_complex_1000.xlsx  # writes artifacts/
python deploy_space.py                                       # publishes the Space
```

The Space receives `app.py`, `pipeline.py`, `predictor.py`, `requirements-space.txt` and the
trained artifacts — never the ticket spreadsheet. It loads the persisted models and serves
immediately rather than retraining on a cold start.
