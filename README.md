# 📩 Customer Support Ticket Classifier

This project classifies customer support tickets by **issue type** and **urgency level**, and also extracts key entities like product names, complaint keywords, and dates.

Built with:
- 🧠 Machine Learning (Random Forest, KNN)
- 💬 NLP (TF-IDF, Lemmatization, Entity Extraction)
- 🎨 Gradio Interface for live testing

---

## 🧰 Features

- **Text Preprocessing**: Lowercasing, punctuation removal, tokenization, POS tagging, and lemmatization using NLTK.
- **Entity Extraction**:
  - Products (`laptop`, `phone`, `charger`, etc.)
  - Complaint Keywords (`broken`, `error`, `not working`, etc.)
  - Dates (with flexible date format matching)
- **Sentiment & Meta Features**:
  - Sentiment score (VADER)
  - Ticket and character length
  - Special character counts (`!`, `?`, UPPERCASE words)
- **TF-IDF Text Features**: Up to 3000 word and bi-gram features.
- **Model Training**:
  - `RandomForestClassifier` for issue type
  - Optimized `KNeighborsClassifier` for urgency level
- **Prediction Function**: Combines all features for real-time prediction.
- **Gradio Interface**: Easy-to-use web app for live testing.

---

## 🛠 Installation

```bash
pip install -r requirements.txt

If Gradio is not installed, run:

```bash
pip install gradio
```

You also need to download required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

---

## 📊 Dataset

The model expects a dataset with the following columns:

- `ticket_text`: Text of the support ticket  
- `issue_type`: Label for classification  
- `urgency_level`: Label for classification

Load it from Excel like this:

```python
import pandas as pd
df = pd.read_excel("ai_dev_assignment_tickets_complex_1000.xls")
```

---

## 🚀 Running the Model

Run the script using:

```bash
jupyter notebook Task1_Ticket_Classifier_Final.ipynb
```

---

## 🧪 Sample Prediction

```python
sample_text = "My phone crashed on 25/05/2024 and it hasn't worked since. Very frustrated!"
result = predict_ticket(sample_text)
print(result)
```

### Example Output

```json
{
  "issue_type": "Software Issue",
  "urgency_level": "High",
  "entities": {
    "products": ["phone"],
    "dates": ["25/05/2024"],
    "complaint_keywords": ["crash", "not working"]
  }
}
```

---

## 🌐 Gradio Web App

The Gradio interface lets you test the model live in your browser.

After running the script, visit the local URL shown in the terminal:

---

## 📁 Project Structure

```
.
├── ticket_classifier.py
├── ai_dev_assignment_tickets_complex_1000.xls
├── README.md
└── requirements.txt
```

---

## 📜 License

MIT License. Use and modify freely.

---

## 🙌 Credits

Developed by **Vishnujan Narayanan** for the AI Developer Internship Assignment.
