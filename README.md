# Fake News Detection using Ensemble Learning

A Dockerized NLP mini project that detects whether a news article is **Fake** or **Real** using ensemble machine learning models.

## Models

This project uses the following models with a TF-IDF vectorizer (max 5000 features):

| Model | Accuracy |
|-------|----------|
| Naive Bayes | 92.87% |
| Logistic Regression | 98.64% |
| Voting Classifier (NB + LR) | 96.82% |
| Random Forest | 99.76% |
| AdaBoost | 99.39% |

The **Voting Classifier** (soft voting between Naive Bayes and Logistic Regression) is used for predictions in the Gradio demo.

## Quick Start

### Prerequisites

- Docker & Docker Compose

### Run with Docker Compose

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Place your trained model files in the `models/` directory:
```
models/
├── vectorizer.pkl
├── voting_classifier_model.pkl
├── naive_bayes_model.pkl
├── logistic_regression_model.pkl
├── random_forest_model.pkl
└── adaboost_model.pkl
```

3. Start the application:
```bash
docker compose up --build
```

4. Open your browser and navigate to:
```
http://localhost:7860
```

## Run Locally (without Docker)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to `http://localhost:7860`

## Project Structure

```
.
├── app.py                  # Gradio application
├── models/                 # Trained model files
│   ├── vectorizer.pkl
│   ├── voting_classifier_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── adaboost_model.pkl
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── NLP_8_mini_Project.ipynb # Original Colab notebook
└── README.md
```

## Dataset

The model was trained on a dataset containing:
- **Fake.csv**: Fake news articles
- **True.csv**: Real news articles

Labels: `0` = Real News, `1` = Fake News

## Tech Stack

- **Python 3.11**
- **scikit-learn** - Machine learning
- **Gradio** - Web UI
- **Docker** - Containerization

## License

MIT
