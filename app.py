import joblib
import gradio as gr
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
voting = joblib.load(os.path.join(MODEL_DIR, "voting_classifier_model.pkl"))


def detect_news(news):
    vec = vectorizer.transform([news])
    pred = voting.predict(vec)[0]

    if pred == 1:
        return "Fake News"
    else:
        return "Real News"


app = gr.Interface(
    fn=detect_news,
    inputs=gr.Textbox(lines=6, placeholder="Enter news text here..."),
    outputs="text",
    title="Fake News Detection using Ensemble Learning",
    description="Detect whether a news article is Fake or Real using pre-trained models."
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
