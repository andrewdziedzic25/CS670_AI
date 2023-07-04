import gradio as gr 
from transformers import pipeline
import time

sentiment_classifier = pipeline("text-classification", return_all_scores=True)

def classifier(text):
    pred = sentiment_classifier(text)
    return {p["label"]: p["score"] for p in pred[0]}

def sleep_for_test():
    time.sleep(10)
    return 2

with gr.Blocks(theme="gstaff/xkcd") as demo:
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Text")
            with gr.Row():
                classify = gr.Button("Classify Sentiment")
        with gr.Column():
            label = gr.Label(label="Predicted Sentiment")
        number = gr.Number()
        btn = gr.Button("Sleep then print")
    classify.click(classifier, input_text, label, api_name="classify")
    btn.click(sleep_for_test, None, number, api_name="sleep")
demo.launch(enable_queue=False)
