import tensorflow as tf
from transformers import BertTokenizer
import gradio as gr
from classifier import infer_text_sentiment


MODEL_EXPORT_DIR = "model/bertClassifier"
BERT_MODEL_NAME = 'bert-base-multilingual-cased'
LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]


def get_sentiment(input_text):
  res = infer_text_sentiment(input_text, inferModel, tokenizer)
  prediction = res[2][0]

  return {LABELS[i]: float(prediction[i]) for i in range(3)}


if __name__ == "__main__":
    print("inference")
    
    inferModel = tf.keras.models.load_model(MODEL_EXPORT_DIR)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    label = gr.outputs.Label(num_top_classes=3)
    iface = gr.Interface(fn = get_sentiment, 
                        inputs = "text", 
                        outputs = label,
                        title = 'CoVaxBDSentiment Analysis', 
                        description="Infer Sentiment for input text")
                        
    iface.launch(inline = False, share=False)    