
import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer
from transformers import TFAutoModel
from classifier import tokenize_text, tf_dataset_from_df, modelBertClassifier
from utils import plotTraining, remove_emoji

DATASET_DIR = "dataset/CoVaxBD.xlsx"
BERT_MODEL_NAME = 'bert-base-multilingual-cased'
MODEL_EXPORT_DIR = "model/bertClassifier"
EXPERIMENT_DIR = "experiment/bertClassifier"

def cleanDataset (df):
    df['Class']= df.apply(lambda row: sentiment_to_id(row), axis=1)
    df.drop(df[df['Class'] == -1].index, inplace = True)
    print("Cleaning...")
    for index in tqdm(df.index):
        df.loc[index,'TEXT'] = remove_emoji(df.loc[index,'TEXT']) # Apply different cleaning methods 
    df.drop(df[df['TEXT'] == ""].index, inplace = True)
    return(df)
    

if __name__ == "__main__":
    
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert = TFAutoModel.from_pretrained(BERT_MODEL_NAME)
    
    dataset= pd.read_excel(DATASET_DIR)
    
    print(DATASET_DIR, dataset['Sentiment'].value_counts())
    
    # dataset = cleanDataset(dataset)
    
    train_data, val_data, classes =  tf_dataset_from_df(dataset, tokenizer)    
    
    bertClassifier = modelBertClassifier(bert, classes)
    history = bertClassifier.fit( train_data, validation_data=val_data, epochs=5)
    
    bertClassifier.save(MODEL_EXPORT_DIR)
    
    plotTraining(history, EXPERIMENT_DIR)

    
