import pandas as pd
import re
import matplotlib.pyplot as plt

def plotTraining(history, outputDir):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']

    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig(outputDir, dpi=300)
    
def sentiment_to_id(row):
   if row['Sentiment'] == "NEGATIVE" :
      return int(0)
   if row['Sentiment'] == "NEUTRAL" :
      return int(1)
   if row['Sentiment'] == "POSITIVE" :
      return int(2)    
   return -1

def id_to_sentiment(row):
   if row['Class'] == 1 :
      return "NEUTRAL"
   if row['Class'] == 2 :
      return "POSITIVE"
   if row['Class'] == 0 :
      return "NEGATIVE"   
   return "UNKNOWN"


def get_prediction_labels(id):
   if id == 0 :
      return "NEGATIVE"
   if id == 1:
      return "NEUTRAL"
   if id ==2:
     return "POSITIVE"   
   return "NONE"


def remove_emoji(string):
    emoji_pattern = re.compile("["
      u"\U0001F600-\U0001F64F"  # emoticons
      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
      u"\U0001F680-\U0001F6FF"  # transport & map symbols
      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
      u"\U00002500-\U00002BEF"  # chinese char
      u"\U00002702-\U000027B0"
      u"\U00002702-\U000027B0"
      u"\U000024C2-\U0001F251"
      u"\U0001f926-\U0001f937"
      u"\U00010000-\U0010ffff"
      u"\u2640-\u2642"
      u"\u2600-\u2B55"
      u"\u200d"
      u"\u23cf"
      u"\u23e9"
      u"\u231a"
      u"\ufe0f"  # dingbats
      u"\u3030"
      "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def clean_en(value:str)->str:
    ''' Get clean English-only Text '''
    clean_value = re.sub("[^A-Za-z0-9\/\-' ]+", '', value)
    return clean_value


def remove_links(text):
  text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))', '', text, flags=re.MULTILINE)
  return text


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample) 


if __name__ == "__main__":
    print("utils")    

