from tqdm import tqdm
import tensorflow as tf
import numpy as np
from utils import get_prediction_labels

SEQUENCE_LENGTH = 512
BATCH_SIZE = 4
SPLIT_RATIO = 0.9

def map_data(input_ids, masks, labels):
  return {'input_ids': input_ids, 'attention_mask': masks}, labels


def tokenize_text(text, tokenizer):
  tokens = tokenizer.encode_plus(text, max_length=SEQUENCE_LENGTH, truncation=True,
                                 padding='max_length', add_special_tokens=True,
                                 return_token_type_ids=False, return_tensors='tf')
  return {
      'input_ids': tf.cast(tokens['input_ids'], tf.float64),
      'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)
  }


def tf_dataset_from_df(dataset, tokenizer, textColumn='TEXT', classColumn='Class'):
    num_samples = len(dataset)
    Xids = np.zeros((num_samples, SEQUENCE_LENGTH), dtype=int)
    Xmask = np.zeros((num_samples, SEQUENCE_LENGTH), dtype=int)


    for i, TEXT in enumerate(dataset[textColumn]):
      tokens = tokenizer.encode_plus(TEXT, max_length=SEQUENCE_LENGTH, truncation=True,
                                     padding='max_length', add_special_tokens=True, return_tensors='tf')
      Xids[i,:] = tokens['input_ids']
      Xmask[i,:] = tokens['attention_mask']


    classes = dataset[classColumn].values
    classes = np.array(classes).astype(int)
    
    labels = np.zeros((num_samples, classes.max()+1))
    labels[np.arange(num_samples), classes] = 1
    
    dataset_tf = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))
    dataset_tf.take(1)
    
    dataset_tf = dataset_tf.map(map_data)
    dataset_tf.take(1)    
    
    dataset_tf = dataset_tf.shuffle(3000).batch(BATCH_SIZE, drop_remainder=True)
    dataset_tf.take(1)
    
    size = int((num_samples/BATCH_SIZE)* SPLIT_RATIO) 
    
    train_data = dataset_tf.take(size)
    val_data =  dataset_tf.skip(size)
    del dataset_tf
    
    return train_data, val_data, classes 


def infer_text_sentiment(text, inferModel, tokenizer):
    prediction = inferModel.predict(tokenize_text(text, tokenizer))
    output = np.argmax(prediction[0])
    return output, get_prediction_labels(output), prediction


def modelBertClassifier(bert, classes):
    input_ids = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), name='attention_mask', dtype='int32')

    embeddings = bert.bert(input_ids, attention_mask=mask)[1]

    x = tf.keras.layers.Dense(1024, activation='relu')(embeddings)
    y = tf.keras.layers.Dense(classes.max()+1, activation='softmax', name='outputs')(x)

    
    model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

    model.layers[2].trainable = True
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5,decay=1e-6)
    loss = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    return model
  
if __name__ == "__main__":
    print("classifier")    