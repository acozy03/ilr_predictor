import os
import json
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import tensorflow as tf
from architecture import build_model
import keras
import tf_keras as keras
from transformers.models.xlm_roberta import TFXLMRobertaModel
from sklearn.utils.class_weight import compute_class_weight
from metrics import MacroPrecision, MacroRecall, MacroF1Score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow_text as tf_text
import spacy
from transformers import AutoTokenizer
import re
import random
import numpy as np
import gc



SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


tf.keras.backend.clear_session()
gc.collect()

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_max_chunks(passages, tokenizer, max_tokens=256):
    chunk_lengths = []
    for text in passages:
        chunks = chunk_passage(text, tokenizer, max_tokens)
        chunk_lengths.append(len(chunks))
    max_chunks = max(chunk_lengths)
    p95_chunks = int(np.percentile(chunk_lengths, 95))
    print(f"Max: {max_chunks}, 95th percentile: {p95_chunks}")
    return p95_chunks

BATCH_SIZE = 3
LABEL_MAP = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4
}

MAX_TOKENS = 256
LABEL_MAP_INVERSE = {v: k for k, v in LABEL_MAP.items()}

        
# def validate_labels(df):
#     valid_labels = [0, 1, 2, 3, 4]  
#     invalid = df[~df['label'].isin(valid_labels)]
#     if not invalid.empty:
#         print(f"Invalid labels found:\n{invalid}")
#     return df[df['label'].isin(valid_labels)]

def load_and_prepare_data(train_path, dev_path):
    def load_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]
        df = pd.DataFrame(lines)
        df['text'] = df['text'].astype(str).str.strip()
        df.dropna(subset=['text', 'label'], inplace=True)  
        
        return df

    train_df = load_jsonl(train_path)
    dev_df = load_jsonl(dev_path)

    return (
        train_df['text'].tolist(),
        dev_df['text'].tolist(),
        train_df['label'].values,
        dev_df['label'].values
    )

def chunk_sentences(sentences, tokenizer, MAX_TOKENS):
    chunks = []
    current_chunk = []
    current_length = 0

    for sent in sentences:
        tokenized = tokenizer.tokenize(sent)
        token_len = len(tokenized)

        if current_length + token_len > MAX_TOKENS:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_length = token_len
        else:
            current_chunk.append(sent)
            current_length += token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def split_sentences(text):
    import re
    sentence_endings = re.compile(r'(?<=[.!؟!?。])\s+')
    return [s.strip() for s in sentence_endings.split(text.strip()) if s.strip()]

def chunk_passage(text, tokenizer, MAX_TOKENS):
    sentences = split_sentences(text)
    chunks, current_chunk, current_length = [], [], 0
    for sent in sentences:
        token_len = len(tokenizer.tokenize(sent))
        if current_length + token_len > MAX_TOKENS:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_length = token_len
        else:
            current_chunk.append(sent)
            current_length += token_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def chunk_and_tokenize_passage(passage, tokenizer):
    sentences = split_sentences(passage)  
    chunks = chunk_sentences(sentences, tokenizer, MAX_TOKENS)

    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]

    while len(chunks) < MAX_CHUNKS:
        chunks.append("")

    encodings = tokenizer(
        chunks,
        max_length=MAX_TOKENS,
        padding='max_length',
        truncation=True,
        return_tensors='np',
        return_token_type_ids=False
    )

    return encodings['input_ids'], encodings['attention_mask']

def predict_with_chunking(texts, model, tokenizer):
    mean_predictions = []
    max_predictions = []

    for passage in texts:
        input_ids, attention_mask = chunk_and_tokenize_passage(passage, tokenizer)

        input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
        attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int32)

        if tf.size(input_ids) == 0 or tf.size(attention_mask) == 0:
            print(f"[Warning] Empty logits for input. Defaulting to class 0.\nText: {passage[:120]}...")
            mean_predictions.append(0)
            max_predictions.append(0)
            continue

        input_ids = tf.expand_dims(input_ids, axis=0)
        attention_mask = tf.expand_dims(attention_mask, axis=0)

        logits = model.predict({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }, verbose=0)

        if logits is None or tf.size(logits) == 0:
            print(f"[Warning] Model returned empty logits. Defaulting to class 0.\nText: {passage[:120]}...")
            mean_predictions.append(0)
            max_predictions.append(0)
            continue

        logits = tf.squeeze(logits, axis=0)

        if logits.shape.rank == 1:
            logits = tf.expand_dims(logits, axis=0)

        if logits.shape.rank != 2:
            print(f"[Warning] Unexpected logits shape {logits.shape}. Defaulting to class 0.\nText: {passage[:120]}...")
            mean_predictions.append(0)
            max_predictions.append(0)
            continue


        avg_pred = tf.argmax(tf.reduce_mean(logits, axis=0)).numpy()
        max_pred = tf.argmax(tf.reduce_max(logits, axis=0)).numpy()

        mean_predictions.append(avg_pred)
        max_predictions.append(max_pred)

    return mean_predictions, max_predictions



def build_chunked_dataset(passages, labels, tokenizer):
    input_ids_list = []
    attention_masks_list = []

    for text in passages:
        input_ids, attention_mask = chunk_and_tokenize_passage(text, tokenizer)
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)

    input_ids_tensor = tf.convert_to_tensor(input_ids_list, dtype=tf.int32)  # (B, M, N)
    attention_mask_tensor = tf.convert_to_tensor(attention_masks_list, dtype=tf.int32)
    label_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor
        },
        label_tensor
    )).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    return dataset


def initialize_tokenizer():
    return AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_data(tokenizer, texts):
    cleaned_texts = [str(text).strip() for text in texts]
    
    return tokenizer(
        cleaned_texts,
        max_length=MAX_TOKENS,
        padding='max_length',
        truncation=True,  
        return_tensors='tf',
        return_token_type_ids=False
    )
    
def create_tf_dataset(input_ids, attention_mask, labels):
    return tf.data.Dataset.from_tensor_slices((
        {'input_ids': input_ids, 'attention_mask': attention_mask},
        labels
    )).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def predict_file(input_file, output_file='predictions.csv', model_path='./final_model'):
    tf.keras.backend.clear_session()
     
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'TFXLMRobertaModel': TFXLMRobertaModel}
    )
    
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
    
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                data.append({
                    'english_text': parts[0],
                    'tamil_text': parts[1],
                    'true_label': parts[2]
                })
    
    df = pd.DataFrame(data)
    
    encodings = tokenizer(
        df['english_text'].tolist(),
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='tf',
        return_token_type_ids=False  
    )
    
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
    )).batch(16)
    
    predictions = model.predict(dataset)
    predicted_indices = tf.argmax(predictions, axis=1).numpy()  
    
    inverse_map = {v: k for k, v in LABEL_MAP.items()}
    df['predicted_label'] = [inverse_map[idx] for idx in predicted_indices]
    
    df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")
    
    return df

def predict_test_set(dev_jsonl_path='data/dev_english.jsonl', model_path='./final_model'):
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        'TFXLMRobertaModel': TFXLMRobertaModel,
        'MacroPrecision': MacroPrecision,
        'MacroRecall': MacroRecall,
        'MacroF1Score': MacroF1Score
    }
)

    
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
    

    with open(dev_jsonl_path, 'r', encoding='utf-8') as f:
        dev_data = [json.loads(line) for line in f]


    df = pd.DataFrame(dev_data)

    if 'text' in df.columns:
        df['text'] = df['text'].astype(str).str.strip()
    elif 'english_text' in df.columns:
        df['text'] = df['english_text'].astype(str).str.strip()
    else:
        raise ValueError("Missing expected text column: 'text' or 'english_text'")

    if 'label' in df.columns:
        test_labels = df['label'].values
    elif 'ilr_level' in df.columns:
        df['label'] = df['ilr_level'].map(LABEL_MAP)
        test_labels = df['label'].values
    else:
        raise ValueError("Missing expected label column: 'label' or 'ilr_level'")

    test_texts = df['text'].tolist()

    encodings = tokenizer(
        test_texts,
        max_length=MAX_TOKENS,
        padding='max_length',
        truncation=True,
        return_tensors='tf',
        return_token_type_ids=False
    )

    test_ds = tf.data.Dataset.from_tensor_slices((
        {'input_ids': encodings['input_ids'],
         'attention_mask': encodings['attention_mask']}
    )).batch(BATCH_SIZE)

    predictions = model.predict(test_ds)
    predicted_indices = tf.argmax(predictions, axis=1).numpy()

    return pd.DataFrame({
        'text': test_texts,
        'true_label': [LABEL_MAP_INVERSE[l] for l in test_labels],
        'predicted_label': [LABEL_MAP_INVERSE[idx] for idx in predicted_indices]
    })



def plot_training_history(history, save_path='training_plot.png'):
    for metric in history.history:
        plt.plot(history.history[metric], label=metric)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)  
    plt.close()             
    print(f"Training plot saved to {save_path}")



def train_model(train_path, dev_path):
    try:
        tf.keras.backend.clear_session()
        tokenizer = initialize_tokenizer()
        
        train_texts, test_texts, train_labels, test_labels = load_and_prepare_data(train_path, dev_path)
        
        train_labels = np.array(train_labels, dtype=int)
        test_labels = np.array(test_labels, dtype=int)


        global MAX_CHUNKS
        MAX_CHUNKS = get_max_chunks(train_texts, tokenizer)
        MAX_CHUNKS = min(MAX_CHUNKS, 5) 
        print(f"Using MAX_CHUNKS = {MAX_CHUNKS}")
        print(f"Training samples: {len(train_texts)}, Validation samples: {len(test_texts)}")

        train_ds = build_chunked_dataset(train_texts, train_labels, tokenizer)
        test_ds = build_chunked_dataset(test_texts, test_labels, tokenizer)

        model = build_model(MAX_CHUNKS, MAX_TOKENS)  
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
        
        classes = np.unique(train_labels)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
        class_weight = {int(k): float(v) for k, v in zip(classes, weights)}
        print("Class weights:", class_weight)

        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=8,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=4),
                tf.keras.callbacks.ModelCheckpoint('./best_model'),
                tensorboard_callback
            ],
            class_weight=class_weight
        )

        plot_training_history(history)
        
        model.save('./final_model_arabic', save_format="tf")
        tokenizer.save_pretrained('./tokenizer_arabic')

        return model, tokenizer
    
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train_path = os.path.join(os.path.dirname(__file__), 'Training-Testing Files', 'train_arabic.jsonl')
    dev_path = os.path.join(os.path.dirname(__file__), 'Training-Testing Files', 'dev_arabic.jsonl')

    trained_model, trained_tokenizer = train_model(train_path, dev_path)

    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_data = [json.loads(line) for line in f]
    df = pd.DataFrame(dev_data)

    if 'text' in df.columns:
        df['text'] = df['text'].astype(str).str.strip()
    elif 'english_text' in df.columns:
        df['text'] = df['english_text'].astype(str).str.strip()
    else:
        raise ValueError("Missing expected text column: 'text' or 'english_text'")

    if 'label' in df.columns:
        df['label'] = df['label'].astype(int)
    elif 'ilr_level' in df.columns:
        df['label'] = df['ilr_level'].astype(str).map(LABEL_MAP)
    else:
        raise ValueError("Missing expected label column: 'label' or 'ilr_level'")

    test_labels = df['label'].values
    mean_preds, max_preds = predict_with_chunking(df['text'].tolist(), trained_model, trained_tokenizer)

    df['true_label'] = test_labels
    df['mean_predicted_label'] = mean_preds
    df['max_predicted_label'] = max_preds


    df.to_csv('test_set_predictions.csv', index=False, columns=['text', 'true_label', 'mean_predicted_label', 'max_predicted_label'])

    df['true_label'] = df['true_label'].astype(int)
    df['mean_predicted_label'] = df['mean_predicted_label'].astype(int)
    df['max_predicted_label'] = df['max_predicted_label'].astype(int)


    print("\n=== Mean Prediction Evaluation ===")
    mean_acc = (df['true_label'] == df['mean_predicted_label']).mean()
    print(f"Mean Accuracy: {mean_acc:.2%}")

    used_labels = [0, 1, 2, 3, 4]
    print(classification_report(df['true_label'], df['mean_predicted_label'], labels=used_labels))

    cm_mean = confusion_matrix(df['true_label'], df['mean_predicted_label'], labels=used_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_mean, annot=True, fmt='d',
                xticklabels=used_labels,
                yticklabels=used_labels)
    plt.xlabel('Predicted (Mean)')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Mean Prediction')
    plt.show()

    print("\n=== Max Prediction Evaluation ===")
    max_acc = (df['true_label'] == df['max_predicted_label']).mean()
    print(f"Max Accuracy: {max_acc:.2%}")

    used_labels_max = [0, 1, 2, 3, 4]
    print(classification_report(df['true_label'], df['max_predicted_label'], labels=used_labels_max))

    cm_max = confusion_matrix(df['true_label'], df['max_predicted_label'], labels=used_labels_max)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_max, annot=True, fmt='d',
                xticklabels=used_labels_max,
                yticklabels=used_labels_max)
    plt.xlabel('Predicted (Max)')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Max Prediction')
    plt.show()

    print("\n=== [Fast Test] Verifying Chunking + Model Prediction ===")
    sample_texts = df['text'].tolist()[:5]
    sample_mean_preds, sample_max_preds = predict_with_chunking(sample_texts, trained_model, trained_tokenizer)

    for i, text in enumerate(sample_texts):
        print(f"\n--- Sample {i+1} ---")
        print(f"Text: {text[:100]}...")
        print(f"Mean prediction: {LABEL_MAP_INVERSE[sample_mean_preds[i]]}")
        print(f"Max prediction:  {LABEL_MAP_INVERSE[sample_max_preds[i]]}")


    
