from transformers.models.xlm_roberta import TFXLMRobertaModel
import tensorflow as tf
from metrics import MacroPrecision as precision, MacroRecall as recall, MacroF1Score as f1_score

NUM_CLASSES = 5
EMBED_DIM = 768  

def build_model(MAX_CHUNKS, MAX_TOKENS):
    inputs = {
        'input_ids': tf.keras.Input(shape=(MAX_CHUNKS, MAX_TOKENS), dtype=tf.int32, name='input_ids'),
        'attention_mask': tf.keras.Input(shape=(MAX_CHUNKS, MAX_TOKENS), dtype=tf.int32, name='attention_mask')
    }

    base_model = TFXLMRobertaModel.from_pretrained("xlm-roberta-base")

    flat_input_ids = tf.reshape(inputs['input_ids'], (-1, MAX_TOKENS))
    flat_attention_mask = tf.reshape(inputs['attention_mask'], (-1, MAX_TOKENS))

    outputs = base_model(flat_input_ids, attention_mask=flat_attention_mask)
    pooled_output = outputs.pooler_output  

    chunk_embeddings = tf.reshape(pooled_output, (-1, MAX_CHUNKS, pooled_output.shape[-1]))

    pooled = tf.reduce_mean(chunk_embeddings, axis=1)

    dense1 = tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(pooled)
    dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
    dense2 = tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.3)(dense2)

    output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(dropout2)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss='sparse_categorical_crossentropy',
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            precision(num_classes=NUM_CLASSES, name='macro_precision'),
            recall(num_classes=NUM_CLASSES, name='macro_recall'),
            f1_score(num_classes=NUM_CLASSES, name='macro_f1')
        ]
    )

    return model

