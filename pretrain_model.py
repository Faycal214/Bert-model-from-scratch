from main import Bert
import tensorflow as tf
import nltk
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

# Example usage
nltk.download('punkt')

nltk.download('punkt')

vocab_size = tokenizer.vocab_size  # Example vocab size, replace with actual tokenizer vocab size
num_heads = 8
dff = 2048
num_layers = 12
d_model = 512

# create the model object
bert_model = Bert(vocab_size, d_model, num_heads, dff, num_layers)
# define the tokenizer object of BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Define the optimizer
optimizer = Adam(learning_rate=1e-4)


# READ THE DATASET
df = pd.read_parquet("hf://datasets/PaulineSanchez/Translation_words_and_sentences_english_french/data/train-00000-of-00001-3d14582ea46e1b17.parquet")

# DATA PREPROCESSING
def process_row(row):
    # Extract the French and English sentences from the row
    sentence_a = row['English words/sentences']
    sentence_b = row['French words/sentences']
    # Tokenize and prepare inputs for MLM and NSP
    _, _, mlm_labels, nsp_labels = bert_model.process_texts(sentence_a, sentence_b)

    return _, _, mlm_labels, nsp_labels

def process_dataset(data):
    all_mlm_labels, all_nsp_labels = [], []

    for i in tqdm(range(data.shape[0]), total=data.shape[0], ncols=65):
        _, _, mlm_labels, nsp_labels = process_row(data.iloc[i])

        if mlm_labels is not None and nsp_labels is not None:
            all_mlm_labels.append(mlm_labels)
            all_nsp_labels.append(nsp_labels)

    return all_mlm_labels, all_nsp_labels

# Prepare the data for training
mlm_labels, nsp_labels = process_dataset(df)
# Ensure they are tensors
mlm_labels= tf.expand_dims(tf.ragged.constant(mlm_labels), axis= 0)
nsp_labels = tf.expand_dims(tf.ragged.constant(nsp_labels), axis= 0)


#### LOSS FUNCTIONS 
# MLM Loss Function (SparseCategoricalCrossentropy)
def mlm_loss_fn(real_labels, predictions):
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)

    # Convert RaggedTensors to dense tensors
    real_labels = real_labels.to_tensor(default_value=-100)
    predictions = predictions.to_tensor()

    # Create mask: positions where real_labels are not -100 (i.e., masked tokens)
    mask = tf.not_equal(real_labels, -100)

    # Apply the mask: only consider masked tokens for loss calculation
    real_labels_masked = tf.boolean_mask(real_labels, mask)
    predictions_masked = tf.boolean_mask(predictions, mask)

    # Calculate loss only on masked tokens
    loss = loss_fn(real_labels_masked, predictions_masked)

    return tf.reduce_mean(loss)

# NSP Loss Function (Binary Crossentropy)
def nsp_loss_fn(real_labels, predictions):
    loss_fn = BinaryCrossentropy(from_logits=True)

    # Convert RaggedTensors to dense tensors
    real_labels = real_labels.to_tensor()
    predictions = predictions.to_tensor()

    # Calculate NSP loss
    loss = loss_fn(real_labels, predictions)

    return tf.reduce_mean(loss)



### PRE-TRAINING MODEL
# compute the MLM loss and NSP loss
# use the optimizer to apply gradients and update model weights

# Pretraining step for BERT
def pretraining_step(mlm_labels, nsp_labels, bert_model, batch_data):
    with tf.GradientTape() as tape:
        # Forward pass
        mlm_outputs, nsp_outputs = [], []
        for i in range(batch_data.shape[0]) :
            sentence_a, sentence_b = batch_data['English words/sentences'].iloc[i], batch_data['French words/sentences'].iloc[i]

            mlm_output, nsp_output = bert_model(sentence_a, sentence_b)

            mlm_outputs.append(mlm_output)
            nsp_outputs.append(nsp_output)

        # Convert mlm_outputs to a RaggedTensor
        mlm_outputs = tf.ragged.constant(mlm_outputs)
        nsp_outputs = tf.ragged.constant(nsp_outputs)

        # Calculate losses
        mlm_loss = mlm_loss_fn(mlm_labels, mlm_outputs)
        nsp_loss = nsp_loss_fn(nsp_labels, nsp_outputs)

        # Total loss (sum of MLM and NSP losses)
        total_loss = mlm_loss + nsp_loss

    # Compute gradients and apply them
    gradients = tape.gradient(total_loss, bert_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, bert_model.trainable_variables))

    return total_loss

# Training loop :
# It will iterate over the dataset for a fixed number of epochs and update the model's weights
def pretrain_bert(bert_model, mlm_labels, nsp_labels , data, epochs, batch_size=128):
    dataset_size = data.shape[0]
    for epoch in range(epochs) :
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0

        for i in tqdm(range(0, dataset_size, batch_size), total=dataset_size, desc= 'Batches', ncols= 65) :
            batch_mlm_labels = mlm_labels[i:i+batch_size]
            batch_nsp_labels = nsp_labels[i:i+batch_size]
            batch_data = data.iloc[i: i+batch_size, :]
            # Perform a pretraining step
            loss = pretraining_step(batch_mlm_labels, batch_nsp_labels, bert_model, batch_data)
            epoch_loss += loss

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / (dataset_size // batch_size)}")

# Call the training loop
pretrain_bert(bert_model, mlm_labels, nsp_labels, df, epochs=17)

# Assuming `bert_model` is an instance of your BertModel class
checkpoint = tf.train.Checkpoint(model=bert_model)

# Save the checkpoint
checkpoint.save('bert_checkpoint')
print("Model saved successfully!")