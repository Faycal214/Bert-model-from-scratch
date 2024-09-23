import tensorflow as tf
import random
import nltk
from transformers import BertTokenizer
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Embedding, MultiHeadAttention
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

class Bert:
    def __init__(self, vocab_size, d_model, num_heads, dff, num_layers, rate=0.1):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.rate = rate

        self.embedding_layer = self.BertEmbeddings(vocab_size, d_model)
        self.transformer_blocks = [self.BertEncoderBlock(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)
        self.dense_mlm = Dense(vocab_size)  # Output layer for MLM
        self.dense_nsp = Dense(2)  # Output layer for NSP (binary classification)


    class BertEmbeddings:
        def __init__(self, vocab_size, d_model):
            self.token_embeddings = Embedding(input_dim=vocab_size, output_dim=d_model)
            self.segment_embeddings = Embedding(input_dim=2, output_dim=d_model)
            self.position_embeddings = Embedding(input_dim=5000, output_dim=d_model)

        def __call__(self, token_ids, segment_ids):
            seq_length = tf.shape(token_ids)[1]

            token_embed = self.token_embeddings(token_ids)
            segment_embed = self.segment_embeddings(segment_ids)

            position_indices = tf.range(start=0, limit=seq_length, delta=1)
            position_embeddings = self.position_embeddings(position_indices)

            embeddings = token_embed + segment_embed + position_embeddings
            return embeddings

        @property
        def trainable_variables(self):
            return (self.token_embeddings.trainable_variables +
                    self.segment_embeddings.trainable_variables +
                    self.position_embeddings.trainable_variables)


    class BertEncoderBlock:
        def __init__(self, d_model, num_heads, d_ff, learning_rate=0.1):
            self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
            self.dense1 = Dense(d_ff, activation='relu')
            self.dense2 = Dense(d_model)
            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)
            self.dropout1 = Dropout(learning_rate)
            self.dropout2 = Dropout(learning_rate)

        def __call__(self, x, training=False):
            attn_output = self.attention(x, x)
            out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
            ffn_output = self.dense2(self.dense1(out1))
            return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

        @property
        def trainable_variables(self):
            return (self.attention.trainable_variables +
                    self.dense1.trainable_variables +
                    self.dense2.trainable_variables +
                    self.layernorm1.trainable_variables +
                    self.layernorm2.trainable_variables)


    def process_texts(self, sentence_a, sentence_b, mask_proba=0.15):
        tokens_a = self.tokenizer.tokenize(sentence_a)
        tokens_b = self.tokenizer.tokenize(sentence_b)

        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]']

        def mask_tokens(tokens):
            output_tokens = []
            mlm_labels = []
            for token in tokens:
                if random.random() < mask_proba and token not in ['[CLS]', '[SEP]']:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    mlm_labels.append(token_id)
                    prob = random.random()
                    if prob < 0.8:
                        output_tokens.append('[MASK]')
                    elif prob < 0.9:
                        random_token = self.tokenizer.convert_ids_to_tokens(random.randint(0, self.vocab_size - 1))
                        output_tokens.append(random_token)
                    else:
                        output_tokens.append(token)
                else:
                    output_tokens.append(token)
                    mlm_labels.append(-100) # for the tokens that havent been chose randomly for the masking process

            return output_tokens, mlm_labels

        masked_tokens_a, mlm_labels_a = mask_tokens(tokens_a)
        masked_tokens_b, mlm_labels_b = mask_tokens(tokens_b)

        final_masked_tokens = masked_tokens_a + masked_tokens_b
        input_ids = self.tokenizer.convert_tokens_to_ids(final_masked_tokens)
        segment_ids = [0] * len(masked_tokens_a) + [1] * len(masked_tokens_b)

        return input_ids, segment_ids, mlm_labels_a + mlm_labels_b, 1  # Assuming is_next


    def __call__(self, sentence_a, sentence_b):
        # Get the token and segment ids from the input sentences
        token_ids, segment_ids, _ , _ = self.process_texts(sentence_a, sentence_b)

        # Convert to tensors and expand dimensions for batch processing
        token_ids = tf.expand_dims(tf.convert_to_tensor(token_ids, dtype=tf.int32), axis=0)
        segment_ids = tf.expand_dims(tf.convert_to_tensor(segment_ids, dtype=tf.int32), axis=0)

        # Get embeddings and pass through transformer blocks
        embeddings = self.embedding_layer(token_ids, segment_ids)
        x = self.dropout(embeddings)

        # Pass through all transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, training=True)

        # Get the MLM output (logits)
        mlm_output = self.dense_mlm(x)  # Shape: (batch_size, sequence_length, vocab_size)

        # Get the NSP output (logits)
        nsp_output = self.dense_nsp(x[:, 0, :])  # Use [CLS] token for NSP. Shape: (batch_size, 2)

        # ----- NSP Prediction -----
        # Apply softmax to get the probabilities for NSP (binary classification)
        nsp_output = tf.nn.softmax(nsp_output, axis=-1)  # Shape: (batch_size, 2)

        # Get the final NSP prediction (0 or 1) by taking argmax
        nsp_output = nsp_output.numpy()[0]
        return mlm_output, nsp_output

    @property
    def trainable_variables(self):
        # Gather trainable variables from embeddings and all transformer blocks
        trainable_vars = self.embedding_layer.trainable_variables
        for transformer in self.transformer_blocks:
            trainable_vars += transformer.trainable_variables
        # Add output layer variables
        trainable_vars += self.dense_mlm.trainable_variables
        trainable_vars += self.dense_nsp.trainable_variables
        return trainable_vars