# POS Tagging and Language Translation Project

## Project Overview
This project explores **Natural Language Processing (NLP)** through two major tasks:
1. **Part-of-Speech (POS) tagging** using a **Bidirectional LSTM model**.  
2. **Language translation** using **pre-trained transformers** from Hugging Face.  

The aim was to understand sequential data processing in NLP, build deep learning models to predict POS tags, and explore state-of-the-art transformer models for translation.

---

## Part 1: POS Tagging with LSTM

### Objective
To develop a deep learning model that predicts the **POS tags** of words in a sentence using the **Treebank corpus** (simplified universal tagset).

### Dataset
- Used **NLTK’s Treebank corpus**, with **simplified POS tags**.  
- Selected a subset of **2,000 sentences** for training and testing.  

Example of a tagged sentence:
```python
Sentence: ['Pierre', 'Vinken', ',', '61', 'years', 'old', ',', 'will', 'join', 'the', 'board', 'as', 'a', 'nonexecutive', 'director', 'Nov.', '29', '.']
POS Tags: ['NOUN', 'NOUN', '.', 'NUM', 'NOUN', 'ADJ', '.', 'VERB', 'VERB', 'DET', 'NOUN', 'ADP', 'DET', 'ADJ', 'NOUN', 'NOUN', 'NUM', '.'].
```

## Data Preprocessing

- **Tokenization**: Split sentences into words and POS tags, and converted them into sequences.  
- **Padding**: Ensured uniform length across sequences using `pad_sequences`.  
- **One-hot encoding**: Converted POS tags into categorical format for training.  

---

## Model Architecture

- **Embedding Layer**: Converts words into dense vectors (`embedding_dim = 64`).  
- **Bidirectional LSTM**: Captures contextual information from both past and future words (`lstm_units = 64`).  
- **TimeDistributed Dense Layer**: Outputs POS tag probabilities for each word using `softmax` activation.  

Input -> Embedding -> Bidirectional LSTM -> TimeDistributed Dense (softmax) -> Output


- **Loss Function**: Categorical cross-entropy  
- **Optimizer**: Adam  
- **Evaluation Metric**: Accuracy  

---

## Training & Validation

- Trained on **2,000 sentences** with **90% training and 10% validation split**.  
- **Epochs**: 5  
- **Batch size**: 32  

---

## Prediction Example

```python
Sentence: ['She', 'joins', 'the','class']
Predicted POS tags: ['PRON', 'VERB', 'DET', 'NOUN']
```

Outcome: The model accurately predicted the POS tags, demonstrating strong performance in sequential word tagging.

# Part 2: Language Translation with Transformers

## Objective

To explore transformer models for language translation and understand why they perform better on sequential data compared to traditional RNN-based models.

Approach

* Utilized pre-trained Hugging Face transformer models for translation tasks.

* Transformers rely on self-attention mechanisms, allowing the model to capture long-range dependencies in sequences efficiently, unlike LSTMs which process data sequentially.

* Translation involved encoding the source sentence and decoding into the target language, leveraging pre-trained weights for accurate results.


### Why Transformers are Better for Sequential Data

* Parallel Processing: Unlike LSTM/RNN, transformers process entire sequences simultaneously, speeding up training.

* Long-Range Dependency Handling: Self-attention allows models to learn relationships between distant words in a sentence.

* Scalability: Transformers can scale to larger datasets and complex NLP tasks more effectively.


### Outcome

* Successfully translated sentences between languages with high fidelity using Hugging Face transformers.

* Demonstrated the superior handling of sequential dependencies compared to LSTMs.

* Highlighted practical usage of pre-trained models for NLP tasks beyond POS tagging.


### Project Highlights

* Built a Bidirectional LSTM POS tagger achieving high accuracy on a subset of the Treebank corpus.

* Implemented sequence preprocessing, tokenization, padding, and one-hot encoding for POS tagging.

* Explored Hugging Face transformers for language translation.

* Demonstrated the strength of transformers in handling long-term dependencies in sequential data.

* Gained hands-on experience with both RNN-based and transformer-based architectures for NLP.


### Technologies Used

* Python

* TensorFlow / Keras

* NLTK (Treebank Corpus)

* Hugging Face Transformers

* NumPy, Pandas




