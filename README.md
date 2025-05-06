# Machine Learning, Deep Learning, and NLP Models Overview

## 1. Traditional Machine Learning Models
- **Linear Regression**
  - *Simple Linear Regression*: Predicts a continuous output using a single feature.
  - *Multiple Linear Regression*: Uses multiple features for prediction.
  - *Polynomial/Non-linear Regression*: Models non-linear relationships with polynomial terms.
- **Logistic Regression**: Binary classification using a sigmoid function.
- **Softmax Regression**: Multi-class extension of logistic regression.
- **k-Nearest Neighbors (kNN)**: Instance-based learning for classification/regression.
- **Decision Trees**: Hierarchical splits based on feature thresholds (e.g., CART, ID3).
- **Support Vector Machines (SVM)**: Finds optimal hyperplanes for classification/regression.
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem (e.g., Gaussian, Multinomial).

## 2. Deep Learning Architectures
- **Multilayer Perceptron (MLP)**: Basic neural network with fully connected layers.
- **Convolutional Neural Networks (CNN)**: For grid-like data (images), uses convolutional layers (e.g., ResNet, VGG, Inception).
- **Recurrent Neural Networks (RNN)**: Processes sequential data (e.g., time series, text).
  - *Long Short-Term Memory (LSTM)*: Addresses vanishing gradients in RNNs.
  - *Gated Recurrent Unit (GRU)*: Simplified LSTM variant.
- **Transformers**: Self-attention-based architecture for sequences (basis for LLMs).
- **Autoencoders**: Unsupervised learning for dimensionality reduction or generation.
- **Generative Adversarial Networks (GAN)**: Generates data via generator-discriminator competition.
- **Diffusion Models**: Generates data by iteratively denoising (e.g., Stable Diffusion).

## 3. Ensemble Methods
- **Random Forest**: Ensemble of decorrelated decision trees.
- **Gradient Boosting Machines (GBM)**:
  - *XGBoost*: Optimized gradient boosting.
  - *LightGBM*: Efficient histogram-based boosting.
  - *CatBoost*: Handles categorical features natively.
- **Stacking**: Combines multiple models via a meta-learner.
- **Bagging (Bootstrap Aggregating)**: Reduces variance (e.g., Bagged Trees).

## 4. Clustering Algorithms
- **k-Means**: Partitions data into *k* clusters.
- **Hierarchical Clustering**: Builds nested clusters via dendrograms.
- **DBSCAN**: Density-based clustering for arbitrary-shaped clusters.
- **Gaussian Mixture Models (GMM)**: Probabilistic clustering via Gaussian distributions.

## 5. Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Linear projection to orthogonal components.
- **t-SNE**: Non-linear visualization of high-dimensional data.
- **UMAP**: Preserves global and local structure for visualization/clustering.
- **Linear Discriminant Analysis (LDA)**: Supervised dimensionality reduction.

## 6. Probabilistic Models
- **Gaussian Processes**: Non-parametric Bayesian regression.
- **Hidden Markov Models (HMM)**: Models sequences with hidden states.

## 7. Optimization Techniques
- **Mini-Batch Gradient Descent**: Updates weights using subsets of data.
- **Stochastic Gradient Descent (SGD)**: Updates weights per data point.
- **Adam/RMSprop**: Adaptive learning rate optimizers.

## 8. Natural Language Processing (NLP) Topics
### Text Preprocessing & Feature Extraction
- **Tokenization**: Splitting text into words, subwords, or sentences.
- **Stemming/Lemmatization**: Reducing words to root forms.
- **Stopword Removal**: Filtering common non-informative words.
- **TF-IDF**: Term frequency-inverse document frequency for text vectorization.
- **Word Embeddings**: Distributed representations (e.g., Word2Vec, GloVe, FastText).
- **Subword Embeddings**: Byte-Pair Encoding (BPE), SentencePiece.

### Core NLP Tasks
- **Named Entity Recognition (NER)**: Identifying entities (e.g., persons, locations).
- **Part-of-Speech (POS) Tagging**: Labeling words with grammatical roles.
- **Sentiment Analysis**: Classifying text sentiment (positive/negative/neutral).
- **Text Classification**: Categorizing documents (e.g., spam detection).
- **Machine Translation**: Translating text between languages (e.g., seq2seq models).
- **Question Answering**: Extracting answers from context (e.g., SQuAD).
- **Text Summarization**: Generating concise summaries (extractive/abstractive).
- **Coreference Resolution**: Linking pronouns to entities.

### Traditional NLP Techniques
- **n-gram Language Models**: Probability-based text generation.
- **Conditional Random Fields (CRF)**: For sequence labeling tasks (e.g., NER).
- **Latent Dirichlet Allocation (LDA)**: Topic modeling.

### Neural NLP Models
- **Seq2Seq Models**: Encoder-decoder architecture for translation/summarization.
- **Attention Mechanisms**: Focus on relevant context (e.g., Bahdanau attention).
- **Transformer-Based Models**: BERT, GPT, T5, etc. (see LLM section below).
- **Siamese Networks**: For semantic similarity tasks.

### Evaluation Metrics (NLP)
- **BLEU**: For machine translation quality.
- **ROUGE**: For summarization and text generation.
- **METEOR**: Semantic-aware translation metric.
- **F1 Score**: For classification/sequence labeling tasks.
- **Perplexity**: For language model evaluation.

### NLP Tools & Libraries
- **NLTK**: Classic NLP toolkit for Python.
- **spaCy**: Industrial-strength NLP library.
- **Hugging Face Transformers**: State-of-the-art pretrained models (BERT, GPT, etc.).
- **Gensim**: Topic modeling and word embeddings.
- **Stanford CoreNLP**: Java-based NLP toolkit.
- **AllenNLP**: Research-focused NLP library.

### Advanced Topics
- **Multilingual NLP**: Handling low-resource languages (e.g., mBERT, XLM-R).
- **Transfer Learning in NLP**: Fine-tuning pretrained models.
- **Ethics in NLP**: Bias detection/mitigation, fairness, and interpretability.

## 9. Large Language Models (LLMs)
- **Encoder-Only Models**:
  - *BERT*: Bidirectional pre-training for tasks like QA.
  - *RoBERTa*: Optimized BERT with dynamic masking.
- **Decoder-Only Models**:
  - *GPT* (Generative Pre-trained Transformer): Autoregressive text generation (e.g., GPT-3, GPT-4).
  - *LLaMA/Mistral*: Open-source LLMs for research.
- **Encoder-Decoder Models**:
  - *T5*: Text-to-text framework for diverse NLP tasks.
  - *BART*: Denoising autoencoder for generation/translation.
- **Sparse/Retrieval-Augmented Models**:
  - *RETRO*: Combines retrieval with LLMs.
  - *PaLM*: Google’s large-scale LLM.

## 10. Key Architectures in Modern ML
- **Attention Mechanisms**: Core to transformers (e.g., multi-head attention).
- **Vision Transformers (ViT)**: Applies transformers to image patches.
- **Graph Neural Networks (GNN)**: Processes graph-structured data.
- **Capsule Networks**: Encodes hierarchical spatial relationships.

## 11. Reinforcement Learning (Bonus)
- **Q-Learning**: Value-based method for policy learning.
- **Deep Q-Networks (DQN)**: Combines Q-learning with deep neural networks.
- **Policy Gradient Methods**: Directly optimizes policy parameters (e.g., PPO).