# Machine Learning, Deep Learning, and LLM Models Overview

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

## 8. Large Language Models (LLMs)
- **Encoder-Only Models**:
  - *BERT*: Bidirectional pre-training for tasks like QA.
  - *RoBERTa*: Optimized BART with dynamic masking.
- **Decoder-Only Models**:
  - *GPT* (Generative Pre-trained Transformer): Autoregressive text generation (e.g., GPT-3, GPT-4).
  - *LLaMA/Mistral*: Open-source LLMs for research.
- **Encoder-Decoder Models**:
  - *T5*: Text-to-text framework for diverse NLP tasks.
  - *BART*: Denoising autoencoder for generation/translation.
- **Sparse/Retrieval-Augmented Models**:
  - *RETRO*: Combines retrieval with LLMs.
  - *PaLM*: Google’s large-scale LLM.

## 9. Key Architectures in Modern ML
- **Attention Mechanisms**: Core to transformers (e.g., multi-head attention).
- **Vision Transformers (ViT)**: Applies transformers to image patches.
- **Graph Neural Networks (GNN)**: Processes graph-structured data.
- **Capsule Networks**: Encodes hierarchical spatial relationships.

## 10. Reinforcement Learning (Bonus)
- **Q-Learning**: Value-based method for policy learning.
- **Deep Q-Networks (DQN)**: Combines Q-learning with deep neural networks.
- **Policy Gradient Methods**: Directly optimizes policy parameters (e.g., PPO).