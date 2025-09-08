# 🧩 Building GPT-2 from Scratch

This contains an implementation of **GPT-2**, built entirely from scratch inside a Colab notebook.  
The goal is to break down how large language models like GPT-2 work internally, by coding each building block step by step.

---

## 🚀 Features
- Implements GPT-2 core components **without Hugging Face or external transformer libraries**
- Covers:
  - Token and positional embeddings  
  - Multi-head self-attention (with causal masking)  
  - Feed-forward layers  
  - Skip (residual) connections  
  - Layer normalization  
  - Transformer block stacking  
  - Training loop with text generation
- Generates text after training on a sample dataset  

---

## 🧠 Key Concepts

### 🔹 Multi-Head Self-Attention
Self-attention allows each token to “look at” other tokens in the sequence, learning dependencies regardless of distance.  
- **Scaled Dot-Product Attention** computes similarity between queries, keys, and values.  
- Multiple heads run in parallel, each learning different relationships.  
- Their outputs are concatenated and projected back to the embedding dimension.

### 🔹 Skip (Residual) Connections
Training deep models is challenging because gradients may vanish.  
Skip connections add the input of a layer back to its output:  

\[
\text{Output} = \text{Layer}(x) + x
\]

This helps stabilize training and preserve information across layers.

### 🔹 Layer Normalization
Instead of normalizing across the batch, **LayerNorm** normalizes across features within each token.  
This keeps training stable and accelerates convergence. GPT-2 uses the **Pre-LN** setup (normalization before each sublayer).




## 📖 References
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)  
- [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  


