# LLM From Scratch using PyTorch

This project implements a **GPT-style Large Language Model (LLM) from scratch** using PyTorch.
It covers everything from **attention mechanisms → transformer blocks → pretraining → fine-tuning → instruction tuning**.

---

##  Features

* ✅ Multi-Head Self Attention (from scratch)
* ✅ Custom Layer Normalization
* ✅ GELU Activation Function
* ✅ Transformer Block Implementation
* ✅ Full GPT Architecture
* ✅ Text Generation (Greedy, Temperature, Top-K Sampling)
* ✅ Pretraining on custom dataset
* ✅ Classification Fine-tuning (Spam Detection)
* ✅ Instruction Fine-tuning
* ✅ Loading Pretrained GPT-2 Weights
* ✅ Evaluation using external LLM (Ollama)

---

##  Model Architecture

The model follows a **decoder-only Transformer (GPT-style)** architecture:

* Token Embeddings + Positional Embeddings
* Stacked Transformer Blocks:

  * Multi-Head Attention
  * Feed Forward Network (GELU)
  * Residual Connections
  * Layer Normalization
* Final Linear Output Head



---

## Installation

```bash
pip install torch numpy matplotlib pandas tqdm tiktoken tensorflow
```

---

##  Training Pipeline

### 1️⃣ Data Preparation

* Tokenization using `tiktoken`
* Sliding window dataset creation
* Train/Validation split

---

### 2️⃣ Pretraining

* Objective: Next Token Prediction
* Loss: Cross Entropy
* Metrics:

  * Loss
  * Perplexity

```python
loss = torch.nn.functional.cross_entropy(logits, targets)
perplexity = torch.exp(loss)
```

---

### 3️⃣ Text Generation

Supports multiple decoding strategies:

* Greedy (argmax)
* Temperature sampling
* Top-K sampling

```python
generate(model, idx, temperature=1.4, top_k=25)
```

---

### 4️⃣ Fine-Tuning (Classification)

* Task: Spam Detection
* Dataset: SMS Spam Dataset (UCI)
* Approach:

  * Freeze base model
  * Replace output head
  * Train last layers

---

### 5️⃣ Instruction Fine-Tuning

* Dataset: Instruction-response pairs
* Format:

```
### Instruction:
...

### Input:
...

### Response:
...
```

* Uses custom collate function for:

  * Padding
  * Masking (-100 for ignored tokens)

---

## Key Components

### Multi-Head Attention

* Scaled dot-product attention
* Causal masking (no future token leakage)

---

### Custom LayerNorm

```python
norm_x = (x - mean) / sqrt(var + eps)
```

---

### GELU Activation

Better than ReLU for transformers.

---

### Transformer Block

* Attention + FFN
* Residual connections
* Dropout

---

##  Evaluation

* Training vs Validation Loss
* Accuracy (for classification)
* Perplexity (for language modeling)

---

##  Example Outputs

```text
Input: "Every effort moves you"
Output: "Every effort moves you closer to your goal..."
```

---

## Pretrained Model Support

* Downloads GPT-2 weights
* Maps weights into custom architecture
* Enables transfer learning

---

##  Advanced Features

* Temperature Scaling
* Top-K Sampling
* Custom Dataset Classes
* Custom Collate Function
* Instruction Dataset Formatting
* External Evaluation via Ollama (LLaMA3)



---

## Learning Outcomes

This project helps you understand:

* How LLMs work internally
* Transformer architecture in depth
* Training pipelines for LLMs
* Fine-tuning strategies
* Tokenization & sequence modeling

---

## License

This project is for educational purposes.

---

## Acknowledgements

* GPT-2 Architecture (OpenAI)
* PyTorch
* UCI Dataset
* Raschka's LLM from Scratch resources

---

## Author

**Avatanshu Gupta**

---
