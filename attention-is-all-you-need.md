**“Attention Is All You Need” (Vaswani et al., 2017)** 

_Attribution:_  
The paper “Attention Is All You Need” was originally published as:  
> Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
> Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin (2017).  
> *Attention Is All You Need*. arXiv:1706.03762v7 [cs.CL].  

https://arxiv.org/abs/1706.03762

---

## Table of Contents

1. **Introduction (ELI5 Overview)**  
2. **The Core Idea: Attention Mechanisms**  
3. **The Transformer Architecture**  
   - 3.1 Encoder  
   - 3.2 Decoder  
   - 3.3 Multi-Head Self-Attention  
   - 3.4 Feed-Forward Network (FFN)  
   - 3.5 Positional Encoding  
4. **Why Self-Attention Over RNNs or CNNs?**  
5. **Training Details**  
6. **Results and Main Contributions**  
   - 6.1 Machine Translation  
   - 6.2 Model Variations  
   - 6.3 Parsing Task  
7. **Working Code Samples**  
   - 7.1 Scaled Dot-Product Attention Example  
   - 7.2 Multi-Head Attention Example  
   - 7.3 Putting It All Together (Mini-Transformer)  
8. **Conclusion**  

---

## 1. Introduction (ELI5 Overview)

The big question addressed by the paper:  
“How can a neural network process sequences (like sentences) efficiently and effectively without relying on the usual recurrence (RNNs) or convolution (CNNs)?”  

**RNNs** (like LSTM or GRU) read a sentence word by word in order, making them slow to train when sentences get long.  
**CNNs** look at local chunks of words (like a sliding window) to speed things up, but still need multiple stacked layers to see the whole sentence.  

The paper’s answer:  
> “We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.”  

Attention is a method for a model to look at all positions in the input (or all previously generated words in the output) at once, focusing on those most relevant at each step. This “self-attention” approach is extremely parallelizable, enabling faster training than RNNs, and can capture long-range dependencies better than CNNs.  

---

## 2. The Core Idea: Attention Mechanisms

An attention mechanism takes a **query** and a set of **key–value pairs** and computes an output as a weighted sum of the values, where each weight is determined by a similarity of the query to the corresponding key.

> “An attention function can be described as mapping a query and a set of key-value pairs to an output.”  

### Scaled Dot-Product Attention (Excerpt)

> “We compute the dot products of the query with all keys, divide each by the square root of the dimension of the key, and apply a softmax function to obtain the weights on the values.”  

Mathematically:

\[
\text{Attention}(Q, K, V) = \text{softmax}\Bigl(\frac{QK^T}{\sqrt{d_k}}\Bigr)V
\]

Where:  
- \(Q\) = Query matrix  
- \(K\) = Key matrix  
- \(V\) = Value matrix  
- \(d_k\) = dimension of each key vector  

---

## 3. The Transformer Architecture

The Transformer is an **encoder–decoder** structure, but it doesn’t use recurrent cells (like LSTMs). Instead, each layer uses a **multi-head self-attention** sub-layer and a **feed-forward** sub-layer.  

> “The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.”  

Figure below (reproduced from the paper) shows an overview:

```
( Reproduced Figure 1: The Transformer architecture. )

+-------------------------+              +-------------------------+
|        Encoder         |              |        Decoder         |
|   [Self-Attention]     |    --->      | [Masked Self-Attention]|
|   [Feed Forward]       |              | [Encoder-Decoder Attn] |
|        *Repeat*        |              | [Feed Forward]         |
+-------------------------+              +-------------------------+
```

### 3.1 Encoder

- The encoder is a stack of \(N\) identical layers (paper uses \(N = 6\) typically).  
- Each layer has:
  1. A **multi-head self-attention** sub-layer.  
  2. A **position-wise feed-forward** sub-layer.  
- Residual connections (add & layer norm) wrap around each sub-layer.

### 3.2 Decoder

- Similar stack of \(N\) layers.  
- Each layer has:
  1. A **masked multi-head self-attention** sub-layer (to prevent looking ahead in the output).  
  2. Multi-head **encoder–decoder attention** (queries from the decoder, keys and values from the encoder).  
  3. A **position-wise feed-forward** sub-layer.  

### 3.3 Multi-Head Self-Attention

Instead of a single attention over all hidden vectors, they use multiple attention “heads,” each focusing on different parts or aspects. Then these heads are concatenated.  

> “Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.”  

Formally, each head is a separate scaled dot-product attention:

\[
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V),
\]

and all heads are concatenated and projected again:

\[
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h) W^O.
\]

### 3.4 Feed-Forward Network (FFN)

After the self-attention sub-layer, there is a position-wise feed-forward network. This means the same 2-layer MLP is applied to each position’s vector, independently.  

\[
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2.
\]

### 3.5 Positional Encoding

Because there is no recurrence or convolution, the model doesn’t inherently know the order of tokens. So the paper adds **positional encodings**—sine and cosine signals of varying frequencies—to each embedding.  

> “We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions.”  

---

## 4. Why Self-Attention Over RNNs or CNNs?

RNNs are slow to train on long sequences because they must process tokens one after the other. CNNs can be parallelized but usually need many layers (or wide kernels) to capture distant dependencies.  

Self-attention layers:  
- Can see **all** positions in a single step (constant path length).  
- Require fewer sequential operations, enabling much better parallelization.  
- Often yield more interpretable attention patterns.  

> “In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions.”  

From Table 1 in the paper:

| Layer Type        | Complexity per Layer | Sequential Ops | Max Path Length |
|-------------------|----------------------|----------------|-----------------|
| Self-Attention    | \(O(n \cdot d)\)     | \(O(1)\)       | \(O(1)\)        |
| Recurrent         | \(O(n \cdot d^2)\)   | \(O(n)\)       | \(O(n)\)        |
| Convolutional     | \(O(k \cdot n \cdot d^2)\) | \(O(1)\)  | \(O(\log_k(n))\)|  

---

## 5. Training Details

The paper reports results on large-scale machine translation benchmarks. Key points:

- **Optimizer**: Adam with particular \(\beta_1, \beta_2\), and a learning-rate scheduling that **warms up** then **decays**.  
- **Regularization**:
  - Dropout (applied to attention weights, residual connections, and embeddings).  
  - Label smoothing (makes the model less overconfident).  
- **Hardware**: They trained on 8 NVIDIA P100 GPUs.  

> “We trained the base models for a total of 100,000 steps or 12 hours. … The big models … were trained for 300,000 steps (3.5 days).”  

---

## 6. Results and Main Contributions

### 6.1 Machine Translation

They tested on **WMT 2014 English-German** and **WMT 2014 English-French**.  

> “Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.”  

This was a major result at the time.  

### 6.2 Model Variations

The paper examines how changing hyperparameters like:
- number of heads \(h\),  
- model dimension \(d_\text{model}\),  
- feed-forward size \(d_\text{ff}\),  
- dropout rates, etc.  

affects performance. The “big” model has \(d_\text{model} = 1024\) and \(d_\text{ff} = 4096\).  

### 6.3 Parsing Task

They tested on **English constituency parsing** to show the Transformer is not just for translation. They achieved results comparable to or better than state-of-the-art.  

> “Despite the lack of task-specific tuning, our model performs surprisingly well, yielding better results than all previously reported models with the exception of the RNN Grammar.”  

---

## 7. Working Code Samples

Below are simplified code snippets in PyTorch to illustrate the core concepts behind the Transformer. These examples demonstrate the basic mechanics; they are not fully production-ready or optimized.

### 7.1 Scaled Dot-Product Attention Example

```python
import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [batch_size, heads, seq_len_q, d_k]
    K: [batch_size, heads, seq_len_k, d_k]
    V: [batch_size, heads, seq_len_k, d_v]
    mask: [batch_size, 1, 1, seq_len_k] (optional)
    """
    d_k = Q.size(-1)  # dimension of keys
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, heads, seq_len_q, seq_len_k]
    scores = scores / math.sqrt(d_k)              # scale

    if mask is not None:
        # mask out (set to a very negative value) the positions we do not want to attend to
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)   # attention weights
    output = torch.matmul(attn_weights, V)         # [batch, heads, seq_len_q, d_v]
    return output, attn_weights
```

Key points:  
- We take queries \(Q\), keys \(K\), and values \(V\).  
- Compute raw attention scores as \(QK^T\).  
- Scale by \(\sqrt{d_k}\).  
- Apply a (possible) mask (for causal or padding).  
- Normalize by softmax.  
- Multiply by \(V\) to get final outputs.  

### 7.2 Multi-Head Attention Example

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        # Learnable projection matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Project Q, K, V
        Q = self.W_q(Q)  # [batch, seq_len, d_model]
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Reshape Q, K, V to [batch, heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # Final linear layer
        output = self.out(attn_output)
        
        return output, attn_weights
```

Here, each head is basically a separate projection of Q, K, and V. We then perform the scaled dot-product attention in parallel, concatenate results, and run a final linear projection.

### 7.3 Putting It All Together (Mini-Transformer)

Below is a *very simplified* encoder block with:
1. A multi-head self-attention sub-layer, and
2. A feed-forward sub-layer, each with residual connections and layer normalization.

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention sub-layer
        attn_output, _ = self.self_attn(x, x, x, mask=mask)
        x = x + self.dropout(attn_output)  # residual connection
        x = self.norm1(x)
        
        # Feed-forward sub-layer
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)    # residual connection
        x = self.norm2(x)
        
        return x
```

A **decoder layer** would be similar, but includes:
- Masked self-attention for the decoder input,  
- “Encoder–decoder” attention to attend to the encoder output,  
- Then the feed-forward sub-layer.  

**Positional encoding** can be implemented as:

```python
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        # add position encoding
        x = x + self.pe[:, :seq_len]
        return x
```

---

## 8. Conclusion

“Attention Is All You Need” introduces a radically simpler and often more effective approach to sequence-to-sequence modeling. Key takeaways:

1. **Self-attention** handles long-range dependencies without the sequential constraints of RNNs.  
2. **Multi-head attention** lets the model look at multiple aspects of the sequence in parallel.  
3. **Parallelization** significantly reduces training time.  
4. The **Transformer** achieved state-of-the-art results in machine translation and also generalized well to other tasks like parsing.  

As the paper concludes:

> “We are excited about the future of attention-based models and plan to apply them to other tasks. … We plan to extend the Transformer to problems involving input and output modalities other than text … Making generation less sequential is another research goals of ours.”  

With these ideas, Transformers have since become a building block in many modern NLP systems, powering large models such as BERT, GPT, and numerous others. Indeed, attention turned out to be “all you need” for a wide range of tasks.  

---

**References**  
- Vaswani et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762v7](https://arxiv.org/abs/1706.03762).  
- Additional references cited within the original paper.  

**Code**  
- Official TensorFlow implementation: <https://github.com/tensorflow/tensor2tensor>  
- Many PyTorch implementations, such as <https://github.com/jadore801120/attention-is-all-you-need-pytorch>  
