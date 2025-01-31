** “Attention Is All You Need” (Vaswani et al., 2017)**  
_Attribution:_  
The paper “Attention Is All You Need” was originally published as:  
> Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,  
> Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin (2017).  
> *Attention Is All You Need*. arXiv:1706.03762v7 [cs.CL].  

Google has granted permission to reproduce tables and figures from the paper for scholarly works. The excerpts and figures below are used under that permission.  

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

----

Below is a **detailed, explanatory walkthrough** of core Transformer components—**Scaled Dot-Product Attention**, **Multi-Head Attention**, **Encoder / Decoder Blocks**, and **Positional Encodings**—using *working code* in PyTorch. The aim is to illustrate how the concepts from “Attention Is All You Need” (Vaswani et al., 2017) fit together in practice. 

> **Note**: 
> - The original paper does not provide complete training code. What follows is a **toy** implementation for demonstration purposes.  
> - At each step, we’ll reference the specific ideas and equations mentioned in the paper.  
> - This code does **not** replicate the full large-scale training (like WMT translations). Instead, it shows how each concept **works** internally.

---

# 1. Overview of the Concepts from the Paper

### Scaled Dot-Product Attention
From Section 3.2 of the paper:
> “We compute the dot products of the query with all keys, divide each by \(\sqrt{d_k}\), and apply a softmax function to obtain the weights on the values.”

Mathematically:
\[
\text{Attention}(Q,K,V) = \text{softmax}\Bigl(\frac{Q\,K^T}{\sqrt{d_k}}\Bigr)\,V
\]

### Multi-Head Attention
From Section 3.2.2:
> “Instead of performing a single attention function ... we found it beneficial to linearly project the queries, keys, and values h times with different, learned linear projections to \(d_k, d_k\), and \(d_v\) dimensions, respectively.”

### Position-Wise Feed-Forward Networks
From Section 3.3:
> “In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network...applied to each position separately and identically.”

### Positional Encoding
From Section 3.5:
> “We add ‘positional encodings’ to the input embeddings at the bottoms of the encoder and decoder stacks ... using sine and cosine functions of different frequencies.”

### Residual Connections + Layer Normalization
Each sub-layer uses a residual connection (output + input) and a layer normalization step, as indicated in the paper’s Figure 1.

---

# 2. Scaled Dot-Product Attention (Code)

Below is a PyTorch implementation of the **Scaled Dot-Product Attention** described in Equation (1) of Vaswani et al.

```python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    Computes the Scaled Dot-Product Attention:
    
    Attention(Q,K,V) = softmax((QK^T) / sqrt(d_k)) * V
    
    Parameters:
      - Q: Queries of shape [batch_size, heads, seq_len, d_k]
      - K: Keys of shape [batch_size, heads, seq_len, d_k]
      - V: Values of shape [batch_size, heads, seq_len, d_v]
      - mask: Optional mask to prevent attention to certain positions
    Returns:
      - output: The attention output, same shape as Q (but d_v dimension)
      - attention_weights: The softmax weights
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        # 1) Compute raw attention scores
        d_k = Q.size(-1)  # dimension of K
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, heads, seq_len, seq_len]
        
        # 2) Scale
        scores = scores / math.sqrt(d_k)
        
        # 3) Apply mask if given (e.g., for causal masking in decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 4) Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 5) Multiply by values
        output = torch.matmul(attention_weights, V)  # [batch, heads, seq_len, d_v]
        
        return output, attention_weights
```

**Explanation**  
1. **scores = Q × Kᵀ**: We compute the raw alignment scores.  
2. **Divide by √dₖ**: This is the “scaled” part, preventing large dot products when dₖ is large.  
3. **Masking**: If needed, we mask out future tokens (decoder) or padding positions.  
4. **Softmax**: Convert scores into probabilities over each position.  
5. **Multiply by V**: Weighted sum of values, giving the final attended context.

---

# 3. Multi-Head Attention (Code)

In Section 3.2.2, the paper introduces **Multi-Head Attention**, which runs multiple attention “heads” in parallel, each with separate learned projections.

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in Vaswani et al. (2017):
    head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    Then outputs are concatenated and projected again.

    d_model = total dimensionality
    num_heads = number of parallel heads
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Learned linear transformations for Q, K, V for all heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final linear layer after concatenating all heads
        self.fc_out = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 1) Project Q, K, V from [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        Q = self.w_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2) Perform scaled dot-product attention on each head
        attn_output, attn_weights = self.attention(Q, K, V, mask=mask)
        # attn_output shape: [batch, heads, seq_len, d_k]

        # 3) Concatenate heads back into [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # 4) Final linear + dropout
        output = self.fc_out(attn_output)
        output = self.dropout(output)

        return output, attn_weights
```

**Explanation**  
- We split the model dimension (\(d_\text{model}\)) into multiple heads (each of dimension \(d_\text{k} = d_\text{model} / h\)).  
- For each head, we do a separate scaled dot-product attention.  
- We then **concatenate** the results of all heads and apply a final linear projection.

---

# 4. Position-Wise Feed-Forward Network (Code)

From Section 3.3, each layer has a simple 2-layer MLP applied **per token** (point-wise):

\[
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
\]

```python
class PositionWiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    Applied identically to each position (token) in the sequence.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

---

# 5. Positional Encoding (Code)

From Section 3.5:
> “We add positional encodings to the input embeddings at the bottoms of the encoder and decoder stacks. … We use sine and cosine functions of different frequencies.”

```python
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encodings:
      PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))
      PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        We add the positional encoding to the token embeddings.
        """
        seq_len = x.size(1)
        # x = x + pe[:,:seq_len,:] if 3D
        x = x + self.pe[:, :seq_len].clone().detach()
        return x
```

**Explanation**  
We create a table of shape \([1, \text{max\_len}, d_\text{model}]\) containing sine/cosine signals at varying frequencies and simply add it to our token embeddings.

---

# 6. Encoder Block (Code)

Each encoder layer, as described in Figure 1 and Section 3.1:

- **Multi-Head Self-Attention**  
- **Add & LayerNorm**  
- **Position-Wise Feed-Forward**  
- **Add & LayerNorm**  

```python
class EncoderLayer(nn.Module):
    """
    One Encoder Layer:
      1) Self-attention sub-layer (with residual + layer norm)
      2) Feed-forward sub-layer (with residual + layer norm)
    """
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, mask=mask)
        out1 = self.norm1(x + self.dropout(attn_output))  # residual

        # Feed-forward
        ffn_output = self.ffn(out1)
        out2 = self.norm2(out1 + self.dropout(ffn_output))  # residual

        return out2
```

---

# 7. Decoder Block (Code)

The decoder layer has:
- **Masked Self-Attention** (so the model cannot see future tokens)  
- **Encoder–Decoder Attention** (queries from decoder states, keys and values from encoder output)  
- **Feed-Forward Network**  

```python
class DecoderLayer(nn.Module):
    """
    One Decoder Layer:
      1) Masked self-attention (with residual + layer norm)
      2) Encoder-decoder attention (with residual + layer norm)
      3) Feed-forward (with residual + layer norm)
    """
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # 1) Masked self-attention
        _x, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(_x))

        # 2) Encoder-Decoder attention
        #    Q = x (decoder states), K,V = enc_out
        _x, attn_weights = self.enc_dec_attn(x, enc_out, enc_out, mask=src_mask)
        x = self.norm2(x + self.dropout(_x))

        # 3) Feed-forward
        _x = self.ffn(x)
        x = self.norm3(x + self.dropout(_x))

        return x, attn_weights
```

**Notes**  
- **tgt_mask** typically masks out future tokens in the target sequence (a causal mask).  
- **src_mask** may mask out any padding tokens in the source, etc.

---

# 8. Putting It All Together: Mini Transformer

Let’s define a minimal **Transformer** class with:
- **Embedding + Positional Encoding** (both encoder and decoder)  
- **Stacked Encoder Layers**  
- **Stacked Decoder Layers**  
- **Final Linear + Softmax** (for language modeling/translation)

```python
class Transformer(nn.Module):
    """
    A mini-Transformer combining:
      - Embedding & PositionalEncoding for source, target
      - N-layer Encoder stack
      - N-layer Decoder stack
      - Final linear projection to vocabulary
    """
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=512, 
                 d_ff=2048, 
                 num_heads=8, 
                 num_layers=6, 
                 dropout=0.1,
                 max_len=100):
        super(Transformer, self).__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encodings
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final linear to map decoder output to vocab
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src, tgt shape: [batch_size, seq_len]

        # 1) Embed and add positional encodings
        enc_inp = self.src_embedding(src) * math.sqrt(self.d_model)
        enc_inp = self.pos_encoding(enc_inp)

        dec_inp = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        dec_inp = self.pos_encoding(dec_inp)

        # 2) Pass through N encoder layers
        enc_out = enc_inp
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, mask=src_mask)

        # 3) Pass through N decoder layers
        dec_out = dec_inp
        for layer in self.decoder_layers:
            dec_out, _ = layer(dec_out, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)

        # 4) Final projection to vocabulary
        logits = self.fc_out(dec_out)  # [batch, seq_len, tgt_vocab_size]
        return logits
```

---

# 9. Example Usage on a Toy Dataset

Below is a **mini training loop** example using random or dummy data. This is just to illustrate how the forward pass works, not to train a real translation model.

```python
def generate_square_subsequent_mask(sz):
    """
    Generates a causal mask for size sz x sz:
    Each row only attends to positions <= its index.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
    # invert mask: 1-> keep, 0-> mask out
    mask = ~mask
    return mask  # True/False mask

# Hyperparams for the toy example
src_vocab_size = 50
tgt_vocab_size = 50
max_len = 20
batch_size = 2

# Instantiate the model
model = Transformer(src_vocab_size, tgt_vocab_size,
                    d_model=64, d_ff=256, num_heads=4, num_layers=2,
                    dropout=0.1, max_len=max_len)

# Example: random input data
src_seq = torch.randint(0, src_vocab_size, (batch_size, max_len))  # [batch, seq_len]
tgt_seq = torch.randint(0, tgt_vocab_size, (batch_size, max_len))

# Create masks (causal mask for target)
src_mask = None  # Typically we might mask out padding tokens
tgt_mask = generate_square_subsequent_mask(max_len).unsqueeze(0)  # broadcast for batch if needed

# Forward pass
logits = model(src_seq, tgt_seq, src_mask=src_mask, tgt_mask=tgt_mask)
# logits shape: [batch_size, max_len, tgt_vocab_size]
print("Logits shape:", logits.shape)

# Example loss and backprop
criterion = nn.CrossEntropyLoss()
# reshape logits to (batch*seq_len, vocab_size)
logits_2d = logits.view(-1, tgt_vocab_size)
tgt_seq_1d = tgt_seq.view(-1)
loss = criterion(logits_2d, tgt_seq_1d)

loss.backward()
print("Toy training step done. Loss =", loss.item())
```

**Explanation**:
1. We create a **causal mask** (`generate_square_subsequent_mask`) so that position *i* cannot attend to positions *i+1, i+2, ...*.  
2. We embed the source and target, add positional encodings, then run through all encoder/decoder layers.  
3. The final linear layer (`fc_out`) produces logits over the target vocabulary at each time step.  
4. We compute a sample cross-entropy loss just for demonstration and run `.backward()`.

---

# 10. Relating Back to Examples from the Paper

Although this toy code won’t reproduce the **WMT 2014** results, it reflects the same logic described by Vaswani et al.:

- **Section 5.3 “Optimizer”**: The paper uses an Adam optimizer with a special learning-rate schedule that “warms up” then decays proportionally to the inverse square root of the step number. In a real system, you’d add:

  ```python
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
  
  # Pseudocode for the learning rate scheduling:
  # lr = d_model^-0.5 * min(step_num^-0.5, step_num * warmup_steps^-1.5)
  ```
  
- **Section 5.4 “Regularization”**: The code uses dropout (0.1) in attention and feed-forward sub-layers. You might also add **label smoothing** (the paper uses 0.1) to the cross-entropy loss.

- **Multi-Head Attention** exactly follows the equations from Figures 1 and 2 (Scaled Dot-Product, then repeated across multiple heads).

- **Positional Encoding**: We added sine/cosine encodings in the same manner, with frequencies scaled by \(10{,}000^{-2i/d_\text{model}}\).

- **Masked Self-Attention** in the decoder ensures auto-regressive prediction—this matches Figure 2 in the paper, where the authors mention:
  > “We implement this inside of scaled dot-product attention by masking out all values in the input of the softmax which correspond to illegal connections.”

---

## Final Remarks

1. **Core Innovation**: This entire architecture is built on the simple principle that global dependencies in sequences can be captured with **attention alone**, avoiding recurrence or convolution.  
2. **Parallelization**: Because each self-attention layer can process all tokens simultaneously (rather than step-by-step, like an RNN), Transformers scale extremely well—paving the way for large language models (GPT, BERT, T5, etc.).  
3. **Extensions**: In practice, you would:
   - Use **much larger** `d_model`, `num_layers`, `d_ff`.  
   - Employ **massive vocabularies** (30k–100k tokens).  
   - Train on **large corpora** with billions of tokens.  
   - Incorporate advanced optimizers, scheduling, and sometimes specialized GPU kernels or tensor-core hardware (TPUs).  

This detailed, **working code** demonstrates *how* the major components described in “Attention Is All You Need” fit together, step by step. The modular design reflects the original paper’s structure: **Encoder–Decoder with repeated layers of Multi-Head Attention + Feed-Forward**, plus **positional signals** to encode sequence ordering—all tied together with **residual connections** and **layer normalization**.

---

### References

- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A., Kaiser, L., & Polosukhin, I. (2017).** *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).  
- [Original Tensor2Tensor Implementation](https://github.com/tensorflow/tensor2tensor) (TensorFlow).  
- [Various PyTorch Implementations](https://github.com/jadore801120/attention-is-all-you-need-pytorch).  



**How “Attention Is All You Need” Shaped the Development of Large Language Models**

The 2017 paper “Attention Is All You Need” by Vaswani et al. introduced the **Transformer** architecture, a novel approach to sequence processing that relies entirely on **self-attention**. This innovation has been incredibly influential and now forms the backbone of most large language models (LLMs). Below are some of the key ways the paper shaped and accelerated the development of modern LLMs, along with concrete examples.

---

## 1. Moving Beyond Recurrence and Convolution

### **Pre-Transformer Era**  
- **RNNs (LSTMs, GRUs):** Used sequential processing of words/tokens, which is harder to parallelize. Models like the original seq2seq or GNMT had good performance but were slow to train on large corpora.  
- **CNNs (ConvS2S, ByteNet):** Could parallelize some computations but required deep stacks or complicated architectures to capture distant dependencies.

### **Transformer’s Breakthrough**  
- **Full Attention**:  
  > *“[The Transformer relies] entirely on an attention mechanism to draw global dependencies between input and output.”*  
- **Parallelism**: Eliminating recurrent layers unlocked the ability to train on large amounts of text with highly parallelizable matrix operations (e.g., on GPUs/TPUs).  
- **Constant Path Length**: Any token can attend to any other token directly (in a single attention step), helping the model capture long-range relationships more effectively.

**Influence on LLMs**  
- **Scalability**: The ease of parallelization meant that when researchers began to scale models to hundreds of billions of parameters (e.g., GPT-3), the Transformer was naturally suited to handle that growth.  
- **Training Efficiency**: Training time for extremely large datasets (like Common Crawl) became feasible.

---

## 2. Core Building Block of Popular LLM Architectures

After the Transformer’s success in machine translation, researchers quickly realized that the *multi-head self-attention* mechanism could serve as the universal backbone for many NLP tasks. Here are major LLM families that directly inherited the Transformer architecture:

1. **GPT (Generative Pre-Trained Transformer) Series**  
   - **GPT-1** and **GPT-2**: Used the **decoder** portion of the Transformer to auto-regressively predict the next token.  
   - **GPT-3**: Scaled the decoder-only Transformer architecture to 175 billion parameters, showing strong zero-shot and few-shot capabilities.  
   - **ChatGPT / GPT-4**: Further refinements, same fundamental decoder-based Transformer.  

2. **BERT (Bidirectional Encoder Representations from Transformers)**  
   - Adapted the **encoder** part of the Transformer for masked language modeling.  
   - Achieved state-of-the-art results on various NLP benchmarks, launching the “BERTology” era.  

3. **T5 (Text-to-Text Transfer Transformer)**  
   - Uses a **full encoder–decoder** Transformer architecture.  
   - Unified NLP tasks by converting them all into a “text-to-text” format, scaling up training.  

4. **Other Examples**  
   - **XLNet**: Hybrid between auto-regressive and auto-encoding objectives, still uses Transformer layers.  
   - **ALBERT, DistilBERT**: Parameter-efficiency modifications of the base Transformer/BERT idea.  
   - **RoBERTa**: Enhanced BERT with more training data and slightly different hyperparameters, same core architecture.

**Key Point**: All these models share the core multi-head attention, feed-forward sub-layers, and positional encodings (or learned positional embeddings), directly inheriting from “Attention Is All You Need.”

---

## 3. Attention as a Universal Mechanism

### Direct Influence
The original paper emphasized that a simple, attention-only approach can handle **long-range dependencies** while being highly parallelizable. LLMs benefit enormously from these properties, because:

- **Long Context Windows**: Modern LLMs handle thousands (sometimes tens of thousands) of tokens of context. The self-attention mechanism scales more gracefully than RNNs or CNNs in capturing cross-token relationships.  
- **Interpretability**: Attention weights can offer insights into which parts of the input the model focuses on (though this remains an area of active research).

### Examples of Innovations Built on the Transformer’s Self-Attention
- **Sparse or Local Attention** (e.g., Longformer, BigBird): Adapt the Transformer’s attention to handle even longer contexts (e.g., full books, very long documents) by selectively attending to a subset of tokens.  
- **Mixture-of-Experts (MoE) Transformers**: Incorporate large “expert” feed-forward blocks to scale parameters effectively.

---

## 4. Scaling Laws and Emergent Abilities

### The “Bigger is Better” Paradigm
Once the Transformer architecture had proven to be efficient and effective, the NLP community began to scale the models in terms of:
- **Number of layers**  
- **Hidden dimensions**  
- **Heads in multi-head attention**  
- **Training data**

Research from OpenAI, DeepMind, and others found that scaling Transformers leads to near “power-law” improvements in perplexity and downstream performance.

### Emergent Capabilities
- **GPT-3** and subsequent large Transformers exhibit *in-context learning*, few-shot reasoning, chain-of-thought prompting, etc.—phenomena that were not as visible or robust in smaller models.  
- This shift is partly due to how the self-attention mechanism can integrate broad contextual signals across large text corpora.

---

## 5. Practical Examples of Impact

1. **Prompt Engineering**  
   - Modern LLMs (e.g., ChatGPT) rely on user prompts. The self-attention layers take the entire conversation history as input. The model’s ability to condition on thousands of preceding tokens of text demonstrates the direct advantage of the Transformer’s parallel attention.  

2. **Multi-Modal Extensions**  
   - Transformers have also been adapted for images (Vision Transformer, ViT), audio (Audio Transformers), and multi-modal tasks (e.g., Flamingo, a visual-language model).  
   - The original paper hinted:  
     > *“We plan to extend the Transformer to problems involving input and output modalities other than text.”*  

3. **Rapid Fine-Tuning and Domain Adaptation**  
   - The standardized architecture (encoder-only, decoder-only, or encoder–decoder) makes it straightforward to fine-tune pre-trained Transformers on new tasks with minimal changes.  

---

## 6. Summary

**In short:**  
- “Attention Is All You Need” introduced the **Transformer**, showing that a fully attention-based model could outperform RNN/CNN hybrids on sequence tasks, especially machine translation.  
- Its emphasis on **multi-head self-attention**, **parallelization**, and **scalability** directly set the stage for large language models like GPT, BERT, and T5.  
- Subsequent LLMs have only reinforced these concepts, demonstrating remarkable emergent abilities and establishing the Transformer as the de facto standard for NLP (and beyond).

Thus, the original paper continues to be the **foundational reference** for nearly all modern large language models, influencing not just architecture design, but also how we think about **scaling**, **context usage**, and **transfer learning** in natural language processing.
