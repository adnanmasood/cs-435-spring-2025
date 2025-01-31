# DeepSeek Models

DeepSeek is a family of **large language models (LLMs)** that push performance in math, code generation, logical reasoning, and general-purpose QA. They do so with techniques that optimize both **training efficiency** and **inference**:

- **DeepSeek‑V3** is a **671B parameter Mixture-of-Experts (MoE)** model that only activates **37B parameters** per token (the rest are “lying dormant” to handle different tokens, in parallel). This drastically cuts the total computation needed at run-time, compared to a 671B dense model.
- **DeepSeek‑R1** focuses on boosting **reasoning** ability purely from **reinforcement learning (RL)** signals. It shows that large models can learn advanced “chain-of-thought” style reasoning even without large supervised sets—by exploring solutions and receiving numerical rewards.

In short:

- **DeepSeek‑R1**: Focuses on *pure RL for advanced reasoning*; fosters reflection, self-checking, “Aha! moments.”
- **DeepSeek‑V3**: Focuses on *efficiency* (cheap training & fast inference) via an MoE architecture and other speedups.

---

# 2. SFT and RL in DeepSeek

DeepSeek employs a **two-phase** finishing strategy after pre-training:

1. **Supervised Fine-Tuning (SFT)**: Gather curated data or use model-generated data (with humans or automated checks). Then do standard supervised training on these curated question–answer pairs, or chain-of-thought solutions.
2. **Reinforcement Learning (RL)**: Expand the performance further by letting the model *explore multiple responses*, measure the quality of each (through a reward model or rule-based checks), and then optimize using an RL algorithm called **GRPO** (Group Relative Policy Optimization).

### 2.1 SFT 
- SFT is basically: “We have inputs (prompts) and desired outputs (answers or explanations). We do supervised gradient descent on them.”  
- This ensures the model’s “default style” is user-friendly, aligned, and more coherent.

### 2.2 RL
- RL steps in to “fine-tune from feedback” when you have tasks with *hard* or *expensive* solutions, or multiple possible solutions. 
- For example, math and coding tasks can have *deterministic correctness checks*, so a rule-based system sees if the model was correct, giving a numeric “reward” to the model. 
- Or, for more open-ended tasks (creative writing), DeepSeek can have a learned *reward model*, or it can even self-evaluate to produce a reward.

By combining SFT + RL, the resulting DeepSeek model yields strong alignment and reasoning with minimal data cost.

---

# 3. How Does DeepSeek Architecture Differ from Typical Transformers?

A standard Transformer layer has:
1. **Multi-Head Attention** over Key-Value pairs from previous tokens,
2. **Feed-Forward** sub-layer,
3. Normalization layers.

**DeepSeek** modifies two main parts:

1. **Multi-Head Latent Attention (MLA)**  
   Instead of storing Key/Value for *each head* in large matrices, it factorizes them through a smaller “latent dimension.” This drastically reduces Key/Value “cache” sizes at inference time.

2. **Mixture-of-Experts (MoE)** with *top-K routing*  
   Instead of having one large feed-forward module, the Transformer feed-forward is replaced by *many* “expert” sub-networks, each a standard feed-forward but with smaller dimension. For each token, a “router” picks which experts handle that token. This means each token only sees a *small slice* of the total model (the top-K experts). 
   - This is far more parameter-efficient: e.g., the entire model might have 256 or more experts, but each token only uses 8 of them.

Additionally, DeepSeek introduces:

3. **Multi-Token Prediction (MTP)**  
   During training, it does not only guess the “next token” but also tries to guess the 2nd token, or 3rd token, etc. This “densifies” training signals and can also be leveraged for *speculative decoding* at inference time.

4. **FP8 Quantization**  
   DeepSeek uses custom low-precision math to do **Forward/Backward** in FP8 with certain carefully designed hacks (fine-grained scaling, partial-sum accumulation, etc.). This speeds training and shrinks memory usage.

**Diagrammatically**:

```
 ┌─────────────────────────────────────────────────────────────────┐
 │   Transformer Block                                            │
 │  ┌───────────────────────────┐                                  │
 │  │        RMS Norm          │                                  │
 │  ├───────────────────────────┤                                  │
 │  │Multi-Head Latent Attn( ) │  <-- MLA is used instead of MHA   │
 │  ├───────────────────────────┤                                  │
 │  │        RMS Norm          │                                  │
 │  └───────────────────────────┘                                  │
 │  ┌───────────────────────────┐
 │  │    MoE / FFN (DeepSeekMoE)│  <-- Replaces typical feed-forward
 │  └───────────────────────────┘
 └─────────────────────────────────────────────────────────────────┘
```

---

# 4. The “Secret Sauce” for Fast & Cheaper Training

Below are the **four** major improvements that let DeepSeek train quickly yet keep high accuracy.

---

## 4.1 Mixture of Experts (MoE)

### Concept

**Mixture-of-Experts** means you have many separate feed-forward networks (called “experts”), each with its own parameters. For each token, a router chooses the top-K experts to process that token. This is more parameter-efficient because:

- With a huge dense model, all tokens go through the entire model (billions of parameters).
- With MoE, tokens only go through a small subset (top-K experts), so we effectively skip the others.  
- **Key**: The total number of parameters is large, but the *activated* (used) parameters per token are smaller, so we keep speed up.

### Code Sketch

Below is a very simplified code snippet in PyTorch-like pseudocode, showing how you might implement a single MoE feed-forward layer with top-K gating. (We do not show the distributed dispatch mechanics—just the concept.)

```python
import torch
import torch.nn.functional as F

class SimpleMoEFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, n_experts, k_r=2):
        """
        d_model: hidden dimension
        d_ff: dimension of each expert's feed-forward sub-layer
        n_experts: total number of experts
        k_r: how many experts to activate per token
        """
        super().__init__()
        self.n_experts = n_experts
        self.k_r = k_r

        # Suppose each expert is a two-layer MLP:
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_model, d_ff),
                torch.nn.ReLU(),
                torch.nn.Linear(d_ff, d_model)
            ) for _ in range(n_experts)
        ])

        # A set of "centroid" or gating vectors, dimension d_model
        self.expert_centroids = torch.nn.Parameter(
            torch.empty(n_experts, d_model).normal_(0, 0.01)
        )

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        # Let's flatten batch+seq into one dimension for gating
        B, S, D = x.shape
        x_2d = x.view(B*S, D)  # (B*S, D_model)

        # 1) Compute "affinity" to each expert: (B*S, n_experts)
        # s[i, e] = dot( x_2d[i], expert_centroids[e] )
        s = torch.einsum('id,ed->ie', x_2d, self.expert_centroids)
        
        # We can use Sigmoid or Softmax or top-K gating logic
        # Suppose we do top-K based on s:
        # For simplicity, let's get the top-K indices:
        topk_values, topk_indices = torch.topk(s, self.k_r, dim=1)
        
        # We'll create an output buffer:
        out = torch.zeros_like(x_2d)

        # For each token, we route to these top-K experts
        for i in range(x_2d.size(0)):
            for k in range(self.k_r):
                e_idx = topk_indices[i, k].item()  # which expert
                # In the real system, we'd do a distributed "dispatch."
                # But locally, we can just run the chosen expert:
                out[i] += self.experts[e_idx](x_2d[i:i+1])

            # Optionally scale out[i] by 1/k_r or topk_values[i,k], etc.

        # Return shape back: (B, S, D)
        return out.view(B, S, D)
```

This snippet highlights:

- **Router**: The top-K operation picks the “active experts.”
- A real system would dispatch tokens to remote GPUs if the experts are sharded. Also, gating values might be normalizing or applying “load-balancing” logic.

### Special Load Balancing
To prevent “everyone using the same single expert,” DeepSeek uses an **auxiliary-loss-free** strategy that adjusts a small bias for each expert to “nudge” the gating distribution to be more balanced—without forcing an extra penalty on the main loss.

---

## 4.2 Multi-Head Latent Attention (MLA)

### Concept

**MLA** compresses the *key* and *value* vectors into a smaller “latent dimension,” drastically shrinking how much we must store in the attention “KV cache.” Typically, an LLM must keep (batch_size × seq_length × hidden_dim) in memory for each layer. With MLA, we only store a compressed vector plus a small “rotary embedding” part.

### Code Sketch

Below is a toy example: Instead of building the Key matrix as `K = Wk * x`, we do:

```
c_KV = W_down * x        # compress input x => smaller dimension
K = W_upK * c_KV
V = W_upV * c_KV
```

Hence we only store `c_KV` in the cache, which is smaller.

```python
class MultiHeadLatentAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, latent_dim):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.latent_dim = latent_dim

        # Down-projection
        self.W_down = torch.nn.Linear(d_model, latent_dim, bias=False)
        # Up-projection for keys and values
        self.W_upK = torch.nn.Linear(latent_dim, d_model, bias=False)
        self.W_upV = torch.nn.Linear(latent_dim, d_model, bias=False)

        # Out-projection
        self.W_out = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, past_keys=None, past_vals=None):
        """
        x: (batch, seq, d_model)
        For simplicity, assume self-attention.
        """
        # Step 1: Compress x
        c_KV = self.W_down(x)  # (batch, seq, latent_dim)

        # Step 2: Create Key, Value
        K = self.W_upK(c_KV)   # shape: (batch, seq, d_model)
        V = self.W_upV(c_KV)

        # Optionally store K, V in "past" (cache):
        # but we only store c_KV + a small positional piece in real MLA

        # Step 3: Create Q from x ( or from compressed Q ), omitted here
        Q = x  # simplistic

        # Step 4: Standard scaled dot-product attention
        # reshape Q, K, V into multiple heads
        # We'll omit the exact reshape code for brevity
        # ...
        # attn_out = ...
        
        # Step 5: Out projection
        out = self.W_out(Q)  # (batch, seq, d_model)
        return out
```

**Why it speeds up**: 
- Smaller KV cache => less GPU memory needed => can handle longer context or bigger batch.
- If we had to store big `K, V`, it would be `num_heads × hidden_dim × seq_len`; MLA cuts it drastically.

---

## 4.3 Multi-Token Prediction (MTP)

### Concept

Instead of *only* predicting the **next** token at position `t+1`, the model *also* tries to predict the **t+2** token, **t+3** token, etc. This multi-token training objective can help the model “see further” and reinforce consistent internal representations. In practice, DeepSeek‑V3 uses MTP=1 (predict t+1 *and* t+2).

This also has a side benefit that during inference, the model might guess multiple tokens at once for “speculative decoding,” so it can generate text faster.

### Code Sketch

Below is a toy example that appends an extra head to predict t+2 from the same hidden state (plus some bridging layers). We show a single-step extension; you can repeat for t+3, etc.

```python
class MultiTokenPredictionHead(torch.nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = torch.nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden):
        # hidden: shape (B, S, D)
        logits = self.proj(hidden)  # (B, S, vocab_size)
        return logits

class TransformerWithMTP(torch.nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.transformer = MyTransformer(...)  # the main next-token part
        self.extra_head = MultiTokenPredictionHead(d_model, vocab_size)

    def forward(self, x):
        # x has shape [batch, seq_len], token ids
        hidden_main = self.transformer(x)     # normal next-token path

        # For t+2, we might combine hidden_main at position t with embedding of token t+2
        # Simplify here, just do another linear:
        hidden_extra = hidden_main  # toy version

        # Then produce next-2-token logits
        logits_mtp = self.extra_head(hidden_extra)
        return logits_mtp
```

**During training**:
- We compute cross-entropy for the standard next token, plus cross-entropy for the “secondary token” predictions, and sum them up (optionally scaled by a factor).

---

## 4.4 FP8 Quantization

### Concept

Instead of using **FP32** or **BF16** for matrix multiplications, we train in **FP8**. That means each floating number has only **8 bits** total, including exponent and mantissa. Of course, 8 bits is *very* small. The trick is to do:

1. **Fine-grained scaling** for small patches or tiles in the matrix so that the distribution doesn’t overflow.  
2. **Increased-precision accumulation** so partial sums don’t lose accuracy. 
3. Storing the “master copy” of weights in BF16 or FP32 for stability, but converting them to FP8 on-the-fly for matrix multiplications.

By doing that, you cut memory usage drastically, which reduces GPU memory overhead and can accelerate your training by close to 2×.

### Code Sketch

Here’s a schematic for “FP8 gemm” + “tile-based scale”:

```python
def quantize_fp8(tensor, tile_size=128):
    """
    We'll do a simplistic per-tile max-based scaling:
    """
    B, H = tensor.shape  # e.g. shape might be [batch_size, hidden_dim]
    # in real code, you might flatten or chunk the matrix
    out = torch.zeros_like(tensor)
    # how many tiles in the row dimension
    # We'll just illustrate row-based for simplicity
    n_tiles = (H + tile_size - 1) // tile_size

    for tile_idx in range(n_tiles):
        start = tile_idx * tile_size
        end = min(start+tile_size, H)

        # Extract tile
        tile = tensor[:, start:end]
        # Compute max value
        maxval = tile.abs().max().item()
        # scale to fit e4m3 range (roughly 15 exponent values)
        # real code has a more advanced scheme
        scale = maxval / 127.0 if maxval > 1e-9 else 1e-9

        # quant = round( x / scale ), clamp to [-128..127]
        # store as int8 plus store scale for dequant
        out[:, start:end] = (tile / scale).round().clamp_(-128, 127)
        # In real code, we also store scale in a side buffer

    return out

def gemm_fp8(A, B):
    """
    Suppose A and B are stored in quantized int8 with scaling factors.
    We'll do naive matmul in float32 for demonstration.
    """
    # In real code, this is done on the GPU with special kernels.
    # We also must do partial-sum accumulation carefully.
    # For demonstration:
    A_fp32 = dequant(A)  # read scale from side buffer
    B_fp32 = dequant(B)
    out = A_fp32 @ B_fp32
    return out
```

In practice, it’s far more intricate (e.g., “per-block scaling,” roped in with the partial-sum accumulation every 128 elements). Still, the high-level idea is:

1. **Tile or block** the matrix.  
2. **Find scale** inside each block.  
3. Convert block to int8 range with exponent bits interpreted.  
4. Perform carefully-accumulated matrix multiply in a higher precision (like 16- or 32-bit partial sums).  

DeepSeek uses a specialized method with “tile-wise quantization” on the activation dimension and “block-wise quantization” on the weight dimension, plus custom GPU kernels.

---

# 5. Summary of the Papers’ Main Concepts

Below is a more narrative summary of the big ideas from the two main DeepSeek papers:

## 5.1 “DeepSeek-V3 Technical Report”

1. **Model Architecture**  
   - **Multi-Head Latent Attention** to compress Key/Value.  
   - **DeepSeekMoE** with “auxiliary-loss-free” load balancing.  
   - **Multi-Token Prediction** for better training signals and faster decoding.  
2. **Training Efficiency**  
   - Uses **FP8** mixed-precision to speed up training and reduce memory.  
   - Overlaps pipeline parallelism and MoE all-to-all communication.  
   - Achieves stable training on a large corpus (14.8 trillion tokens) at cost ~2.7M GPU hours.  
3. **Performance**  
   - Surpasses all open-source LLMs on many tasks, including math, code, logic.  
   - Approaches or rivals closed-source GPT-4o or Claude-3.5 Sonnet on big benchmarks.  

## 5.2 “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL”

1. **Pure RL for Reasoning**  
   - Train a base LLM with no supervised data initially, using a reward for correct solutions.  
   - Emerges with long chain-of-thought, reflection, “Aha!” moments.  
2. **Challenges**  
   - The model can produce messy or multi-lingual outputs in a single response.  
3. **Cold-Start + RL**  
   - Add a small “seed” supervised set for better readability, then do RL again.  
   - Achieves SOTA math/coding capabilities, enabling strong reasoning.  
4. **Distillation**  
   - Once R1 is advanced, you can distill that reasoning skill into smaller or more “normal” models to get the best of both worlds (accuracy + neat style).

---

# 6. Putting It All Together

DeepSeek modifies the **typical Transformer** in the feed-forward sub-layer (MoE), the attention sub-layer (MLA), the training objective (MTP), and the numerical format (FP8). On top of that, it uses SFT + RL to produce a strong, “aligned,” or “reasoning-savvy” model. The synergy of these changes yields a near–state-of-the-art open-source model at a fraction of the cost.

---

# 7. Complete Mini Example

Below is a more end-to-end pseudo-code that merges (1) MoE feed-forward, (2) MLA attention, (3) MTP, and (4) partial RL. Of course, we skip many details for brevity:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLAAttn(nn.Module):
    def __init__(self, d_model, n_heads, d_compress):
        super().__init__()
        self.W_down = nn.Linear(d_model, d_compress, bias=False)
        self.W_upK = nn.Linear(d_compress, d_model, bias=False)
        self.W_upV = nn.Linear(d_compress, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        # We omit query compression for brevity

        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, x):
        B, S, D = x.shape
        # 1) compress
        cKV = self.W_down(x) # (B, S, d_compress)
        # 2) up-projection => K, V
        K = self.W_upK(cKV)
        V = self.W_upV(cKV)
        # 3) form queries
        Q = x
        # 4) split into heads (toy)
        # omitted: we do normal multi-head steps
        # 5) out-proj
        out = self.W_out(Q) # toy
        return out

class MoEFFN(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, k_r=2):
        super().__init__()
        self.n_experts = n_experts
        self.k_r = k_r
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(n_experts)
        ])
        # gating centroids
        self.expert_centroids = nn.Parameter(
            torch.randn(n_experts, d_model) * 0.01
        )

    def forward(self, x):
        B, S, D = x.shape
        x_2d = x.view(B*S, D)
        # gating
        s = torch.einsum('id,ed->ie', x_2d, self.expert_centroids)
        topk_vals, topk_idx = torch.topk(s, self.k_r, dim=1)

        out = torch.zeros_like(x_2d)
        for i in range(x_2d.size(0)):
            for k in range(self.k_r):
                e_idx = topk_idx[i,k].item()
                out[i] += self.experts[e_idx](x_2d[i:i+1])
        return out.view(B,S,D)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_experts):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MLAAttn(d_model, n_heads, d_compress=1024)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoEFFN(d_model, d_ff, n_experts)

    def forward(self, x):
        h = x + self.attn(self.norm1(x))
        h = h + self.moe(self.norm2(h))
        return h

class DeepSeekMini(nn.Module):
    def __init__(self, d_model=1024, n_heads=16, n_experts=64, n_layers=4,
                 d_ff=4096, vocab_size=32000, mtp_depth=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, n_experts)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, vocab_size, bias=False)

        # MTP module: (we do 1 extra next+2)
        self.mtp_depth = mtp_depth
        if mtp_depth >= 1:
            self.out_head2 = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)  # [B, S, d_model]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits_next = self.out_head(x)   # always next-token

        if self.mtp_depth == 0:
            return logits_next

        # Suppose we combine x with offset embedding for next+2:
        # For demonstration, do something naive:
        # Shift x by 1 and feed to out_head2
        # (Proper approach is to incorporate extra transformations)
        x_shifted = torch.roll(x, shifts=-1, dims=1)
        logits_mtp = self.out_head2(x_shifted)
        return logits_next, logits_mtp

###
# Simple usage
###

# Suppose we do a forward pass:
model = DeepSeekMini()
input_ids = torch.randint(0, 32000, (2, 10))  # B=2, S=10
outputs = model(input_ids)
if model.mtp_depth == 1:
    logits_next, logits_second = outputs
    # We compute cross_entropy for each, then combine
else:
    logits_next = outputs
```

This is highly simplified, but it gives a flavor of how all the *four big changes* can be integrated in a Transformer-like model.

---

# 8. Conclusion and Next Steps

DeepSeek shows that:

1. **Mixture-of-Experts** can cheaply scale to enormous sizes if done carefully (load balancing, custom dispatch/communication).
2. **MLA** can cut memory overhead of Key/Value caching, supporting long contexts more easily.
3. **MTP** can yield better training data efficiency, plus accelerate inference with speculative decoding.
4. **FP8** training is feasible at large scale—**671B** parameters in V3—if we handle outliers and do partial sums in higher precision.

Finally, **SFT** and **RL** shape the final behavior: 
- SFT ensures a “friendly” style and correctness on typical tasks.  
- RL incentivizes deeper chain-of-thought, advanced reasoning steps, and self-verification.

**DeepSeek** aims to push open-source LLMs closer to the capabilities that historically were only found in top-tier closed-source models. By publicly releasing these architectures and checkpoints, the team hopes others will build upon them for further breakthroughs, expansions, and applications.

> *Quotes and data in the above explanation come directly from the DeepSeek-V3 Technical Report and the DeepSeek-R1 paper.*
