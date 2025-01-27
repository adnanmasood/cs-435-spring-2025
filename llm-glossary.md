# LLM Glossary

Below is a collection of key terms commonly used in the realm of Large Language Models (LLMs). Each term includes an expanded definition, and where applicable, a mathematical formulation or context is provided.

---

## Adversarial Examples
**Definition**: Adversarial examples are inputs intentionally crafted to mislead an LLM, with the goal of exposing the model’s vulnerabilities. By making subtle perturbations to the input data, an attacker can cause large shifts in the model’s output or predictions.

**Mathematical Context**: Consider a model $f(\mathbf{x}; \theta)$, where $\theta$ represents the model parameters and $\mathbf{x}$ is the input. An adversarial example $\mathbf{x}_\text{adv}$ can be found by adding a small perturbation $\delta$ that maximizes a loss function $L$:

$$
\mathbf{x}_\text{adv} = \mathbf{x} + \delta 
\quad \text{where} \quad
\delta = \arg \max_{\|\delta\|\leq \epsilon} L\bigl(f(\mathbf{x} + \delta; \theta)\bigr).
$$

Here, $\epsilon$ controls the size of the perturbation.

---

## Agents
**Definition**: In the context of LLMs, an agent is an AI-driven entity—often embodied by the language model—capable of perceiving its environment, making decisions, and performing tasks or actions. This might include responding to user queries or interacting with other systems.

---

## Attention Mechanism
**Definition**: An attention mechanism helps an LLM focus on specific parts of the input sequence when generating an output. It allows the model to weigh the relevance of different tokens, enabling more efficient processing of long or complex sequences.

**Mathematical Context**: In a simplified form, given a set of queries $Q$, keys $K$, and values $V$, the attention output is:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\biggl(\frac{QK^\top}{\sqrt{d_k}}\biggr)V,
$$

where $d_k$ is the dimensionality of the keys.

---

## Autoregressive
**Definition**: An autoregressive model generates text one token at a time, conditioning each subsequent token on all previously generated tokens. This approach underlies many popular LLMs, such as the GPT series.

**Mathematical Context**: For a sequence of tokens $x_1, x_2, \dots, x_n$, an autoregressive model factorizes the joint probability as:

$$
P(x_1, x_2, \dots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, \dots, x_{t-1}).
$$

---

## Backpropagation
**Definition**: Backpropagation is the primary algorithm for training neural networks, including LLMs. It calculates gradients of the loss function with respect to each parameter, then updates parameters in the direction that reduces the loss.

**Mathematical Context**: Given a loss function $L(\theta)$, the gradient update for a parameter $\theta$ is:

$$
\theta \leftarrow \theta - \eta \frac{\partial L(\theta)}{\partial \theta},
$$

where $\eta$ is the learning rate.

---

## Beam Search
**Definition**: Beam search is a decoding strategy in which multiple candidate sequences are kept ("beam width") at each generation step. The method tracks the most likely sequences to produce a more optimal final output.

**Mathematical Context**: For each step $t$, beam search expands each sequence in the beam by all possible next tokens, keeping only the top $k$ sequences (where $k$ is the beam width) based on their cumulative log probabilities.

---

## Bias
**Definition**: In LLMs, bias refers to unintended or unfair behaviors stemming from imbalances or stereotypes in the training data. This can result in discriminatory or harmful outputs if not properly identified and mitigated.

---

## BPE (Byte Pair Encoding)
**Definition**: Byte Pair Encoding is a tokenization technique that starts from individual characters and merges the most frequent pairs of tokens to create new tokens. It is effective at handling rare or out-of-vocabulary words.

**Mathematical Context**: BPE iteratively applies merges of byte pairs until a specified vocabulary size is reached. For each merge step:
1. Identify the most frequent pair of adjacent tokens in the training corpus.
2. Merge this pair into a new token.
3. Update the frequency counts.

---

## Calibration
**Definition**: Calibration ensures that a model’s predicted probabilities align with the actual likelihood of correctness. A well-calibrated LLM will assign higher confidence to correct answers and lower confidence to incorrect ones.

---

## Common Crawl
**Definition**: Common Crawl is a massive archive of web data that is freely available. Its textual data is often used to train large-scale language models due to its breadth and diversity.

---

## Context Window
**Definition**: The context window is the amount of preceding text the model can use when predicting the next token. Larger context windows let the model draw upon more history but also require more computational resources.

---

## Cosine Similarity

**Definition**: Cosine similarity measures how similar two vectors are by computing the cosine of the angle between them. It is widely used to compare word embeddings or text representations.

**Mathematical Context**: For two vectors \(\mathbf{a}\) and \(\mathbf{b}\):

$$
\text{cosine\_similarity}(\mathbf{a}, \mathbf{b})
= \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}.
$$

---

## Cross-entropy Loss
**Definition**: Cross-entropy loss is a measure of the difference between two probability distributions, often used to train and evaluate language models. Minimizing cross-entropy typically leads to better predictive performance.

**Mathematical Context**: For a true distribution $p$ and a predicted distribution $q$, cross-entropy is defined as:

$$
H(p, q) = - \sum_{x} p(x) \log q(x).
$$

---

## Data Augmentation
**Definition**: Data augmentation involves creating additional training examples by applying transformations or manipulations to existing data. This helps increase the diversity and size of the dataset, improving model robustness.

---

## Dataset
**Definition**: A dataset is a collection of data (e.g., text, images, or other modalities) used for training or evaluating a machine learning model. For LLMs, datasets often contain millions or billions of text samples.

---

## Decoder
**Definition**: In transformer-based architectures, the decoder is the component responsible for generating output sequences. It uses the representations produced by the encoder (when present) and applies attention mechanisms over them.

---

## Embedding
**Definition**: An embedding is a dense, low-dimensional vector representation of tokens (e.g., words, subwords). These vectors capture semantic relationships between tokens, enabling models to reason about language effectively.

---

## Encoder
**Definition**: In transformer architectures, the encoder processes the input tokens and produces a set of contextualized embeddings. These embeddings are then consumed by a decoder (or another component) for downstream tasks like classification or generation.

---

## Evaluation
**Definition**: Evaluation refers to the process of measuring how well an LLM performs on a given task. Common metrics include accuracy, F1 score, perplexity, and more, depending on the application.

---

## F1 Score
**Definition**: The F1 score combines precision (the fraction of predicted positives that are truly positive) and recall (the fraction of true positives that are correctly identified). It is given by the harmonic mean of precision and recall.

**Mathematical Context**:

$$
F_1 = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}.
$$

---

## Factuality
**Definition**: Factuality describes an LLM’s ability to generate responses that are truthful and supported by evidence. Ensuring factual correctness is crucial in applications requiring reliable information.

---

## Fine-tuning
**Definition**: Fine-tuning adapts a pre-trained language model to a specific task by continuing training on a smaller, task-specific dataset. This enhances the model’s performance on the target task.

---

## Generation
**Definition**: Generation is the process by which an LLM produces text outputs (tokens or sequences) in response to a prompt, question, or other input.

---

## GPT (Generative Pre-trained Transformer)
**Definition**: GPT refers to a series of influential autoregressive language models developed by OpenAI. They are notable for their capacity to generate coherent and contextually relevant text on a wide range of topics.

---

## Hallucination
**Definition**: Hallucination occurs when an LLM produces text that is nonsensical or factually incorrect, often due to uncertainty or biases in the training data.

---

## Inference
**Definition**: Inference is the stage at which a trained LLM is used for predictions or generation. During inference, the model’s parameters are fixed, and the model processes new input to produce output.

---

## KL Divergence (Kullback-Leibler Divergence)
**Definition**: KL divergence measures the dissimilarity between two probability distributions. It is frequently used in machine learning to compare model predictions with true distributions.

**Mathematical Context**:

$$
D_{\mathrm{KL}}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}.
$$

---

## Knowledge Graph
**Definition**: A knowledge graph is a structured representation of entities and their relationships. It can provide LLMs with factual or relational context to improve accuracy in tasks like question answering.

---

## Labeling
**Definition**: Labeling is the process of annotating data with the correct answers or categories. These labels enable supervised learning methods to train models effectively.

---

## LaMDA
**Definition**: LaMDA (Language Model for Dialog Applications) is Google’s conversational language model focused on generating responses that are factual, engaging, and context-aware.

---

## Large Language Model (LLM)
**Definition**: A Large Language Model is a neural network—often transformer-based—trained on massive volumes of text. It excels at text generation, comprehension, and a variety of NLP tasks, often leveraging context and prior knowledge.

---

## Masked Language Modeling (MLM)
**Definition**: MLM is a pre-training objective where certain tokens in a text sequence are masked, and the model is trained to predict these hidden tokens. This encourages contextual learning of language.

---

## Meta-learning
**Definition**: Meta-learning, or “learning to learn,” teaches a model to quickly adapt to new tasks. Rather than just learning a single task, the model acquires a more general understanding that applies to various tasks.

---

## Model Architecture
**Definition**: Model architecture describes the design and organization of layers and operations in a neural network. For language models, the transformer architecture is currently predominant.

---

## Multi-Task Learning
**Definition**: In multi-task learning, a single model is trained simultaneously on multiple related tasks. This often improves overall performance and helps the model generalize better.

---

## Natural Language Generation (NLG)
**Definition**: NLG is the task of producing coherent, contextually appropriate text from some input data or structure. LLMs are widely used for NLG tasks like summarization or chatbots.

---

## Natural Language Processing (NLP)
**Definition**: NLP focuses on enabling computers to understand, interpret, and generate human language. This encompasses tasks like text classification, sentiment analysis, and machine translation.

---

## Natural Language Understanding (NLU)
**Definition**: NLU is a subfield of NLP concerned specifically with how machines interpret and understand language. It involves semantic parsing, intent recognition, and other tasks related to understanding meaning.

---

## Neural Network
**Definition**: A neural network is a computational model inspired by the human brain’s network of neurons. Through layers of interconnected units, it learns patterns from data for tasks like classification, translation, or generation.

---

## N-gram
**Definition**: An n-gram is a contiguous sequence of $n$ items (usually words) from a text. N-gram models were early methods of statistical language modeling before the advent of deep learning.

---

## Overfitting
**Definition**: Overfitting happens when a model learns patterns specific to the training data, including noise, rather than generalizable features. As a result, it performs poorly on unseen data.

---

## Parameter
**Definition**: Parameters are the variables of a model (e.g., weights and biases in a neural network) that are learned during training to minimize the model’s loss function.

---

## Perplexity
**Definition**: Perplexity is a metric used to evaluate the predictive power of language models. A lower perplexity indicates better performance.

**Mathematical Context**: For a model that assigns probability $p(x_1, x_2, \dots, x_N)$ to a test set of $N$ tokens,

$$
\text{Perplexity} = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 p(x_i)}.
$$

---

## Pre-training
**Definition**: Pre-training is the initial training phase where a large model is trained on a vast, general corpus of text. The model then learns general language features and knowledge, which can be transferred to more specific tasks.

---

## Prompt Engineering
**Definition**: Prompt engineering is the method of crafting and optimizing prompts to guide an LLM’s responses. Effective prompts can significantly improve the quality and relevance of generated text.

---

## Quantization
**Definition**: Quantization reduces the numerical precision of a model’s parameters (e.g., from 32-bit floating-point to 8-bit). This decreases memory usage and can speed up inference without drastically harming performance.

---

## Recall
**Definition**: Recall measures the proportion of relevant instances correctly identified by the model. It is especially important when missing a relevant instance is costly.

**Mathematical Context**:

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}.
$$

---

## Recurrent Neural Network (RNN)
**Definition**: An RNN processes input sequences one step at a time, keeping hidden states that “remember” information about previous steps. This makes RNNs well-suited for sequence data, though they are often overshadowed by transformers in modern LLMs.

---

## Reinforcement Learning
**Definition**: Reinforcement Learning (RL) trains an agent to take actions in an environment so as to maximize cumulative rewards. Though not always central to LLMs, RL techniques (e.g., RLHF) are used to refine language model outputs based on human feedback.

---

## Self-Attention
**Definition**: Self-attention is the mechanism within transformers that lets each token in a sequence focus on other tokens in the same sequence. This creates a contextual representation of the sequence without relying on recurrence.

---

## Semantic Analysis
**Definition**: Semantic analysis seeks to interpret the meaning of text by examining the relationships between words and phrases. In LLMs, this underpins tasks like question answering and topic classification.

---

## Sequence-to-Sequence (Seq2Seq) Models
**Definition**: Seq2Seq models convert an input sequence (e.g., a sentence in one language) to an output sequence (e.g., the translated sentence in another language). Common applications include translation and summarization.

---

## Supervised Learning
**Definition**: Supervised learning trains a model on labeled data, where each training example has an associated correct label. The model learns to map inputs to outputs, generalizing to unseen data.

---

## Tokenization
**Definition**: Tokenization splits text into smaller units called tokens (words, subwords, or characters). These tokens are the basic elements an LLM processes when generating or analyzing text.

---

## Transfer Learning
**Definition**: Transfer learning applies knowledge gained from one task to another, related task. By leveraging pre-trained models, significantly less data or training time is required for the new task.

---

## Transformer
**Definition**: A transformer is a neural network architecture based primarily on self-attention mechanisms. It has become the foundation for most modern LLMs due to its efficiency and ability to capture long-range dependencies in text.

---

## Unsupervised Learning
**Definition**: Unsupervised learning looks for patterns or structures in unlabeled data. Many LLM pre-training objectives, such as language modeling, fall under unsupervised learning paradigms.

---

## Vocabulary
**Definition**: The vocabulary is the set of all possible tokens an LLM can recognize and generate. Larger vocabularies offer more expressive power but can also increase model complexity.

---

## Weight
**Definition**: A weight is a parameter within a neural network layer that transforms input data. During backpropagation, weights are updated to reduce the model’s loss.

---

## Zero-Shot Learning
**Definition**: Zero-shot learning is the ability of a model to perform tasks that it was never explicitly trained on, relying on the general language or domain understanding it acquired during pre-training.

---
