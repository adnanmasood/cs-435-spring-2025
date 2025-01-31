# On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?
by Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 

---

# 1. Historical Significance of the Paper

In recent years (roughly since 2018), the field of Natural Language Processing (NLP) has seen the rapid emergence of extremely large language models (LMs)—think of BERT, GPT-2, GPT-3, and more recently GPT-4 or PaLM-like systems. The paper *On the Dangers of Stochastic Parrots* (often referred to simply as the “Stochastic Parrots paper”) is historically significant because it:

1. **Sparked Broad Ethical Conversations**: It played a key role in bringing discussions about the ethical, social, and environmental implications of massive language models to the forefront of AI research. It highlighted how bigger is not always better, especially when it comes to potential harms.

2. **Emphasized Environmental and Social Costs**: Although many prior publications had hinted at biases in data or the carbon footprint of AI models, this paper centralizes these issues, connecting them explicitly to the design choices of large-scale language models.

3. **Helped Shift Industry and Academic Practices**: After its publication (and the ensuing controversies around large organizations and ethical AI teams), many institutions began paying more attention to energy consumption, data documentation (e.g., “datasheets for datasets”), and auditing language models for bias.

4. **Coined the “Stochastic Parrots” Metaphor**: The phrase “stochastic parrot” encapsulates the idea that these large models do not truly “understand” language; they cleverly stitch together sequences of text that *look* coherent because they have memorized huge amounts of form-based patterns. The paper’s insight that a language model can produce fluent text without any grounding in meaning is both a caution and a reminder of what these systems can (and cannot) do.

---

# 2. Overview and Context

The paper begins by pointing out the trend toward ever-larger language models—GPT-3 has 175 billion parameters, Switch-C soared to 1.6 trillion parameters—and highlights the variety of “costs” associated with such large-scale development.

> **Excerpt**:  
> “Using these pretrained models and the methodology of fine-tuning them for specific tasks, researchers have extended the state of the art on a wide array of tasks as measured by leaderboards on specific benchmarks. In this paper, we take a step back and ask: How big is too big?”

Link: https://dl.acm.org/doi/pdf/10.1145/3442188.3445922

They then outline four main risk areas:

1. **Environmental Costs**  
2. **Financial Costs and Accessibility**  
3. **Documentation and Dataset Biases**  
4. **Misleading Gains and Misinterpretations of True Language Understanding**

---

## 2.1 ELI5 Version of the Paper’s Premise

**Imagine** you have a giant parrot that has heard everything anyone has ever said on the internet. That parrot can say *tons* of sentences that *sound* like it knows exactly what it’s talking about. But it’s really just repeating and remixing patterns of words it has picked up—and it could repeat some pretty mean, harmful, or misleading stuff it heard. Also, training that giant parrot cost a lot of money and energy. *Is that parrot truly “smart” or just an echo of us (and our biases)?*

---

# 3. Environmental and Financial Costs (Sections 2–3 in the Paper)

The authors cite work showing that training large deep learning models can emit as much carbon as **trans-American flights** or even more—sometimes hundreds of times more. This has two main consequences:

1. **Climate Impact**: The communities most harmed by climate change tend not to be the ones benefiting from advanced English-language models.
2. **Resource Barriers**: Because training these giant models is so expensive, well-funded tech companies have a disproportionate advantage over academic labs or smaller companies. This can worsen inequalities (only a few big players can afford the largest models).

> **Excerpt**:  
> “Increasing the environmental and financial costs of these models doubly punishes marginalized communities that are least likely to benefit from the progress achieved by large LMs…”

### Key Point
They call on researchers to **report training time, carbon footprint, and monetary costs** to emphasize efficiency and to consider carefully before deciding to push for “bigger is better.”

---

## 3.1 ELI5 for Environmental Costs

- **Think**: *When you leave the lights on all day and night, you waste electricity.* Training a huge language model is like leaving **millions** of lights on for **days** (often weeks or months). This uses up a lot of electricity. That electricity often comes from power plants that burn fuel and produce pollution.

---

# 4. Dataset Size, Curation, and Bias (Section 4 in the Paper)

Massive datasets (think hundreds of gigabytes to terabytes of text) are typically pulled from the general internet (like Common Crawl). But:

- **Who posts online?** Not everyone. The dataset can overrepresent younger men from wealthier countries (simply because they post more).
- **Overrepresentation of Certain Viewpoints**: The majority/dominant viewpoint can appear so often that it drowns out minority or marginalized voices.
- **Filtering**: Often, documents are filtered by removing “bad words,” but that can accidentally exclude important LGBT+ expressions or discussions about marginalized identities (because some identity terms appear on “bad word” lists).
- **Bias**: With no thorough curation, the text will contain hateful or discriminatory material, which the model absorbs.

> **Excerpt**:  
> “As argued by Bender and Koller [14], it is important to understand the limitations of LMs and put their success in context. […] Focusing on state-of-the-art results on leaderboards without encouraging deeper understanding of the mechanism can cause misleading results […] and direct resources away from efforts that would facilitate long-term progress.”

**Takeaway**: The paper urges **significant investment** in dataset curation and thorough documentation, so that we know exactly what went into the training data. They highlight frameworks such as “datasheets for datasets” or “data statements,” which systematically list how the data was collected, from whom, for what purposes, and with what licenses, etc.

---

## 4.1 ELI5 for Dataset Bias

**Imagine** you want to learn to speak by listening to crowds in a big city. Except you only listen to people in one rich neighborhood that has a certain worldview. That’s your entire perspective on the city. Clearly, you are missing a lot of other voices. The same happens with large text scraped from just certain corners of the internet.

---

# 5. The Illusion of “Understanding” (Sections 5–6)

Here, the paper introduces the key metaphor: **Stochastic Parrots**. A large language model “seems” to write coherent text. Humans read it and think, “Wow, the machine must understand me.” But that’s because we humans are wired to interpret any fluent text as meaningful.

> **Excerpt**:  
> “Contrary to how it may seem when we observe its output, an LM is a system for haphazardly stitching together sequences of linguistic forms it has observed in its vast training data […] but without any reference to meaning: a stochastic parrot.”

The authors point to:
1. People can be **misled** into thinking an LLM is much smarter or more “sentient” than it really is.
2. The text may contain hidden biases or hateful content that is then repeated or even amplified.
3. **Automation bias**: Users often trust the model’s outputs too easily.
4. **Extremist or malicious** usage: automatically generating text for propaganda or harassment.

---

## 5.1 ELI5 for “Stochastic Parrot”

A “stochastic parrot” is basically a big parrot that randomly picks from everything it’s heard before and glues that together into new sentences. It can sound real (just like a parrot can mimic your voice!). But it does not *know* what it’s saying.

---

# 6. Potential Harms and Real-World Risks (Section 6)

The paper enumerates how these illusions can cause serious harm:

1. **Bias and Harmful Stereotypes**: Models pick up sexist, racist, ableist language, or stereotypes from the web. They can replicate or **amplify** them, further harming marginalized communities.
2. **Misuse**: People can deliberately use LMs to create large amounts of hateful or misleading text (e.g., extremist recruitment).
3. **Lack of Accountability**: The model’s output can appear as though a “real human” stands behind it. But in fact, nobody is truly accountable for that text’s meaning or truthfulness.

---

## 6.1 Illustrative Code Snippet: Simple Bias Detection in Generated Text

Below is a short example (using Python and Hugging Face’s Transformers library) of how one might *start* to detect certain biases or toxicity in generated text. This is *not* from the original paper (the paper itself does not provide code), but serves as an illustration of how one could explore or flag negative content.

```python
!pip install transformers torch perspective-api-client

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from perspective_api_client import PerspectiveApiClient

# We'll use a smaller model as an illustration:
MODEL_NAME = "gpt2"  # Or "EleutherAI/gpt-neo-125M" etc.

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = "Women cannot "
inputs = tokenizer(prompt, return_tensors="pt")
sample_outputs = model.generate(**inputs, 
                                max_length=50, 
                                num_return_sequences=3,
                                do_sample=True, 
                                top_p=0.9)

# Now we'll use Perspective API (a third-party tool from Google) 
# to get a quick "toxicity" or "bias" rating. 
# NOTE: Must have an API key to do this, so this code is illustrative only.

p_client = PerspectiveApiClient("<YOUR_PERSPECTIVE_API_KEY>")

for i, sample_output in enumerate(sample_outputs):
    generated_text = tokenizer.decode(sample_output, skip_special_tokens=True)
    print(f"\n--- Generated text {i+1} ---\n{generated_text}")

    # Send the text to Perspective for analysis:
    analyze_request = p_client.analyze(generated_text, 
                                       requested_attributes={"TOXICITY": {}})
    # Extract toxicity score
    toxicity_score = analyze_request["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
    print(f"\nToxicity Score: {toxicity_score:.2f}")

```

### Explanation
1. **Model Loading**: We load a GPT-2 model (much smaller than GPT-3).
2. **Prompting**: We prompt it with “Women cannot …” to see how it completes the sentence.
3. **Perspective API**: We pass the text to a basic online API that returns a “toxicity” score. Notice that ironically the Perspective API can itself be biased—it’s not foolproof. But it *illustrates* an approach to at least start measuring the negativity or harmful stereotypes in text.

---

# 7. Paths Forward and Recommendations (Section 7)

Rather than simply saying “don’t build big models,” the paper offers recommendations:

1. **Better Data Curation and Documentation**:  
   - Invest resources in high-quality datasets *specific* to the task.  
   - Document them (e.g., datasheets, data statements).  
   - Acknowledge and plan for ongoing changes in social norms and language.

2. **Pre-Mortem Analyses**:  
   - Before building a huge model, carry out “pre-mortems” where the team imagines how things could go horribly wrong—and how to avoid that.

3. **Value Sensitive Design**:  
   - Engage stakeholders who might be harmed.  
   - Identify *who* is impacted by an NLP system (including those who don’t directly use it but are indirectly affected).

4. **Consider Efficiency as a Metric**:  
   - Evaluate the *energy consumption* and *carbon footprint* of training.  
   - Optimize for better performance *per watt* or *per dollar*, not just raw accuracy.

5. **Beware the Dual-Use Problem**:  
   - Even if large LMs can do something beneficial (like improved speech recognition for deaf and hard-of-hearing communities), they also risk generating large-scale disinformation.

> **Excerpt**:  
> “We advocate for research that centers the people who stand to be adversely affected by the resulting technology, with a broad view on the possible ways that technology can affect people.”

---

## 7.1 ELI5 Summary of the Recommendations

**Don’t build a giant parrot for no reason.** If you *are* going to build one, *figure out* how to feed it carefully, keep track of what it has learned, and do “what could go wrong?” checks with the people who might be hurt if that parrot says something mean or untrue.

---

# 8. Concrete Illustrative Code for Carbon Monitoring

Below is another small code snippet showing how you might integrate a library like “codecarbon” to track CO₂ emissions while training a model. Again, this snippet is *illustrative*, not from the original paper.

```python
!pip install transformers torch codecarbon

import os
from codecarbon import EmissionsTracker
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Example dataset and model
dataset = load_dataset("imdb")
model_name = "distilbert-base-uncased"  # a smaller model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./test_trainer",
    evaluation_strategy="epoch",
    logging_steps=10,
    num_train_epochs=1,
    per_device_train_batch_size=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42).select(range(2000)),  # just a small subset
    eval_dataset=dataset["test"].shuffle(seed=42).select(range(500))
)

tracker = EmissionsTracker(project_name="DistilBERT-imdb")
tracker.start()
trainer.train()
emissions = tracker.stop()

print(f"Training emissions (kg CO2 eq): {emissions:.4f}")
```

### Explanation
1. **codecarbon**: A Python package that estimates the carbon footprint of your training run.
2. **Trainer**: We do a tiny training run on DistilBERT for demonstration.
3. **After** the training, we see approximate kg of CO2 used. Obviously, for huge language models with multi-week training times, the footprint becomes massive.

---

# 9. Conclusion

“*On the Dangers of Stochastic Parrots*” is a major milestone in NLP ethics, urging us to question the race toward bigger and bigger language models. It highlights:

- **Environmental and Financial Costs**: The massive power consumption and high financial barrier limit who can participate in cutting-edge NLP research.
- **Bias in Training Data**: Large models can internalize and amplify harmful biases present in uncurated web data.
- **The “Stochastic Parrot” Problem**: These models produce text that humans may interpret as meaningful, but it lacks genuine understanding and can easily include misinformation or prejudice.
- **Accountability and Stakeholders**: We need more systematic ways to consider who is affected by these models, how to measure harm, and how to mitigate negative outcomes.

By shifting resources and attention to carefully curated data, a deeper understanding of language tasks, and inclusive, transparent research practices, the authors believe the community can still make progress in language technology without incurring unnecessary environmental or social harms. That includes focusing on ethical governance, measurement of real-world impacts, and learning to weigh the potential benefits against the potential harms.

---

## Final ELI5 Wrap-Up

**Imagine** you have a magical copying machine that can copy and recombine every phrase it has ever seen on the internet. It doesn’t understand what it’s copying; it just knows certain words tend to come after certain other words. If you rely on it to speak for you in the world, you might spread ideas and biases you don’t agree with—or you might make mistakes that hurt people. Also, running that giant copying machine costs a *lot* of money and power. The paper basically says: *Before building bigger copying machines, let’s slow down, think carefully about the costs, watch out for dangerous or unkind phrases it might produce, and consider smaller or more thoughtful ways to create helpful language tools.*

---

# Further Reading & Resources

- **“Data Statements for NLP” (Bender & Friedman, 2018)**: Detailed guide on how to document data properly.
- **“Datasheets for Datasets” (Gebru et al.)**: A framework for thorough dataset documentation.
- **codecarbon GitHub**: <https://github.com/mlco2/codecarbon> for carbon tracking in ML training.
- **Hugging Face Transformers**: <https://github.com/huggingface/transformers> for easy experimentation with model training and generation.

---

### Acknowledgments

The original paper was authored by Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. The summary here is an educational restatement to help you understand its arguments, significance, and context. Any code examples provided are **illustrative** and not from the original publication.
