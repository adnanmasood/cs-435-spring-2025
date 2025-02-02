## **Assignment: Exploring Guardrails AI Validators**

Guardrails AI Validator Hub
https://hub.guardrailsai.com/
Guardrails AI github codebase
https://github.com/guardrails-ai/guardrails


### **Objective**

1. **Learn & Explore**: Familiarize yourself with the Guardrails AI framework and the variety of **validators** offered in the Guardrails Hub.  
2. **Implement a Validator**: Install and use **one validator** of your choice in a Jupyter notebook to demonstrate how it works.  
3. **Document & Demonstrate**: Provide detailed explanations (via comments and markdown cells) of what your code does and why it is relevant to AI/LLM risk mitigation.  

---

### **Background**

Guardrails AI is a Python framework designed to add **reliability and safety** features to AI applications, particularly those leveraging Large Language Models (LLMs). The [Guardrails Hub](https://www.guardrailsai.com/docs) offers **pre-built validators** that can detect, quantify, and mitigate various types of risks (for instance, **prompt injection**, **PII leakage**, **profanity**, etc.).

---

### **Part 1: Setup**

1. **Installation**  
   - Make sure you have a **Python 3.8+ environment** (e.g., via Anaconda or venv).  
   - Install the core library:  
     ```bash
     pip install guardrails-ai
     ```
2. **Research Validators**  
   - Visit the [Guardrails Hub](https://guardrailsai.com/hub) or the official [GitHub repository](https://github.com/guardrails-ai/guardrails) to review the **list of available validators**.  
   - Identify **one validator** that interests you or is relevant to your project or area of study (e.g., **Detect Prompt Injection**, **Detect Secrets**, **Profanity Free**, **NSFW Text**, etc.).  

---

### **Part 2: Implementation in Jupyter Notebook**

1. **Notebook Setup**  
   - Create (or open) a new **Jupyter Notebook**.  
   - Title it clearly (e.g., `GuardrailsAI_ValidatorDemo.ipynb`).  
   - Start with a brief overview (in a **Markdown cell**) describing what the notebook aims to achieve.

2. **Install/Import Your Chosen Validator**  
   - Many validators are included in the guardrails-ai package by default. Some may require installing from the Hub. For example:  
     ```bash
     guardrails hub install hub://guardrails/profanity_free
     ```
   - In your notebook, **import** the necessary modules, e.g.:
     ```python
     import guardrails
     from guardrails import Guard, OnFailAction
     from guardrails.hub import ProfanityFree  # as an example
     ```
   - Provide short explanatory comments about each import.

3. **Initialize the Validator**  
   - Demonstrate how to **instantiate a Guard** and **add** your chosen validator to it. For example, using ProfanityFree:
     ```python
     guard = Guard().use(
         ProfanityFree(
             threshold=0.5,                # adjust threshold if desired
             validation_method="sentence", # e.g., check each sentence
             on_fail=OnFailAction.EXCEPTION
         )
     )
     ```
   - In **comments**, explain what each parameter means.

4. **Validation Demonstration**  
   - **Show** how the validator works by passing **sample text**. Include at least two examples:
     - **Passing case**: Text that should pass the validator.  
     - **Failing case**: Text that triggers the validator.  
   - Example:
     ```python
     # 1. Text that should pass
     text_pass = "Hello, this is a polite message."
     try:
         validated_pass = guard.validate(text_pass)
         print("Passed Validation:", validated_pass)
     except Exception as e:
         print("Error:", e)

     # 2. Text that should fail
     text_fail = "Shut the hell up!"
     try:
         validated_fail = guard.validate(text_fail)
         print("Passed Validation:", validated_fail)
     except Exception as e:
         print("Validation Failed:", e)
     ```
   - Insert **Markdown cells** explaining what you expect to happen in each case, and comment on the actual output.

5. **(Optional) Additional Exploration**  
   - If your validator allows extra parameters or advanced usage, demonstrate that as well.  
   - Discuss **real-world use cases** for your chosen validator (e.g., moderate user-generated content, filter sensitive data, comply with privacy rules, etc.).

---

### **Part 3: Documentation & Explanation**

- In **Markdown cells**, clearly document:  
  1. **Why** you chose this particular validator (technical or practical reasons).  
  2. **How** the validator addresses a specific risk (e.g., preventing the spread of offensive content, avoiding data leakage, etc.).  
  3. **Any** references to official Guardrails documentation if relevant.

---

### **Final Deliverables**

1. **Jupyter Notebook** (`.ipynb` file) containing:  
   - Installation and import steps.  
   - Validator choice and rationale.  
   - Code examples (pass/fail scenarios).  
   - Commentary on the results.  
2. **Brief Write-Up** *(within the notebook or as a short PDF)* explaining the real-world importance of such validators for AI/LLM applications.

---

## **Grading Rubric**

| **Criteria**                                 | **Exemplary (A)**                                                                                                                                     | **Proficient (B)**                                                                                                                              | **Developing (C)**                                                                                                                            | **Needs Improvement (D/F)**                                                                                                                    | **Weight** |
|:-------------------------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------ |:---------------------------------------------------------------------------------------------------------------------------------------------- |:-----------------------------------------------------------------------------------------------------------------------------------------------|----------:|
| **1. Validator Selection & Justification**   | Clearly identifies a relevant validator and gives *strong, clear reasons* for selection (technical or use-case driven).                                 | Validator is chosen and justified, but reasoning is somewhat general.                                                                            | Chosen validator is loosely relevant, with unclear or minimal explanation.                                                                    | Validator not relevant to the assignment’s AI/risk mitigation theme; or no explanation provided.                                              | 15%       |
| **2. Technical Implementation**              | Code is *well-structured*, commented, and follows best practices; shows *full usage* of validator parameters.                                          | Code is functional and reasonably organized; some parameters are explained.                                                                     | Code runs but is somewhat disorganized or only minimally demonstrates the validator’s features.                                               | Code has errors or does not actually demonstrate the validator in a meaningful way.                                                           | 30%       |
| **3. Demonstration of Validation**           | Provides *multiple test cases* (pass/fail) with clear commentary on expected vs. actual outcomes.                                                      | At least two test cases (pass/fail) are included, with brief commentary.                                                                        | One test case or minimal discussion of outcomes.                                                                                              | No demonstration or commentary on validator results; purely boilerplate code.                                                                 | 25%       |
| **4. Understanding of Guardrails Concepts**  | Thoroughly explains how the validator addresses a *specific AI/LLM risk*, referencing possible real-world applications and limitations.                 | Provides a basic explanation of how the validator reduces risk; some real-world relevance is mentioned.                                        | Explanation is vague or too brief, missing clear links to real-world AI/LLM risk context.                                                     | No mention of AI/LLM risk or practical application.                                                                                           | 20%       |
| **5. Clarity & Documentation**               | Notebook is *well-documented*, logically organized, and includes **helpful Markdown** descriptions or references.                                      | Notebook is adequately documented; structure is clear; some references to official docs or additional resources.                                | Notebook has limited documentation/markdown, making it somewhat hard to follow.                                                               | Notebook is unstructured; code and text are unclear; lacks meaningful commentary.                                                             | 10%       |

---

### **Submission Requirements**

1. **Deadline**: (Specify your due date/time here)  
2. **Upload**: Submit the `.ipynb` file (and any optional write-up) to your course portal or repository as instructed.  
3. **Presentation** *(Optional)*: You may be asked to briefly demo or explain your notebook in class (depending on the instructor’s preference).

---

## **Sample Code Snippet**  
*(For demonstration only; students must adapt or choose a different validator.)*

```python
# Jupyter Notebook Cell

# 1. Imports
import guardrails
from guardrails import Guard, OnFailAction
from guardrails.hub import ProfanityFree

# 2. Create a Guard and attach the chosen validator
guard = Guard().use(
    ProfanityFree(
        threshold=0.5,                # can tweak threshold
        validation_method="sentence", # check sentence-by-sentence
        on_fail=OnFailAction.EXCEPTION
    )
)

# 3. Demonstration - Passing Case
text_pass = "Hello, this is a polite message."
try:
    result_pass = guard.validate(text_pass)
    print("Passed Validation:", result_pass)
except Exception as e:
    print("Validation Failed:", e)

# 4. Demonstration - Failing Case
text_fail = "Shut the hell up!"
try:
    result_fail = guard.validate(text_fail)
    print("Passed Validation:", result_fail)
except Exception as e:
    print("Validation Failed:", e)
```

*In your assignment, include explanatory text and additional test cases or parameters if relevant.*

---

### **Outcome**

By completing this assignment, you will:

- **Gain familiarity** with the Guardrails AI framework.  
- Understand **how validators mitigate risks** in AI/LLM deployments.  
- **Practice** integrating an external Python package into a development workflow.  

Good luck, and have fun exploring Guardrails AI!
