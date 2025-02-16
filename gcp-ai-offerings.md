## Assignment: Exploring Ethical and Explainable AI on Google Cloud

---

### Overview and Goals

In this assignment, you will explore **Google Cloud Platform (GCP)** with a focus on **Explainable AI** and **responsible (ethical) AI** capabilities. You will learn how Google Cloud provides tools to detect bias, generate feature attributions, and incorporate safety measures in ML workflows—particularly within **Vertex AI**. By completing this assignment, you will:

1. **Gain familiarity** with GCP’s approach to ethical AI and explainability, especially via Vertex AI’s **Explainable AI** features.  
2. **Demonstrate hands-on skills** in at least one GCP explainability feature or service (e.g., Vertex AI Explainable AI, the What-If Tool, or Fairness Indicators).  
3. **Optionally** implement advanced ML explainability algorithms (e.g., **SHAP**, **LIME**, **Counterfactual Explanations**, **Integrated Gradients**, **Saliency Maps**, **PDP**, **ICE**, **Anchor Explanations**, **Permutation Feature Importance**, **Feature Ablation**, or **Surrogate Models**) in a Jupyter notebook hosted on GCP (e.g., in Vertex AI Workbench).

This assignment addresses real-world scenarios where AI models must be **interpretable, fair, and safe**, reflecting GCP’s emphasis on **responsible AI**. You will pick one or more GCP resources to show your mastery of ethical AI considerations and advanced interpretability.

---

### Part A: Understanding GCP’s Responsible AI Landscape

**Task**  
1. **Read** about Google Cloud’s responsible AI approach, focusing on:  
   - *Vertex AI Explainable AI* (feature attributions, model cards, integrated grad techniques)  
   - *Fairness and Bias Detection Tools* (What-If Tool, Fairness Indicators)  
   - *Privacy and Security* features in Vertex AI (e.g. data encryption, data governance)  
   - *Safety in Generative AI* (content filtering, RLHF, etc.)

2. **Summarize** in a short report (1–2 pages) how GCP’s AI offerings address:  
   - **Explainability** (e.g., feature attribution, integrated gradients)  
   - **Fairness and bias detection** (What-If Tool, Fairness Indicators)  
   - **Model monitoring** (detect data drift or performance drops)  
   - **Compliance** (model cards, AI principles, ISO/IEC 42001, etc.)

**Deliverables (Part A)**  
- A short PDF (or Markdown) document titled *“Google Cloud Responsible AI Landscape”* with references to relevant GCP docs/tools.

---

### Part B: Concrete Options for Hands-On Work

Below are **four** project tracks. Choose **one** track, or propose a custom approach. Each track should include **some** aspect of explainability or bias detection. 

#### **Option 1: Vertex AI Explainable AI on a Custom Model**

1. **Dataset and Model**  
   - Pick a small dataset (e.g., a classification problem in finance or retail).  
   - Upload data into **Vertex AI** and train a model (either with *Vertex AI AutoML* or using *Custom Training* in your own notebook container).

2. **Explainable AI Integration**  
   - Enable *Explainable AI* in Vertex AI.  
   - Generate **feature attributions** to see which features most influence predictions.  
   - Optionally, produce a **model card** (if using AutoML) to document model performance and responsible AI aspects.

3. **Analysis**  
   - Present screenshots or metrics showing how each feature contributed to predictions.  
   - Discuss whether you see potential bias or surprising influences.  
   - Evaluate how your model’s interpretability might guide improvements (e.g., removing irrelevant features, adjusting data).

---

#### **Option 2: Bias Detection with What-If Tool or Fairness Indicators**

1. **Import a Model**  
   - Train a classification model (e.g. using Vertex AI or local code).  
   - Deploy it to a Vertex AI Endpoint, or export it in TF model format for local usage with the *What-If Tool*.

2. **What-If Tool / Fairness Indicators**  
   - Launch the *What-If Tool* (WIT) within Vertex AI Workbench or TensorBoard.  
   - Explore model predictions for different slices of data (e.g., comparing subgroups by gender, age, or region).  
   - If you prefer, use the *Fairness Indicators* library integrated with Vertex Pipelines or in a notebook to evaluate metrics like false-positive rate across subgroups.

3. **Report**  
   - Document any detected bias or performance disparities across groups.  
   - Show screenshots of the WIT interface or your fairness metrics.  
   - If needed, attempt minor data or model modifications to reduce bias, then re-check results.

---

#### **Option 3: Model Monitoring for Explainability & Drift**

1. **Model Deployment**  
   - Train and deploy a model on Vertex AI (e.g., a regression or classification problem).  
   - Set up **continuous evaluation** or **model monitoring** in Vertex AI to watch incoming data.

2. **Simulate Drift**  
   - Introduce changes in your input data distribution (e.g., shift in a numeric feature’s range).  
   - Let Vertex AI detect data drift or anomalies.

3. **Explain**  
   - Use Vertex Explainable AI or a notebook-based method (SHAP, LIME) to see how predictions or feature importances evolve due to the drift.  
   - Summarize how Google Cloud’s built-in tools alert you and how you might adapt the model to maintain fairness and accuracy.

4. **Discussion**  
   - Reflect on the importance of continuous monitoring for ethical AI and compliance.  
   - Propose a plan to handle the drift (retraining schedule, outlier rejection, etc.).

---

#### **Option 4: Jupyter Notebook with Advanced Explainability Algorithm**

1. **Vertex AI Workbench Setup**  
   - In Vertex AI Workbench, create a JupyterLab instance. Install required Python libraries (e.g., `pip install shap lime alibi fairness-indicators`).

2. **Model Training**  
   - Train a small model (e.g., scikit-learn or TensorFlow) on a sample dataset.  
   - Optionally, deploy to a local or Vertex endpoint.

3. **Advanced Explainability**  
   - Implement **one** or more advanced explainability methods from the list:  
     - **SHAP** (Shapley additive explanations)  
     - **LIME** (Local Interpretable Model-Agnostic Explanations)  
     - **Integrated Gradients**, **Saliency Maps** (if using neural nets)  
     - **Counterfactual Explanations**, **PDP/ICE**, etc.

4. **Visualizations & Conclusion**  
   - Show charts or tables illustrating your chosen method’s output (e.g., SHAP force plot or LIME local explanations).  
   - Interpret how the model behaves, any potential biases found, or surprising features.  
   - Discuss how this approach complements or differs from GCP’s built-in Vertex Explainable AI features.

---

### Part C: Deliverables and Rubric

**Overall Deliverables**  
1. **Short Summary** (Part A) describing GCP’s responsible/ethical AI tools (1–2 pages).  
2. **Hands-on Demo** (Part B) selecting one option:
   - Provide code (notebooks/scripts) + brief instructions to replicate.  
   - Present your results (plots, screenshots, or logs) demonstrating the chosen approach.  
3. **Discussion**:
   - What you learned about model explainability or fairness.  
   - How you interpret the results.  
   - Potential improvements or next steps for an ethically sound ML pipeline.

**Rubric (Total 100 points)**

| **Criterion**                                           | **Points** | **Description**                                                                                                      |
|---------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------------------|
| **1. Understanding GCP Responsible AI (Part A)**        | 15         | Quality and clarity of summary explaining GCP’s approach to ethical AI, referencing Vertex AI Explainable AI, fairness, safety. |
| **2. Project Implementation (Part B)**                  | 25         | Proper setup and usage of the selected approach (AutoML, WIT, custom code, etc.). Effective demonstration of GCP resources.       |
| **3. Explainability Demonstration**                     | 20         | Depth of explainability techniques (either built-in Vertex AI or advanced library). Clarity and correctness of outputs/visuals.  |
| **4. Bias/Fairness Analysis**                           | 20         | Thorough analysis of bias or drift. Reflection on fairness constraints, potential improvements, or best practices.                |
| **5. Documentation & Presentation**                     | 20         | Overall organization, clear code/notebooks, well-presented results (screenshots/figures) plus a coherent final report.           |

**Bonus (+5 points)**  
- Going beyond the basics. E.g., comparing multiple algorithms (SHAP vs LIME), including real-time drift simulation, or adding human-in-the-loop review with custom workflows in Vertex AI.

---

### Submission Instructions

- **You  work individually**.
- **AWS/GCP Access**: Use a GCP free trial or an institutional GCP project if available. Keep resource usage minimal (small instances, short training).  
- **Format**: Submit a GitHub repo (or zip) with:
  - Part A summary (PDF or Markdown)  
  - Notebooks/scripts (Jupyter or Python)  
  - Screenshots of Vertex AI console (if used) showing your model and explanation results  
  - A brief reflection on what worked, what was challenging, and how you addressed ethical considerations
---

### Conclusion

This assignment will give you **hands-on practice** with **Google Cloud’s Explainable AI** offerings, bridging theory (ethical AI principles) and practical techniques (bias detection, feature attribution). By the end, you’ll be more confident using **Vertex AI** (or open-source libraries) to ensure **interpretable, fair, and responsible** machine learning solutions. Good luck exploring GCP’s AI ecosystem!
