## Assignment: Exploring Ethical and Explainable AI on Microsoft Azure

### Overview and Learning Goals

In this assignment, you will delve into **Microsoft Azure’s Responsible AI and Explainable AI** offerings. By the end, you should:

1. **Understand** how Azure implements ethical/Responsible AI principles, including fairness, interpretability, content safety, and governance.  
2. **Demonstrate proficiency** in at least one **Azure-based explainability** method (e.g., using the Responsible AI dashboard in Azure Machine Learning, or a custom approach in a Jupyter notebook on Azure).  
3. **Explore** how Azure’s platform can help developers build and deploy **trustworthy AI solutions** (covering bias detection, model explanations, content filtering, and more).

You will choose **one** of the project tracks below to produce a **hands-on demonstration**. The assignment aims to reinforce practical skills in **Azure ML** (or other Azure AI services) while keeping ethical AI practices front-and-center.

---

### Part A: The Landscape of Responsible AI on Azure

**Task**  
1. **Review** Azure’s Responsible AI documentation and features, focusing on:  
   - Microsoft’s Responsible AI Principles (Fairness, Reliability & Safety, Privacy & Security, Inclusiveness, Transparency, Accountability).  
   - **Azure Machine Learning’s** **Responsible AI Dashboard** (Fairlearn, InterpretML, Error Analysis, Counterfactuals, Causal Analysis).  
   - **Azure AI Content Safety** (filtering and moderation of text or images).  
   - **Limited Access** features (e.g., Face API, Custom Neural Voice) and why Microsoft imposes usage gating.  
   - Data privacy options (storing data in your region, encryption, compliance certificates like HIPAA, GDPR).  

2. **Summarize** your findings in a short PDF/Markdown (1–2 pages), describing how Azure ensures:
   - **Bias detection and mitigation** (Fairlearn, sampling for subgroups).  
   - **Model interpretability** (SHAP-based feature attributions, local vs. global explanations).  
   - **Human oversight** in high-stakes scenarios (content safety, gating sensitive services).  
   - **Ongoing monitoring** (Error Analysis, model drift detection).  

**Deliverable (Part A)**  
- A concise report titled *“Responsible AI and Explainability in Azure: Overview”* with references to official docs. 

---

### Part B: Concrete Hands-On Options

Choose **one** of the following project tracks. Each track requires you to **create or use an ML model** (or generative AI approach) on Azure and apply **at least one** responsible AI or explainability feature.

#### **Option 1: Azure ML Responsible AI Dashboard on a Custom Model**

1. **Dataset and Model**  
   - Select a publicly available dataset (e.g., a classification problem like loan approvals or a regression dataset).  
   - Import data into **Azure Machine Learning**. Train a model using either the Python SDK or **AutoML**.

2. **Responsible AI Dashboard Integration**  
   - Enable and configure the **Responsible AI Dashboard** in Azure ML.  
   - Generate the following analyses:  
     - **Fairlearn** (check if performance differs across subgroups, e.g., by gender or region).  
     - **InterpretML** (feature importance, SHAP plots, local explanations).  
     - **Error Analysis** (explore where the model makes errors).  
     - (Optional) **Counterfactual Analysis** or **Causal Analysis** if relevant to your scenario.

3. **Output & Discussion**  
   - Include screenshots of the RAI Dashboard.  
   - Summarize the insights: any potential bias? Which features most influence predictions?  
   - Reflect on how these results might inform model improvements or data collection changes.

---

#### **Option 2: Using Fairlearn or InterpretML Manually in a Jupyter Notebook**

1. **Notebook Setup**  
   - Create an **Azure ML Compute Instance** (or an Azure DSVM).  
   - Install **fairlearn** and **interpret** packages (or use built-in Python environment that has them).

2. **Model Training & Explainability**  
   - Train a scikit-learn (or PyTorch/LightGBM) model on your dataset.  
   - Use **fairlearn.metrics.MetricFrame** to assess performance by subgroups.  
   - Use **interpret** (SHAP or Mimic Explainer) to derive local and global explanations.  
   - Optionally, visualize partial dependence plots or ICE (Individual Conditional Expectation) to see how features affect predictions.

3. **Report**  
   - Show the code, SHAP plots, fairness metrics, and interpret your findings.  
   - Suggest how to mitigate any discovered bias or how to refine your model based on the explanations.

---

#### **Option 3: Azure OpenAI + Content Moderation Workflow**

1. **Provision Azure OpenAI**  
   - Request or use existing access to **Azure OpenAI Service**.  
   - Deploy at least **one** model endpoint (e.g., “gpt-35-turbo” or “gpt-4”).

2. **Content Safety Integration**  
   - Build a **simple chatbot** (web or console) that takes user prompts and calls the Azure OpenAI completion/chat API.  
   - Before returning the model’s response to the user, pass it through **Azure AI Content Safety** or at least the built-in content filtering in Azure OpenAI.  
   - If flagged as unsafe, provide a safe response or an error message.

3. **Explainability Element**  
   - Because LLMs are less about “feature attributions,” demonstrate how you handle **“system prompts”** or “chain-of-thought” constraints to keep the model’s reasoning safe.  
   - Document your approach to prompt engineering, partial fine-tuning, or use of **retrieval-augmented generation** to ground responses.  
   - If relevant, show logs of what categories of content get flagged (violent, hateful, etc.).

4. **Reflection**  
   - Discuss how generative AI can produce harmful content, how Azure addresses that, and any ethical or policy decisions you made (e.g., blocking certain keywords, limiting certain prompts).

---

#### **Option 4: Advanced Explainability in a Notebook (SHAP, LIME, etc.)**

1. **Azure Notebook Environment**  
   - Spin up an **Azure Machine Learning** notebook or **VS Code Remote** connected to an AML compute cluster.  
   - Install a library for advanced interpretability (e.g., **SHAP**, **LIME**, **Counterfactual_explanations**, **Saliency Maps** with PyTorch models, **Permutation Feature Importance**, or **Surrogate Models**).

2. **Model & Data**  
   - Train or import a custom model in PyTorch, TensorFlow, or scikit-learn.  
   - On inference, apply your chosen **explainability** technique (e.g., LIME for local explanations, or a Surrogate Model approach to interpret a black-box).

3. **Visualizations**  
   - Plot local explanation for a few data samples (e.g., show which features changed the prediction the most).  
   - Provide a global summary (e.g., overall feature ranking). If using images, demonstrate **Grad-CAM** or saliency methods.  
   - Optionally, show a short screencast or screenshots of your process.

4. **Discussion**  
   - Explain how your chosen method reveals insights about the model’s behavior.  
   - Compare it briefly to Azure’s built-in interpretability (if relevant).  
   - Note any potential limitations (e.g., LIME is approximate, SHAP can be expensive, etc.).

---

### Part C: Deliverables and Rubric

1. **Part A Report** – “Responsible AI on Azure” (15 points)  
   - Clarity, completeness of coverage on Azure’s ethical AI approach.  
   - Proper references/citations of official Microsoft documentation.

2. **Hands-on Demonstration (Part B)** (25 points)  
   - Correct setup and usage of the chosen approach (Azure ML pipeline, Azure OpenAI deployment, etc.).  
   - Evidence of a working solution (screenshots, code snippets, or a short demo video).

3. **Explainability / Fairness Analysis** (20 points)  
   - Depth and correctness of your interpretability approach or fairness metrics.  
   - Quality of explanation (SHAP plots, dashboards, or content safety logs).  
   - Clear conclusions about potential biases, key features, or safe content filtering results.

4. **Discussion & Reflection** (20 points)  
   - Thorough analysis of your findings: any biases uncovered, limitations, or ethical challenges.  
   - Proposed improvements or future steps.  
   - Understanding of how the responsible AI tooling influenced your development process.

5. **Documentation & Presentation** (20 points)  
   - Well-structured submission:  
     - A short readme or PDF describing how to run your solution.  
     - Clean code (Python notebooks or scripts) with comments.  
     - Visual aids (plots/screenshots from the RAI dashboard or other tools).  
   - Professional presentation style, referencing best practices in Responsible AI.

**Bonus (+5 points):**  
- Going beyond the basics. E.g., combining multiple advanced interpretability methods, exploring large datasets, using **retrieval-augmented generation** with content safety in Option 3, or automating an end-to-end MLOps pipeline with fairness checks gating production.

---

### Submission Guidelines

- **Individual or Team**: You may work solo or in pairs (up to 3 students).  
- **Azure Resources**: Use a free trial or institutional Azure subscription. Keep compute usage minimal (e.g., small CPU-based runs, free-tier cognitive services if possible).  
- **Project Artifacts**: Submit via GitHub (or compressed file) containing:  
  - Part A summary PDF  
  - Code notebooks/scripts + instructions  
  - Visual evidence (screenshots from Azure ML or deployment logs)  
  - Brief reflection (500 words) on what worked, what was difficult, and how you ensured ethical considerations  
- **Deadline**: *[Instructor sets date]*  

---

### Conclusion

This assignment will immerse you in **Azure’s Responsible AI** capabilities—covering fairness, interpretability, content safety, and more. You’ll develop practical experience with **Azure ML** or **Azure OpenAI** to build an ethically sound AI solution. The end goal is to leave you confident in applying Azure’s tools for real-world AI use cases that demand **explainability and responsible governance**. Good luck!
