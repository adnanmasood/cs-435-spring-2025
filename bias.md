# **Bias**

## 1. Definitions of Bias at Five Levels of Detail

### 1.1. Basic Insight
> **Bias** is when something is unfairly guided or influenced toward a certain outcome. Think of it like a pair of tinted glasses that changes how you see everything.

### 1.2. Foundational Understanding
> **Bias** occurs when data, decisions, or processes are skewed toward certain groups or outcomes, leading to unfair or imbalanced results. This can happen in people (human biases) or in algorithms (machine biases).

### 1.3. Intermediate Deep Dive
> **Bias** in artificial intelligence refers to systematic errors in machine learning models or decision-making systems that cause certain groups or outcomes to be favored or disfavored. These biases can stem from historical data, design choices, or hidden assumptions in algorithms. Over time, they can lead to unfair or discriminatory impacts if not recognized and mitigated.

### 1.4. Comprehensive Overview
> **Bias** is an omnipresent phenomenon manifesting whenever a non-representative process influences data generation, selection, or interpretation. In AI, bias arises through:
> - **Data Bias**: Sample imbalance, historical prejudices, or flawed labeling.
> - **Algorithmic Bias**: Model structures or training procedures that disproportionately weight certain features or outcomes.
> - **Systemic Bias**: Broader societal and institutional patterns embedded into datasets and decision pipelines.
> The result is a systematic deviation of model predictions or decisions that can disadvantage specific subpopulations or distort factual accuracy.

### 1.5. Expert/Scholarly Analysis
> **Bias** in AI systems can be formally conceptualized as a deviation from some normative benchmark of “fairness” or “objectivity,” frequently operationalized through fairness criteria (e.g., demographic parity, equal opportunity). Under this lens, an AI system exhibits bias if its decision boundary or predicted conditional distributions differ significantly across protected groups (e.g., race, gender) absent legitimate, policy-sanctioned reasons. This includes (but is not limited to) biases arising from distributional drift, latent variable confounding, and institutionalized historical inequities in the dataset. In advanced fairness research, bias is further dissected into causal or observational frameworks, establishing robust definitions that consider interventions, counterfactual scenarios, and ethical constraints (see Pearl’s structural causal models and Hardt et al.’s fairness metrics).

---

## 2. Mathematical Definition of Bias

In statistics and machine learning, one common notion of **bias** is the difference between the expected value of an estimator and the true value of the parameter being estimated. Formally:

\[
\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta,
\]

where:
- \(\hat{\theta}\) is the estimator (e.g., a model’s predicted parameter),
- \(\theta\) is the true parameter value,
- \(\mathbb{E}[\hat{\theta}]\) denotes the expected value of \(\hat{\theta}\).

In practical machine learning, we often think of **bias terms** (e.g., \(b\) in a linear model \(y = w^Tx + b\)), or systematically skewed predictions across subgroups. The concept of **bias** can also extend to group-level metrics, for example, differences in false positive rates between subpopulations.

---

## 3. Mathematical Representation of Bias in Models

- **Bias Term in a Linear Model**: In a linear regression or classification model \(y = w^T x + b\), the parameter \(b\) is literally called the "bias." Although this is a different usage of the word than social/ethical bias, it symbolizes a shift in the decision boundary or intercept which can sometimes reflect systematic preference if improperly fitted.

- **Group Fairness Metrics**: Another representation is to measure the difference in outcomes for different groups. For instance, let:
  \[
  \Delta = P(\hat{Y}=1 \mid A=g_1) - P(\hat{Y}=1 \mid A=g_2),
  \]
  where \(A\) is a protected attribute (e.g., gender) and \(\hat{Y}=1\) is a positive classification (e.g., being hired). **Bias** can be operationalized as a significant difference in \(\Delta\).

---

## 4. Real-World Example of Bias (Citation & Attribution)

**Example: COMPAS Algorithm**  
Researchers at ProPublica (Angwin et al., 2016) analyzed the COMPAS system used to predict recidivism in the U.S. criminal justice system. They found that the tool systematically overestimated the risk of re-offense for Black defendants while underestimating it for White defendants, highlighting **racial bias** in risk assessment models.

**Citation**:  
Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). *Machine Bias: There’s software used across the country to predict future criminals. And it’s biased against blacks.* ProPublica.

---

### 4.1. Mathematical / Code Representation of Imbalanced Data Leading to Skewed Predictions

Below is a simplified Python-like pseudocode illustrating how imbalance can lead to biased outcomes:
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Suppose X_minority has fewer samples than X_majority
X_minority, y_minority = load_minority_data()  # smaller group
X_majority, y_majority = load_majority_data()  # larger group

# Combine datasets
X_train = np.concatenate([X_minority, X_majority], axis=0)
y_train = np.concatenate([y_minority, y_majority], axis=0)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_train)

# Check accuracy for each group
acc_minority = accuracy_score(y_minority, model.predict(X_minority))
acc_majority = accuracy_score(y_majority, model.predict(X_majority))

print("Accuracy on minority group:", acc_minority)
print("Accuracy on majority group:", acc_majority)
```
Because \(X_{\text{minority}}\) is underrepresented, the learned parameters may be less tuned to that subgroup, reflecting **bias** in performance.

---

## 5. Quantitative Definition of Bias with a Seminal Example

One seminal example is from **Feldman et al. (2015)** on the notion of **disparate impact**:  
\[
\text{Disparate Impact} = \left| P(\hat{Y} = 1 | A=g_1) - P(\hat{Y} = 1 | A=g_2) \right|.
\]  
If this value is high, there is potential evidence of bias. For instance, if \(g_1\) corresponds to a marginalized community and the system seldom assigns a positive outcome to them, the **disparate impact** is large.

---

## 6. Metrics and Formulas Capturing Different Types of Bias

1. **Statistical Parity / Demographic Parity**  
   \[
   P(\hat{Y}=1 \mid A=g_1) = P(\hat{Y}=1 \mid A=g_2).
   \]  
   A model is considered fair under this criterion if the probability of a positive outcome is the same for both groups.

2. **Equalized Odds / Equality of Opportunity (Hardt et al., 2016)**  
   - **True Positive Rate (TPR) Parity**:  
     \[
     P(\hat{Y} = 1 \mid Y=1, A=g_1) = P(\hat{Y} = 1 \mid Y=1, A=g_2).
     \]
   - **False Positive Rate (FPR) Parity**:  
     \[
     P(\hat{Y} = 1 \mid Y=0, A=g_1) = P(\hat{Y} = 1 \mid Y=0, A=g_2).
     \]

3. **Predictive Parity**  
   \[
   P(Y=1 \mid \hat{Y}=1, A=g_1) = P(Y=1 \mid \hat{Y}=1, A=g_2).
   \]  
   The probability of a positive label given a positive prediction is the same across groups.

4. **Calibration**  
   For any score \(s\), if the model is well-calibrated, then for each group:
   \[
   P(Y=1 \mid S=s, A=g_1) = P(Y=1 \mid S=s, A=g_2).
   \]

---

## 7. Frequently Asked Questions (FAQs) on Bias

1. **Is Bias always negative?**  
   - In everyday language, “bias” often implies unfairness. In statistics, bias is merely a deviation from truth. Not all bias leads to unethical or discriminatory outcomes, but in AI ethics, we typically worry about harmful biases.

2. **What is the difference between prejudice and statistical bias?**  
   - **Prejudice** is a preconceived opinion not based on reason or experience, often directed toward people.  
   - **Statistical Bias** is a systematic deviation in an estimator or model outcomes. A system can be statistically biased without reflecting prejudicial intent (though it can still lead to discriminatory outcomes).

3. **How can bias be introduced at various stages?**  
   - **Data Collection**: Non-representative sampling.  
   - **Data Labeling**: Subjective labeling with possible human prejudice.  
   - **Model Training**: Overfitting majority classes, ignoring minority subgroups.  
   - **Deployment**: Model performance shifts due to changes in the real world, or mismatch in context.

4. **Who is responsible for mitigating AI bias?**  
   - Responsibility often lies with data scientists, product managers, and organizational leadership. Ultimately, it’s a shared responsibility, spanning those who collect data, develop algorithms, and make policy decisions.

---

## 8. Key Papers and Links Discussing Bias in Detail

1. **ProPublica’s “Machine Bias”** (Angwin et al., 2016):  
   [https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)

2. **Gender Shades (Buolamwini & Gebru, 2018)**:  
   [http://proceedings.mlr.press/v81/buolamwini18a.html](http://proceedings.mlr.press/v81/buolamwini18a.html)  
   Demonstrates facial recognition disparities across gender and skin type.

3. **Feldman et al. (2015) - Certifying and Removing Disparate Impact**:  
   [https://papers.nips.cc/paper/2015/file/cfa7a55f79f5c5c4372d54e6f4f4d046-Paper.pdf](https://papers.nips.cc/paper/2015/file/cfa7a55f79f5c5c4372d54e6f4f4d046-Paper.pdf)

4. **Hardt, Price, & Srebro (2016). Equality of Opportunity in Supervised Learning**:  
   [https://papers.nips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html](https://papers.nips.cc/paper/2016/hash/9d2682367c3935defcb1f9e247a97c0d-Abstract.html)

5. **Fairness and Machine Learning (Barocas, Hardt, & Narayanan, 2019)**:  
   [https://fairmlbook.org/](https://fairmlbook.org/)  
   An in-depth open-source book on algorithmic fairness.

6. **FAccT (ACM Conference on Fairness, Accountability, and Transparency)**:  
   [https://facctconference.org/](https://facctconference.org/)  
   Major academic conference on fairness research.

---

## 9. Relevant Algorithms to Mitigate or Measure Bias

### 9.1. Data Preprocessing Algorithms
- **Reweighing (Kamiran & Calders, 2012)**: Adjusts the weights of training examples to decrease the correlation between protected attributes and the outcome.
- **Synthetic Oversampling** (e.g., SMOTE): Balances the dataset by oversampling minority classes or undersampling majority classes.
- **Disparate Impact Remover** (Feldman et al., 2015): Edits feature values to remove correlations with protected attributes.

### 9.2. In-Processing Methods
- **Adversarial Debiasing**: Uses an adversary to minimize the ability of the model to predict sensitive attributes, encouraging fair representations.
- **Regularized Fairness Approaches**: Modifies the loss function (e.g., adds a fairness constraint or regularization term like “\(\lambda \times \text{fairness-penalty}\)”).

### 9.3. Post-Processing Techniques
- **Equalized Odds Post-Processing (Hardt et al., 2016)**: Adjusts the decision threshold separately for different groups to equalize TPR and FPR.
- **Reject Option Classification**: Allows uncertain cases (particularly those near the decision boundary) to be re-labeled to improve fairness metrics.

---

## 10. Relevant Techniques in Great Detail

- **Causal Inference Approaches**: Model relationships using a directed acyclic graph (DAG). Identify whether correlations with protected attributes are causal or spurious.  
- **Counterfactual Fairness**: Ensures that a prediction for an individual in a hypothetical scenario (where sensitive attributes differ) remains consistent.

---

## 11. Relevant Benchmarks in the Context of Bias

1. **COMPAS Dataset**  
   - Risk assessment data focusing on recidivism.  
   - Widely used for demonstrating racial bias.

2. **Adult Income (UCI)**  
   - Predict whether income exceeds \$50K/yr based on census data.  
   - Known for potential gender and racial biases.

3. **German Credit Dataset**  
   - Credit approval data with various demographic features.  
   - Commonly used to evaluate fairness in lending models.

4. **Facial Recognition Benchmarks**  
   - Benchmarks like LFW (Labeled Faces in the Wild) or IJB datasets, studied by Buolamwini & Gebru (2018) to show racial/gender inaccuracies.

---

## 12. Relevant Leaderboards in the Context of Bias

Although there is no single, universally recognized public “leaderboard” exclusively for bias mitigation, **some competitions and platforms** occasionally feature fairness challenges:

- **Kaggle Fairness Challenges**: Kaggle has hosted competitions, like the “DonorsChoose” or “Google AI Challenge,” that included fairness components.  
- **NeurIPS Competitions**: In past years, there have been fairness-based challenges (e.g., “Adverse Impact Reduction” challenges).

---

## 13. Relevant Libraries for Bias Detection & Mitigation

1. **IBM AI Fairness 360 (AIF360)**  
   - [https://github.com/IBM/AIF360](https://github.com/IBM/AIF360)  
   - Provides a comprehensive suite of metrics and algorithms for detecting and mitigating bias.

2. **Fairlearn**  
   - [https://github.com/fairlearn/fairlearn](https://github.com/fairlearn/fairlearn)  
   - An open-source Python package by Microsoft for assessing and improving fairness in ML models.

3. **Themis-ML**  
   - Focuses on fairness-aware ML, includes discrimination discovery and correction.

4. **Google’s ML-Fairness Gym**  
   - [https://github.com/google/ml-fairness-gym](https://github.com/google/ml-fairness-gym)  
   - Simulators for exploring fairness research in sequential decision-making contexts.

---

## 14. Relevant Metrics in Great Detail

1. **Statistical Parity / Demographic Parity**  
   - Pros: Simple to interpret.  
   - Cons: May ignore the true label distribution and lead to “artificial fairness.”

2. **Disparate Impact**  
   - A ratio-based measure:  
     \[
     \frac{P(\hat{Y}=1 \mid A=g_1)}{P(\hat{Y}=1 \mid A=g_2)}.
     \]  
   - The 80% rule states that if the ratio is below 0.8, there may be illegal discrimination.

3. **Equalized Odds**  
   - Pros: Considers label information (TPR, FPR).  
   - Cons: Hard to satisfy simultaneously for multiple groups and can reduce overall accuracy.

4. **Predictive Parity**  
   - Pros: Focuses on calibration.  
   - Cons: Can conflict with other metrics like Equalized Odds.

5. **Calibration**  
   - Ensures that the predicted probability reflects the true likelihood for all groups.  
   - Often in tension with other fairness metrics.

---

## 15. Bias Classroom Discussion Prompts

Below are four discussion prompts designed to spark conversation and critical thinking about bias in AI:

### 15.1. Hiring Algorithms and Diversity
- **Prompt**: Many companies use AI to screen job applicants. If the algorithm favors certain demographics based on historical hiring data, is that bias the fault of AI or human-driven biases embedded in the data?  
- **Discussion Angle**: Encourages debate on the root cause of algorithmic bias (data vs. algorithm vs. society), and explores moral/organizational responsibility for correction.

### 15.2. Facial Recognition and Racial Disparities
- **Prompt**: Should companies be allowed to deploy facial recognition systems known to have higher false positive rates for certain ethnic groups? If so, under what conditions?  
- **Discussion Angle**: Balances the potential utility of facial recognition for public safety against ethical and discriminatory concerns, and highlights corporate accountability and regulatory oversight.

### 15.3. Consumer Lending and Credit Scores
- **Prompt**: A bank uses an AI model that appears to approve fewer loans for certain neighborhoods with historically lower credit scores. Is the AI “biased,” or is it reflecting systemic socio-economic patterns?  
- **Discussion Angle**: Probes the fine line between merely reflecting existing inequalities and perpetuating them. Emphasizes how one might design systems that are equitable and socially responsible.

### 15.4. Self-Driving Vehicles
- **Prompt**: If self-driving cars are tested primarily in affluent neighborhoods, could this bias them against roads, signage, or pedestrian behavior in underserved areas, leading to accidents or errors?  
- **Discussion Angle**: Highlights how geographic or socio-economic sampling biases can lead to safety risks. Explores the importance of diverse testing environments.

---

## Conclusion

Bias in AI is both a technical and a social challenge. It involves understanding **statistical definitions**, **ethical considerations**, and **practical mitigation strategies**. By combining rigorous measurement (using fairness metrics and algorithms) with vigilant organizational oversight, we can strive to build AI systems that are more equitable and just.
