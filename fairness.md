## Fairness

### 1. Foundational Overview
Fairness means treating people as equally as possible, regardless of their backgrounds or characteristics. In AI, it means ensuring that computers make decisions that don’t unfairly favor or hurt certain groups.

### 2. Broader Insight
Fairness involves creating and using algorithms that do not discriminate against individuals or groups based on sensitive factors (like race, gender, or socioeconomic status). Ensuring fairness requires identifying and minimizing any unintended biases that may come from the data or the way a model is built.

### 3. Structured Understanding
In machine learning systems, fairness is the principle that predictive or decision-making models should produce equitable outcomes or treatments for all groups of people, particularly historically marginalized ones. Achieving fairness can involve adjusting data, model training processes, or outputs so that no group is systematically disadvantaged.

### 4. Advanced Academic Interpretation
Fairness in algorithmic contexts encompasses multiple notions—such as **demographic parity, equalized odds, and individual fairness**—each with its own technical definitions and trade-offs. Practitioners often need to choose or balance between these definitions based on ethical, legal, and domain-specific considerations.

### 5. Research-Level Exposition
From a research perspective, fairness is a multi-faceted concept that seeks to ensure that the expected utility or loss across protected classes, sub-populations, or individuals is balanced in a manner consistent with societal and legal norms. The complexities arise in reconciling these mathematical formalizations (e.g., constraints on statistical parity, calibration, and more sophisticated measures) with real-world contexts, policy requirements, and intersectional identities.

---

## II. Mathematical Definition of Fairness

A well-cited mathematical framing of fairness in classification tasks involves **group fairness** criteria. One common criterion is **statistical parity (or demographic parity)**:

> **Statistical Parity**  
> A classifier \(\hat{Y}\) satisfies statistical parity if  
> \[
> P(\hat{Y} = 1 \mid A = a) = P(\hat{Y} = 1 \mid A = a')
> \]  
> for any two groups \(a\) and \(a'\) (e.g., different demographic groups) based on a protected attribute \(A\).  

Other metrics include **Equalized Odds** and **Equality of Opportunity**, which incorporate the ground truth label \(Y\):

> **Equalized Odds**  
> \[
> P(\hat{Y} = 1 \mid A = a, Y = y) = P(\hat{Y} = 1 \mid A = a', Y = y)
> \]  
> for \(y \in \{0,1\}\) and for groups \(a, a'\).  

---

## III. Mathematical Representation and Explanation

In a binary classification setting:

- Let \(X\) be the set of features (e.g., income, age, education level).  
- Let \(A\) be a protected attribute (e.g., race or gender).  
- Let \(Y\) be the true label (e.g., “qualified” vs. “not qualified”).  
- Let \(\hat{Y}\) be the predicted label by the model.

To **mathematically represent** fairness, one commonly imposes constraints on the joint or conditional distributions of \(\hat{Y}\) and \(A\). For instance, *statistical parity* imposes a constraint on the marginal distribution of predictions across groups, whereas *equalized odds* imposes constraints on the conditional distributions given the true label.

---

## IV. Real-World Example of Fairness (with Citation & Attribution)

One prominent real-world example is the use of **risk assessment tools in the criminal justice system**, such as the COMPAS system in the United States. A ProPublica investigation found that the system exhibited potential **racial bias** in predicting recidivism rates, disproportionately labeling Black defendants as higher-risk compared to white defendants with comparable histories (Angwin et al., 2016).

> **Citation:**  
> Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). *Machine Bias*. ProPublica.  
> [https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)

### Mathematical/Code Representation Example

Using Python pseudocode to compute **statistical parity difference**:

```python
import numpy as np

# Suppose y_pred is a list/array of predictions (0 or 1)
# A is a list/array of the protected attribute (e.g., 'Group1' or 'Group2')

def statistical_parity_difference(y_pred, A, protected_value='Group1'):
    # Convert protected_value group to a boolean mask
    mask_protected = (A == protected_value)
    mask_non_protected = (A != protected_value)
    
    # Calculate prediction rate for each group
    rate_protected = np.mean(y_pred[mask_protected])
    rate_non_protected = np.mean(y_pred[mask_non_protected])
    
    # Difference in rates
    return rate_protected - rate_non_protected

# Example usage:
# y_pred = np.array([1, 0, 1, 1, 0])  # example predictions
# A = np.array(['Group1','Group2','Group1','Group2','Group1'])
# diff = statistical_parity_difference(y_pred, A, protected_value='Group1')
# print("Statistical parity difference:", diff)
```

If `diff` is close to zero, the model exhibits less disparity between protected vs. non-protected groups in terms of positive prediction rates.

---

## V. Quantitative Definition of Fairness with a Seminal Example

Consider the **Equalized Odds** metric introduced by Hardt, Price, and Srebro (2016) in *Equality of Opportunity in Supervised Learning* (NeurIPS). Under *equalized odds*, the model’s true positive rate (TPR) and false positive rate (FPR) must be similar across different groups:

- **TPR**: \( P(\hat{Y} = 1 \mid Y=1, A=a) \approx P(\hat{Y} = 1 \mid Y=1, A=a') \)  
- **FPR**: \( P(\hat{Y} = 1 \mid Y=0, A=a) \approx P(\hat{Y} = 1 \mid Y=0, A=a') \)

In a quantitative sense, you might say a model is “fair” if the difference in TPR or FPR across groups does not exceed a certain threshold (e.g., 5%).

---

## VI. Common Questions (FAQ) on Fairness

1. **Why does bias occur in AI systems?**  
   Bias often arises from historical data containing societal prejudices or from unbalanced sampling and labeling processes.

2. **Is it possible to eliminate all bias?**  
   Completely eliminating bias is extremely challenging because societal biases can be deeply ingrained in data. However, techniques exist to mitigate and reduce these biases significantly.

3. **Does ensuring fairness always reduce model accuracy?**  
   There can be a trade-off between accuracy and fairness, but it is not always a strict dichotomy. Sometimes, fairness interventions can improve generalization by reducing overfitting to biased patterns.

4. **Which fairness definition should I use?**  
   The choice depends on the application’s ethical, legal, and contextual requirements. Different fairness definitions may be incompatible with each other.

5. **How do I handle intersectionality (e.g., race and gender combined)?**  
   Intersectional fairness requires evaluating models across combinations of protected attributes. This can increase complexity, but it is crucial for thorough fairness assessments.

---

## VII. Key Papers and Links

1. **“Fairness Through Awareness”**  
   *Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel (STOC, 2012)*  
   [Paper Link](https://arxiv.org/abs/1104.3913)

2. **“Equality of Opportunity in Supervised Learning”**  
   *Moritz Hardt, Eric Price, Nathan Srebro (NeurIPS, 2016)*  
   [Paper Link](https://arxiv.org/abs/1610.02413)

3. **“Algorithmic Fairness and the Law”**  
   *Jon Kleinberg, Jens Ludwig, Sendhil Mullainathan, and Cass R. Sunstein (2018)*  
   [Paper Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3288959)

4. **“Fairness and Machine Learning”**  
   *Solon Barocas, Moritz Hardt, and Arvind Narayanan (Book, 2021)*  
   [Book Link](https://fairmlbook.org/)

---

## VIII. Relevant Algorithms in Detail

1. **Preprocessing Methods**  
   - **Reweighting**: Adjust the weights of training samples to offset underrepresentation or overrepresentation of certain groups.  
   - **Disparate Impact Remover**: Transform data features to remove group-dependent information while preserving rank-ordering.

2. **In-Processing Methods**  
   - **Adversarial Debiasing**: Train a model to predict the outcome while an adversary tries to predict the protected attribute from the model’s internal representations.  
   - **Fair Regularization**: Incorporate fairness constraints directly into the optimization objective (e.g., penalizing disparities in predictions across groups).

3. **Post-Processing Methods**  
   - **Threshold Adjustment**: Apply different decision thresholds for different demographic groups to achieve a specific fairness metric.  
   - **Reject Option**: For examples with low confidence, delay the decision and route them for human oversight.

---

## IX. Relevant Techniques in Detail

- **Data Auditing**: Identifying bias in datasets through exploratory analysis, bias metrics, and segmentation.  
- **Feature Engineering for Fairness**: Removing or transforming sensitive attributes while retaining predictive power.  
- **Fairness Constraint Optimization**: Using convex or non-convex optimization methods that directly encode fairness constraints in the objective function.  
- **Adversarial Training**: Training an auxiliary model to detect protected information in intermediate representations, forcing the main model to “unlearn” that information.

---

## X. Relevant Benchmarks

1. **UCI Adult Dataset**  
   - Commonly used for testing fairness algorithms in income classification tasks.  
   - Known biases in gender and race distribution.

2. **COMPAS Recidivism Dataset**  
   - Used in criminal justice risk assessment.  
   - Exhibits well-documented racial disparities.

3. **German Credit Dataset**  
   - Used for credit risk predictions.  
   - Contains protected attributes related to age and gender.

4. **Bank Marketing Dataset**  
   - Evaluates fairness in marketing campaigns.  
   - Potential biases related to socioeconomic factors.

---

## XI. Relevant Leaderboards

Fairness-related leaderboards are often organized by academic challenges or open repositories. Some are on platforms like [OpenML](https://www.openml.org/) or within research groups hosting competitions. While there is no single universal leaderboard like ImageNet’s top-1 accuracy, many conferences (e.g., NeurIPS, ICML) and workshops track performance on fairness tasks in specialized competitions.

---

## XII. Relevant Libraries

1. **AI Fairness 360 (AIF360)** by IBM  
   - Provides a comprehensive toolkit for bias detection and mitigation (pre-, in-, and post-processing).

2. **Fairlearn** by Microsoft  
   - Focuses on assessing fairness metrics and enabling various mitigation approaches.  
   - Offers a user-friendly dashboard for visualizing model performance across groups.

3. **Themis-ML**  
   - Provides fairness-focused transformers and classifiers, integrating with scikit-learn workflows.

4. **TensorFlow Constrained Optimization**  
   - Allows specifying fairness constraints as part of the model’s optimization objective.

---

## XIII. Relevant Metrics in Detail

1. **Statistical Parity / Demographic Parity**  
   - Checks if the positive prediction rates are the same across groups.

2. **Equalized Odds**  
   - Measures if the TPR and FPR are the same across groups.

3. **Equality of Opportunity**  
   - Ensures equal TPR for different groups (may ignore FPR).

4. **Calibration**  
   - Ensures that among those assigned a risk score \(s\), the actual probability of a positive outcome is the same across groups.

5. **Disparate Impact**  
   - Ratio of prediction rates across groups; used in legal contexts (80% rule).

---

## XIV. Fairness Classroom Discussion Prompts

Below are prompts designed to spark discussion around the ethical and societal implications of Fairness in AI:

### 1. Predictive Policing
**Prompt**: When predictive policing tools disproportionately target minority neighborhoods due to biased historical data, is the AI at fault, or does it merely expose systemic issues within law enforcement?  
**Discussion Angle**: Spurs debate on whether AI amplifies existing biases or acts as a mirror to societal inequities.

### 2. Algorithmic College Admissions
**Prompt**: If an AI system rejects more applicants from certain socioeconomic backgrounds based on historical acceptance trends, is it fair to use such a tool? Should admissions also consider holistic (human) review?  
**Discussion Angle**: Explores the complexity of balancing efficiency with equity in high-stakes decisions.

### 3. Cultural Nuances in Global AI
**Prompt**: Fairness standards in one country might differ from another. How do we ensure global AI products respect cultural norms without imposing a one-size-fits-all fairness framework?  
**Discussion Angle**: Highlights the complexity of local vs. universal ethical standards.

---

## Concluding Remarks

Fairness in AI is a **multi-dimensional challenge** that intersects with social norms, legal frameworks, and the technical design of machine learning systems. Addressing fairness requires:

- Rigorous **data collection** and **bias audits**.  
- Carefully **defining fairness metrics** aligned with the context.  
- Applying **appropriate algorithms** and **techniques** (pre-, in-, or post-processing).  
- Ongoing **monitoring and evaluation** to handle shifting social contexts and emerging biases.

By continuously refining these methods and metrics, and engaging in **inclusive, interdisciplinary dialogue**, we can strive toward AI systems that are equitable, trustworthy, and beneficial to all.
