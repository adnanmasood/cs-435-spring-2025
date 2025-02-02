## 1. The Scenario (Toy Dataset)

Let’s imagine we have 8 applicants for a job. Each applicant has a **Gender** (Male `M` or Female `F`) and an outcome variable called **Hired**, which is `1` if they were hired and `0` if they were not.

Here is our tiny dataset:

| ID | Gender | Hired |
|----|--------|-------|
| 1  | M      | 1     |
| 2  | M      | 1     |
| 3  | M      | 0     |
| 4  | M      | 1     |
| 5  | F      | 0     |
| 6  | F      | 0     |
| 7  | F      | 0     |
| 8  | F      | 1     |

Let’s see what’s happening:

- **Total applicants**: 8.
- **Males (M)**: 4 (IDs: 1, 2, 3, 4).
- **Females (F)**: 4 (IDs: 5, 6, 7, 8).
- **Hired = 1**: Applicants 1, 2, 4, 8 (4 total).
- **Not Hired = 0**: Applicants 3, 5, 6, 7 (4 total).

### Observed “Bias” in Hiring Rates

1. **Male hiring rate**: Out of 4 men, 3 were hired.  
   \[
   \text{Hiring Rate}_{\text{Male}} = \frac{3}{4} = 0.75 \quad (75\%).
   \]

2. **Female hiring rate**: Out of 4 women, 1 was hired.  
   \[
   \text{Hiring Rate}_{\text{Female}} = \frac{1}{4} = 0.25 \quad (25\%).
   \]

**Difference:** \( 75\% - 25\% = 50\% \). This indicates a large gap in hiring outcomes based on gender. We’d like a fairer dataset where this difference is minimized or eliminated.

---

## 2. Introducing the Idea of Reweighting

**Reweighting** is a way to make certain subgroups in the dataset count “more” or “less” so that overall, the dataset becomes more balanced. Formally, we assign a “weight” to each record. Think of the weight as how many “copies” of that record we imagine we have in our balanced world.

- If a record has a **high weight**, it becomes more influential in the analysis.  
- If a record has a **low weight**, it becomes less influential.

By adjusting these weights, we can aim for the **distribution** of (Gender, Hired) to become fairer.

---

## 3. The Mathematical Formula for Reweighting

We will use a standard reweighting method (proposed by Kamiran and Calders, 2012) that tries to make Gender independent of Hired in the dataset. That means in the new weighted world, we want:

\[
P(\text{Gender} = g, \text{Hired} = y) \approx P(\text{Gender} = g) \times P(\text{Hired} = y).
\]

Where:
- \(g\) can be M or F (Male or Female).
- \(y\) can be 0 or 1 (Not Hired or Hired).

The formula for the weight given to an individual who has Gender = \(g\) and Hired = \(y\) is:

\[
w(g, y) 
= \frac{ N \times P(\text{Gender}=g) \times P(\text{Hired}=y) }{ N_{(g,y)} },
\]

where:
- \(N\) = total number of data points (here, \(N=8\)).
- \(P(\text{Gender}=g)\) = fraction of the dataset with gender \(g\).
- \(P(\text{Hired}=y)\) = fraction of the dataset with hired status \(y\).
- \(N_{(g,y)}\) = number of records that have gender = \(g\) and hired status = \(y\).

### Step-by-Step Calculation

1. **Compute \(P(\text{Gender}=M)\)**:  
   There are 4 Males out of 8 total.  
   \[
   P(M) = \frac{4}{8} = 0.5.
   \]

2. **Compute \(P(\text{Gender}=F)\)**:  
   There are 4 Females out of 8 total.  
   \[
   P(F) = \frac{4}{8} = 0.5.
   \]

3. **Compute \(P(\text{Hired}=1)\)**:  
   4 out of 8 are Hired.  
   \[
   P(\text{Hired}=1) = \frac{4}{8} = 0.5.
   \]

4. **Compute \(P(\text{Hired}=0)\)**:  
   4 out of 8 are Not Hired.  
   \[
   P(\text{Hired}=0) = \frac{4}{8} = 0.5.
   \]

So each category (Male, Female, Hired=1, Hired=0) is half the population in this toy example.

#### Next, count how many records are in each (Gender, Hired) pair:

- \(N_{(M,1)}\): Number of Males who were Hired = 3 (IDs 1, 2, 4).  
- \(N_{(M,0)}\): Number of Males who were Not Hired = 1 (ID 3).  
- \(N_{(F,1)}\): Number of Females who were Hired = 1 (ID 8).  
- \(N_{(F,0)}\): Number of Females who were Not Hired = 3 (IDs 5, 6, 7).

Now we apply the formula for the weight:

\[
w(g, y) 
= \frac{ 8 \times P(\text{Gender}=g) \times P(\text{Hired}=y) }{ N_{(g,y)} }.
\]

Let’s compute these one by one:

1. **\( w(M,1) \)**:

   \[
   w(M, 1) 
   = \frac{ 8 \times 0.5 \times 0.5 }{ N_{(M,1)} } 
   = \frac{ 8 \times 0.25 }{ 3 } 
   = \frac{2}{3} \approx 0.6667.
   \]

2. **\( w(M,0) \)**:

   \[
   w(M, 0) 
   = \frac{ 8 \times 0.5 \times 0.5 }{ N_{(M,0)} } 
   = \frac{ 8 \times 0.25 }{ 1 } 
   = 2.
   \]

3. **\( w(F,1) \)**:

   \[
   w(F, 1) 
   = \frac{ 8 \times 0.5 \times 0.5 }{ N_{(F,1)} } 
   = \frac{ 8 \times 0.25 }{ 1 } 
   = 2.
   \]

4. **\( w(F,0) \)**:

   \[
   w(F, 0) 
   = \frac{ 8 \times 0.5 \times 0.5 }{ N_{(F,0)} } 
   = \frac{ 8 \times 0.25 }{ 3 } 
   = \frac{2}{3} \approx 0.6667.
   \]

---

## 4. Assigning Weights to Each Record

Using these group-based weights, we assign:

- If **Gender = M** and **Hired = 1**, the record’s weight = \(\tfrac{2}{3}\).
- If **Gender = M** and **Hired = 0**, the record’s weight = \(2\).
- If **Gender = F** and **Hired = 1**, the record’s weight = \(2\).
- If **Gender = F** and **Hired = 0**, the record’s weight = \(\tfrac{2}{3}\).

So, for **each ID** in our table:

| ID | Gender | Hired | Weight (w)           |
|----|--------|-------|----------------------|
| 1  | M      | 1     | \( \frac{2}{3} \)    |
| 2  | M      | 1     | \( \frac{2}{3} \)    |
| 3  | M      | 0     | \( 2 \)              |
| 4  | M      | 1     | \( \frac{2}{3} \)    |
| 5  | F      | 0     | \( \frac{2}{3} \)    |
| 6  | F      | 0     | \( \frac{2}{3} \)    |
| 7  | F      | 0     | \( \frac{2}{3} \)    |
| 8  | F      | 1     | \( 2 \)              |

---

## 5. Checking the New Weighted Dataset

Let’s see how these weights add up **by subgroup**:

1. **Males who were Hired (M,1):** There are 3 such records, each with weight \(\frac{2}{3}\).  
   Total weight = \(3 \times \frac{2}{3} = 2.\)

2. **Males who were Not Hired (M,0):** There is 1 such record, weight \(2.\)  
   Total weight = \(1 \times 2 = 2.\)

3. **Females who were Hired (F,1):** There is 1 such record, weight \(2.\)  
   Total weight = \(1 \times 2 = 2.\)

4. **Females who were Not Hired (F,0):** There are 3 such records, each with weight \(\frac{2}{3}\).  
   Total weight = \(3 \times \frac{2}{3} = 2.\)

### Summarize the Weighted Sums

- Sum of weights for **Males** (both hired and not hired) = \(2 + 2 = 4.\)  
- Sum of weights for **Females** (both hired and not hired) = \(2 + 2 = 4.\)  
- Sum of weights for **Hired (1)** (both M and F) = \(2 + 2 = 4.\)  
- Sum of weights for **Not Hired (0)** (both M and F) = \(2 + 2 = 4.\)  
- **Grand total** of all weights = \(2 + 2 + 2 + 2 = 8.\)

---

## 6. Verifying that Bias Has Been Reduced

Now, in this **weighted world**, let's see the “weighted hiring rates.”

### Weighted Hiring Rate for Males

- Weighted sum of Males **Hired** = 2 (from above).
- Weighted sum of all Males (Hired + Not Hired) = 2 (hired) + 2 (not hired) = 4.

\[
\text{Weighted Hiring Rate}_{\text{Male}} 
= \frac{\text{(Weighted sum of Males Hired)}}{\text{(Total weighted sum for Males)}} 
= \frac{2}{4} = 0.5 \quad (50\%).
\]

### Weighted Hiring Rate for Females

- Weighted sum of Females **Hired** = 2 (from above).
- Weighted sum of all Females = 2 (hired) + 2 (not hired) = 4.

\[
\text{Weighted Hiring Rate}_{\text{Female}} 
= \frac{\text{(Weighted sum of Females Hired)}}{\text{(Total weighted sum for Females)}} 
= \frac{2}{4} = 0.5 \quad (50\%).
\]

### Conclusion: Rates Are Now Equal

After reweighting, **both Males and Females** have a 50% hiring rate in this weighted dataset. The difference is now:

\[
50\% - 50\% = 0\%.
\]

Hence, we have effectively **removed** the apparent bias (with respect to demographic parity) by assigning appropriate weights to each record.

---

## 7. Why This Makes Sense (High School Explanation)

- We started with a dataset that had more men hired than women.
- By giving **lower weights** to men-who-were-hired (because they were “overrepresented” in the hiring group) and **higher weights** to men-who-were-not-hired (and similarly for females), we adjusted how much each data point “counts.”
- After reweighting, in a sense, we have “resampled” the data to make it look like men and women have the same chance of being hired.
- This is one way to address **demographic parity** fairness: ensuring that the fraction of hires is the same for Males and Females in the data.

---

## 8. Big Takeaways

1. **Reweighting** is a technique used to correct for bias in data by assigning higher or lower weights (importance) to different records.
2. After applying the new weights, we check how the weighted distributions (like weighted hiring rates) match up.
3. In our toy example, we demonstrated that reweighting removed the gap (50% difference) between male and female hiring rates, bringing it to 0%.

---

### Final Note

This simple demonstration shows how reweighting can fix **demographic parity** (the overall hiring rates across groups). In real-world problems, there are often multiple definitions of fairness, and reweighting is one of many possible techniques. However, the main idea remains the same: **make the data more balanced by adjusting how heavily each data point is counted.**

---

**Congratulations, you now understand the basics of how to remove bias via reweighting!**
