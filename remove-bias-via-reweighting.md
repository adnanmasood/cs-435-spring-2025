Below is a **step-by-step** illustration of how to remove (or at least reduce) bias in a simple toy dataset by using a **reweighting** technique. 

---

## 1. The Scenario

Imagine we have 8 applicants for a job. Each applicant has a **Gender** (Male `M` or Female `F`) and an outcome variable **Hired**, which is `1` if they were hired and `0` if they were not.

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

- Total applicants: 8  
  - Males (M): 4 (IDs 1, 2, 3, 4)  
  - Females (F): 4 (IDs 5, 6, 7, 8)  
- Hired (1): 4 applicants (IDs 1, 2, 4, 8)  
- Not Hired (0): 4 applicants (IDs 3, 5, 6, 7)

### Observed “Bias” in Hiring Rates

- **Male hiring rate** = 3 out of 4 males hired = `3/4 = 0.75` (75%).  
- **Female hiring rate** = 1 out of 4 females hired = `1/4 = 0.25` (25%).

The difference is `0.75 − 0.25 = 0.50` (a 50% gap). We’d like a fairer dataset where this difference is minimized or eliminated.

---

## 2. Introducing the Idea of Reweighting

**Reweighting** is a way to make certain subgroups in the dataset count “more” or “less” so that overall, the dataset becomes more balanced. Formally, we assign a “weight” to each record. Think of the weight as how many “copies” of that record we imagine we have in our new, balanced dataset.

- Higher weight => the record is more influential.  
- Lower weight => the record is less influential.

By adjusting these weights, we aim for the distribution of `(Gender, Hired)` to be fairer in some sense (for example, making the hiring rates more similar across genders).

---

## 3. The Mathematical Formula for Reweighting

We use a standard reweighting approach that tries to make `Gender` independent of `Hired`. Concretely, the weight for an individual with gender `g` and hiring outcome `y` is:

```
w(g, y) = [ N * P(Gender = g) * P(Hired = y) ] / N_(g,y)
```

Where:
- `N` = total number of data points (here, 8).
- `P(Gender = g)` = fraction of the dataset with gender `g`.
- `P(Hired = y)` = fraction of the dataset with hired status `y`.
- `N_(g,y)` = number of records that have `(Gender = g, Hired = y)`.

### Step-by-Step Calculation

1. **Compute P(Gender = M)**:  
   - There are 4 Males out of 8 total, so `P(M) = 4/8 = 0.5`.

2. **Compute P(Gender = F)**:  
   - There are 4 Females out of 8 total, so `P(F) = 4/8 = 0.5`.

3. **Compute P(Hired = 1)**:  
   - 4 out of 8 are hired, so `P(Hired=1) = 4/8 = 0.5`.

4. **Compute P(Hired = 0)**:  
   - 4 out of 8 are not hired, so `P(Hired=0) = 4/8 = 0.5`.

Thus each category is half the population in this toy example.

#### Count how many records are in each (Gender, Hired) combination:

- `N_(M,1)` = number of Males hired = 3 (IDs 1, 2, 4).  
- `N_(M,0)` = number of Males not hired = 1 (ID 3).  
- `N_(F,1)` = number of Females hired = 1 (ID 8).  
- `N_(F,0)` = number of Females not hired = 3 (IDs 5, 6, 7).

Now apply the formula:

```
w(g, y) = [ 8 * P(Gender=g) * P(Hired=y) ] / N_(g,y)
```

Compute one by one:

1. w(M,1):

   ```
   w(M,1) = [ 8 * 0.5 * 0.5 ] / 3 
          = (8 * 0.25) / 3 
          = 2 / 3 
          ~ 0.6667
   ```

2. w(M,0):

   ```
   w(M,0) = [ 8 * 0.5 * 0.5 ] / 1
          = (8 * 0.25) / 1
          = 2
   ```

3. w(F,1):

   ```
   w(F,1) = [ 8 * 0.5 * 0.5 ] / 1
          = 2
   ```

4. w(F,0):

   ```
   w(F,0) = [ 8 * 0.5 * 0.5 ] / 3
          = (8 * 0.25) / 3
          = 2 / 3
          ~ 0.6667
   ```

---

## 4. Assigning Weights to Each Record

Using these group-based weights:

- `(Gender = M, Hired = 1)` => weight = `2/3`
- `(Gender = M, Hired = 0)` => weight = `2`
- `(Gender = F, Hired = 1)` => weight = `2`
- `(Gender = F, Hired = 0)` => weight = `2/3`

So, each ID gets:

| ID | Gender | Hired | Weight  |
|----|--------|-------|---------|
| 1  | M      | 1     | 2/3     |
| 2  | M      | 1     | 2/3     |
| 3  | M      | 0     | 2       |
| 4  | M      | 1     | 2/3     |
| 5  | F      | 0     | 2/3     |
| 6  | F      | 0     | 2/3     |
| 7  | F      | 0     | 2/3     |
| 8  | F      | 1     | 2       |

---

## 5. Checking the New Weighted Dataset

Let’s sum the weights by subgroup:

- **Males hired (M,1)**: 3 records each with weight `2/3`.  
  Total = `3 * (2/3) = 2`.

- **Males not hired (M,0)**: 1 record with weight `2`.  
  Total = `2`.

- **Females hired (F,1)**: 1 record with weight `2`.  
  Total = `2`.

- **Females not hired (F,0)**: 3 records each `2/3`.  
  Total = `3 * (2/3) = 2`.

**Summaries**:  
- Sum of all Male weights = `2 + 2 = 4`.  
- Sum of all Female weights = `2 + 2 = 4`.  
- Sum of all Hired weights = `2 + 2 = 4`.  
- Sum of all Not-Hired weights = `2 + 2 = 4`.  
- Grand total of all weights = `8`.

---

## 6. Verifying that Bias Has Been Reduced

### Weighted Hiring Rate for Males

- Weighted sum of Males who are hired = 2.  
- Total weighted sum for Males = 4.

```
Weighted Hiring Rate (Male) = 2 / 4 = 0.5 (50%)
```

### Weighted Hiring Rate for Females

- Weighted sum of Females who are hired = 2.  
- Total weighted sum for Females = 4.

```
Weighted Hiring Rate (Female) = 2 / 4 = 0.5 (50%)
```

The hiring rates are now the same: `50%` vs. `50%`. The difference is `0`.

---

## 7. High-School-Level Explanation

1. We started with a dataset showing a bigger fraction of men hired than women (75% vs. 25%).  
2. By giving **lower weights** to men who were hired (because they were “too common” in hires) and **higher weights** to men who were not hired, plus adjusting female weights similarly, we “rebalanced” the influence of each row.  
3. In the new “weighted world,” men and women both end up with the same hiring rate of 50%.  
4. **Reweighting** is like resampling or adjusting how many copies of each record you have, to correct for imbalances or biases in the original data.

---

## 8. Takeaways

- **Reweighting** is a technique to correct for bias by adjusting how heavily each record counts.  
- After reweighting, the **overall** (weighted) rates for each group can become more equitable.  
- In real situations, more complicated fairness definitions exist, but the concept is the same: **we alter weights** to reduce unwanted bias.

---

**That’s it!** You’ve seen how a simple formula for weights makes the dataset fairer by leveling hiring rates between males and females.
