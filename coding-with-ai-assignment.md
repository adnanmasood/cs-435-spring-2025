## **Assignment: LeetCode Problem Solving with AI**

### **Objective**

1. **Hands-On Practice**: Strengthen algorithmic thinking by solving a coding challenge from LeetCode.  
2. **LLM Comparison**: Explore how different LLMs (e.g., ChatGPT, Gemini, Claude, Llama, etc.) approach and solve the same problem.  
3. **Performance & Accuracy Assessment**: Compare the two LLM-generated solutions to **your own** (manual) solution in terms of correctness, efficiency, readability, and clarity.

---

### **Instructions**

1. **LeetCode Problem Selection**  
   - Each student will receive **one LeetCode problem** from the instructor (e.g., *Medium* or *Hard* difficulty).  
   - **Do not** share your assigned problem with classmates—everyone should have a unique challenge.

2. **Manual Solution**  
   - Solve the assigned LeetCode problem **on your own** in any programming language (e.g., Python, Java, C++).  
   - Write clean, well-documented code.
   - Capture the **time complexity** and **space complexity** of your approach.

3. **LLM Solutions**  
   - Pick **two different LLMs** (e.g., ChatGPT and Bard, or ChatGPT and Llama 2, etc.).  
   - Prompt each LLM with a *succinct, clear problem statement*—ideally the same text from the LeetCode prompt.  
   - Gather the LLM-generated solutions (copy/paste into separate code files or cells).  
   - **Do not** provide your own solution to the LLMs first—let them generate their own solutions from the problem statement alone.

4. **Testing & Comparison**  
   - **Compile and run** each solution (your manual one + the two LLM solutions) against:  
     - Sample test cases provided by LeetCode.  
     - Additional edge cases you design yourself.  
   - Document how each solution performs:  
     1. **Correctness**: Does it pass all test cases?  
     2. **Efficiency**: Track runtime or big-O complexity.  
     3. **Readability & Structure**: Is the code well-organized and commented?

5. **Deliverables**  
   - **Source Code**:  
     1. Your manual solution.  
     2. LLM #1 solution (with the name/version of the LLM used).  
     3. LLM #2 solution (with the name/version of the LLM used).  
   - **Short Report** *(2–3 pages or equivalent)* covering:  
     1. **Problem Statement**: Summarize your assigned LeetCode challenge.  
     2. **Your Solution Approach**: Key algorithmic ideas, time/space complexity, any optimizations.  
     3. **LLM #1 Solution**: Observations on correctness, performance, style.  
     4. **LLM #2 Solution**: Observations on correctness, performance, style.  
     5. **Comparison & Analysis**: Which solution was most accurate? Fastest? Easiest to read?  
     6. **Takeaways**: Reflect on how LLMs might complement or challenge human problem-solving.

---

## **Sample Code Snippet**  
*(For illustration only; each student’s actual problem will differ.)*

**Example Problem**: *Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.*

**Manual/Python Solution** (student’s approach):
```python
def two_sum(nums, target):
    """
    Manual solution for the 'Two Sum' problem.
    Time Complexity: O(n) 
    Space Complexity: O(n)
    Uses a hash map to store needed complements.
    """
    complement_map = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in complement_map:
            return [complement_map[complement], i]
        complement_map[num] = i
    
    return []  # Return an empty list if no solution is found

# Test the function
if __name__ == "__main__":
    nums_test = [2, 7, 11, 15]
    target_test = 9
    print(two_sum(nums_test, target_test))  # Expected output: [0, 1]
```

---

### **LLM-Generated Solutions**

**LLM #1** (e.g., ChatGPT) might provide something like:
```python
def two_sum_llm1(nums, target):
    # LLM #1 solution; possibly similar or entirely different approach
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
```
- Possibly a brute-force approach (`O(n^2)`).

**LLM #2** (e.g., Bard) might produce a variant:
```python
def two_sum_llm2(nums, target):
    # LLM #2 solution with a dictionary approach
    needed = {}
    for i, val in enumerate(nums):
        if (target - val) in needed:
            return [needed[target - val], i]
        needed[val] = i
    return []
```
- Similar to the manual approach but with different variable names or logic structure.

---

### **Short Comparison**

- **LLM #1**:  
  - Correctness: Passes standard test `[2, 7, 11, 15], 9`.  
  - Time Complexity: O(n^2). This is slower for large inputs.  
  - Code Readability: Straightforward brute-force, less efficient.

- **LLM #2**:  
  - Correctness: Also passes the basic test.  
  - Time Complexity: O(n). More optimal solution.  
  - Code Readability: Clear dictionary usage, variable naming might be ambiguous, but logic is concise.

*(In your report, you would detail actual tests and possibly discover differences in edge cases.)*

---

## **Grading Rubric**

| **Criteria**                               | **Exemplary (A)**                                                                                                                         | **Proficient (B)**                                                                                                             | **Developing (C)**                                                                                                       | **Needs Improvement (D/F)**                                                                                         |
|:------------------------------------------ |:------------------------------------------------------------------------------------------------------------------------------------------ |:------------------------------------------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------ |:-------------------------------------------------------------------------------------------------------------------- |
| **Manual Solution Quality**               | Correct, efficient, well-documented code; clear demonstration of problem-solving approach.                                                 | Code works correctly with minimal or no errors; moderate clarity and documentation.                                           | Partially correct or inefficient solution; code is difficult to follow.                                               | Incorrect or incomplete solution; no clear approach or documentation.                                              |
| **LLM Implementation & Comparison**       | Thoroughly tested both LLM solutions; comprehensive discussion of differences in approach, complexity, and correctness.                    | Provided a reasonable set of tests; basic comparison of LLM solutions.                                                         | Limited testing or superficial comparison; missing performance analysis.                                             | Little to no testing or comparison; essentially no analysis of LLM solutions.                                      |
| **Analysis & Reflection**                 | Insightful commentary on LLM vs. human approach; addresses potential pitfalls, advantages, or reliability concerns of AI solutions.         | Reflection on how LLMs performed is present, though somewhat general.                                                          | Lacks depth in analyzing why LLM solutions differ; minimal reflection on future implications.                        | No meaningful reflection on LLM usage or how it compares to human problem-solving.                                  |
| **Code Organization & Clarity**           | Code (all three solutions) is logically structured, easy to read, and thoroughly commented.                                                 | Code is mostly organized and understandable, with moderate commenting.                                                         | Some confusion in code structure; minimal comments hamper readability.                                                | Code is disorganized, poorly commented, or unreadable.                                                              |
| **Report Completeness & Quality**         | Report fully covers the instructions (problem statement, approach, results, comparison) with coherent writing and no major errors.          | Covers required sections clearly, with only minor omissions; writing is largely coherent.                                      | Partially complete; misses significant details or lacks clarity.                                                      | Major portions of the report missing or unclear; writing errors significantly impede understanding.                |

---

### **Outcome**

By completing this assignment, you will:

- Practice **algorithmic problem-solving** and **data structure** concepts.  
- Gain **hands-on experience** comparing AI-generated code with your own.  
- Develop a more nuanced perspective of **LLM capabilities** and potential pitfalls in coding tasks.

**Submit** your final code (three solutions) and **short report** by the due date.
