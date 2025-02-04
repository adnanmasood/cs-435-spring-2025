# Coding Problems

## **1. Candy**
**Problem Statement:**
- There are `n` children in a line, each with a rating given in an array `ratings`.
- You need to distribute candies based on the following rules:
  1. Every child must get **at least one** candy.
  2. A child with a **higher rating** than its neighbor must get **more candies** than the neighbor.
- The goal is to **minimize the total number of candies given**.

**Example Walkthrough:**
```plaintext
Input: ratings = [1,0,2]
Output: 5
```
Explanation:
- Start with `[1,1,1]` (each child gets one candy).
- Adjust according to ratings: The first child (1) gets 2 candies because `1 > 0`, the last child (2) also gets 2 candies.
- Final candy distribution: `[2,1,2]`, total = `5`.

**Approach:**
- Use a **greedy** two-pass algorithm:
  1. Left-to-right pass: Ensure children with higher ratings than the left neighbor get more candies.
  2. Right-to-left pass: Ensure children with higher ratings than the right neighbor get more candies while keeping the previous rule intact.

**Complexity Analysis:**
- Time Complexity: **O(n)** (two passes over the array)
- Space Complexity: **O(n)** (to store candies array)

---

## **2. Text Justification**
**Problem Statement:**
- Given a list of words and a maximum width `maxWidth`, format the text such that:
  1. Each line has exactly `maxWidth` characters.
  2. Words should be **left and right justified**.
  3. Extra spaces are distributed as evenly as possible.

**Example Walkthrough:**
```plaintext
Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]
```
- **First line:** `This is an` (with extra spaces to make it 16 characters).
- **Second line:** `example of text` (evenly spaced).
- **Last line:** `justification.` (left justified, no extra spaces between words).

**Approach:**
1. **Greedy word packing**: Fit as many words as possible in each line.
2. **Evenly distribute spaces** between words.
3. **Handle last line separately**: It should be **left justified**.

**Complexity Analysis:**
- Time Complexity: **O(n)** (processing words once).
- Space Complexity: **O(n)** (output storage).

---

## **3. H-Index**
**Problem Statement:**
- Given `citations[i]` for each research paper, return the maximum `h` such that `h` papers have **at least** `h` citations.

**Example Walkthrough:**
```plaintext
Input: citations = [3,0,6,1,5]
Output: 3
```
Explanation:
- Papers sorted: `[0,1,3,5,6]`
- The **third** paper (index 2) has `3` citations, and at least `3` papers have `≥3` citations.

**Approach:**
- **Sort citations** and find the largest `h` where `citations[i] ≥ h`.

**Complexity Analysis:**
- Time Complexity: **O(n log n)** (sorting step).
- Space Complexity: **O(1)**.

---

## **4. Best Time to Buy and Sell Stock III**
**Problem Statement:**
- You are given `prices[i]`, the stock price on the `i`th day.
- You may complete at most **two** transactions.
- **You must sell a stock before buying again.**

**Example Walkthrough:**
```plaintext
Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
```
Explanation:
- **First buy-sell:** Buy on day 4 (price `0`), sell on day 6 (price `3`), profit `3-0=3`.
- **Second buy-sell:** Buy on day 7 (price `1`), sell on day 8 (price `4`), profit `4-1=3`.
- **Total profit:** `3 + 3 = 6`.

**Approach:**
1. Track **first buy/sell** profit.
2. Track **second buy/sell** profit.
3. Use **dynamic programming** for efficient tracking.

**Complexity Analysis:**
- Time Complexity: **O(n)**.
- Space Complexity: **O(1)**.

---

## **5. Find Median from Data Stream**
**Problem Statement:**
- Implement `MedianFinder` to dynamically add numbers and get the median.

**Example Walkthrough:**
```plaintext
Input: ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
       [[], [1], [2], [], [3], []]
Output: [null, null, null, 1.5, null, 2.0]
```
Explanation:
- `[1]` → median `1`
- `[1,2]` → median `(1+2)/2 = 1.5`
- `[1,2,3]` → median `2`

**Approach:**
- **Two heaps**:
  - **Max heap** for the smaller half.
  - **Min heap** for the larger half.
- **Rebalance** heaps as new numbers arrive.

**Complexity Analysis:**
- **O(log n)** per insertion.
- **O(1)** for finding the median.

---

## **6. IPO**
**Problem Statement:**
- Choose at most `k` projects to **maximize capital**.
- Each project has a **profit** and a **minimum capital requirement**.

**Example Walkthrough:**
```plaintext
Input: k = 2, w = 0, profits = [1,2,3], capital = [0,1,1]
Output: 4
```
Explanation:
- Start with **w=0** → Choose project 0 (profit `1`).
- Now **w=1** → Choose project 2 (profit `3`).
- **Final capital:** `4`.

**Approach:**
1. **Sort projects by capital**.
2. **Use a max heap** to pick the most profitable project within available capital.

**Complexity Analysis:**
- **O(n log n)** (sorting and heap operations).

---

## **7. Median of Two Sorted Arrays**
**Problem Statement:**
- Find the **median** of two sorted arrays in **O(log (m+n))** time.

**Example Walkthrough:**
```plaintext
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
```
Merged array: `[1,2,3]`, median is `2`.

**Approach:**
- **Binary search** on the smaller array.
- **Partition arrays** into two halves such that left half ≤ right half.

**Complexity Analysis:**
- **O(log min(m,n))**.

---

## **8. Merge k Sorted Lists**
**Problem Statement:**
- Given `k` sorted linked lists, merge them into one sorted list.

**Example Walkthrough:**
```plaintext
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
```

**Approach:**
- **Use a min heap** to merge efficiently.

**Complexity Analysis:**
- **O(n log k)**.

---


## **9. Word Search II**
**Difficulty:** Hard  
**Problem Statement:**
- Given a `m x n` board of characters and a list of words, find all words that can be **constructed from adjacent cells**.
- You **cannot use the same letter cell more than once** per word.

### Example:
```plaintext
Input: board = [["o","a","a","n"],
                ["e","t","a","e"],
                ["i","h","k","r"],
                ["i","f","l","v"]], 
       words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
```

### Approach:
1. **Trie Construction**:
   - Store all words in a **Trie** to optimize search.
   
2. **Backtracking DFS**:
   - Start DFS search from every cell.
   - If a letter is found in Trie, continue searching.
   - Mark visited cells to avoid reuse.
   
3. **Optimization**:
   - Use Trie **prefix pruning** to discard unnecessary searches.

### Complexity Analysis:
- **O(N * 4^L)**, where `N` is total board cells and `L` is the longest word length.
- **Trie Construction**: O(WL), where `W` is word count and `L` is max word length.

---

## **10. Word Ladder**
**Difficulty:** Hard  
**Problem Statement:**
- Given `beginWord` and `endWord`, find the **shortest transformation sequence**.
- **Each transformed word** must exist in `wordList`.
- **Each step must change only one letter**.

### Example:
```plaintext
Input: beginWord = "hit", endWord = "cog", 
       wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
```
Explanation:
```
"hit" → "hot" → "dot" → "dog" → "cog"
```

### Approach:
1. **BFS (Breadth-First Search)**:
   - Start from `beginWord` and transform one letter at a time.
   - Use a queue and explore all words differing by **one character**.
   - Use a **set** to mark visited words.

### Complexity Analysis:
- **O(N × M × 26)**, where `N` is the number of words and `M` is the word length.

---

## **11. Binary Tree Maximum Path Sum**
**Difficulty:** Hard  
**Problem Statement:**
- Given a binary tree, find the **maximum path sum**.
- A path must **connect at least one node**.

### Example:
```plaintext
Input: root = [1,2,3]
Output: 6
```
Explanation:
- The path `2 → 1 → 3` gives max sum `6`.

### Approach:
1. **Recursive DFS Traversal**:
   - Compute max sum at each node considering left/right subtrees.
   - Use **global max variable** to track the highest sum found.

### Complexity Analysis:
- **O(N)** traversal of the tree.
- **O(log N)** space for recursive stack.

---

## **12. Reverse Nodes in k-Group**
**Difficulty:** Hard  
**Problem Statement:**
- Given a linked list, **reverse every k nodes**.
- If remaining nodes are **less than k**, leave them unchanged.

### Example:
```plaintext
Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]
```

### Approach:
1. **Iterate through list to find k nodes**.
2. **Reverse nodes in groups of k**.
3. **Recursively process remaining nodes**.

### Complexity Analysis:
- **O(N)** time to traverse list.
- **O(1)** space if done iteratively.

---

## **13. Basic Calculator**
**Difficulty:** Hard  
**Problem Statement:**
- Implement a basic calculator that supports **addition, subtraction, and parentheses**.
- You **cannot use `eval()`**.

### Example:
```plaintext
Input: s = "(1+(4+5+2)-3)+(6+8)"
Output: 23
```

### Approach:
1. **Use a Stack**:
   - Store numbers and operators.
   - Handle parentheses using recursion.
   - Use **+1 / -1** sign multipliers.

### Complexity Analysis:
- **O(N)** for single pass processing.
- **O(N)** space for stack storage.

---

## **14. Minimum Window Substring**
**Difficulty:** Hard  
**Problem Statement:**
- Given `s` and `t`, find the **smallest substring of `s`** that contains all characters of `t`.

### Example:
```plaintext
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
```

### Approach:
1. **Sliding Window**:
   - Expand window while missing characters exist.
   - Shrink window when all characters are found.
   - Keep track of **minimum substring**.

### Complexity Analysis:
- **O(N + M)** using a hashmap to track frequency.
- **O(1)** space.

---

## **15. Substring with Concatenation of All Words**
**Difficulty:** Hard  
**Problem Statement:**
- Find **all starting indices** of substrings in `s` that are a **concatenation of words** in `words[]` (any order).

### Example:
```plaintext
Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]
```

### Approach:
1. **Sliding Window**:
   - Traverse `s` with window size `k * word_length`.
   - Use hashmap to track word counts.

### Complexity Analysis:
- **O(N * k)**, where `N` is string length and `k` is word count.

---

## **Summary**
### **Techniques Used**
| Problem | Technique |
|---------|----------|
| Candy | Greedy |
| Text Justification | Greedy |
| H-Index | Sorting + Binary Search |
| Best Time to Buy and Sell Stock III | DP |
| Find Median from Data Stream | Heaps |
| IPO | Greedy + Heaps |
| Median of Two Sorted Arrays | Binary Search |
| Merge k Sorted Lists | Min Heap |
| Word Search II | Trie + DFS |
| Word Ladder | BFS |
| Binary Tree Max Path Sum | DFS |
| Reverse Nodes in k-Group | Linked List Manipulation |
| Basic Calculator | Stack |
| Minimum Window Substring | Sliding Window |
| Substring Concatenation | Sliding Window + HashMap |


