## Simple conversational chatbot in a Colab notebook.

1. **Read a PDF document** and extract its text.
2. **Split (chunk) the text** into smaller sections.
3. **Generate embeddings** for each chunk using the OpenAI Embeddings API (e.g., `text-embedding-ada-002`).
4. **Store the embeddings** in an in-memory vector database using [FAISS](https://github.com/facebookresearch/faiss).
5. **Perform vector similarity search** to retrieve relevant chunks in response to a user query.
6. **Use a re-ranking step** (through GPT-4 or another OpenAI completion model) to refine the top results.
7. **Formulate a conversational answer** by passing the top chunk(s) to GPT-4 to generate a response.

**Restrictions and Requirements:**

- No external tools besides:
  - [OpenAI Python library](https://pypi.org/project/openai/) for calling GPT-4 and embeddings endpoints.
  - [faiss-cpu](https://pypi.org/project/faiss-cpu/) for building the local vector database.
  - [PyPDF2](https://pypi.org/project/PyPDF2/) (or a similar pure-Python PDF library) for reading the PDF.
- Everything must be in memory with minimal or no dependencies beyond the libraries above.
- The entire workflow must be feasible in a single Google Colab notebook.
- You will need an **OpenAI API key** that has access to GPT-4 (or the relevant model you intend to use). Please set this key as an environment variable or directly in the notebook (though never commit it to a public repo).

---

## Assignment Instructions

### 1. Set Up Your Colab Environment

1. **Open a new Google Colab notebook.**
2. Install the necessary packages:

   ```bash
   !pip install openai faiss-cpu PyPDF2
   ```

3. **Import** the required libraries in your notebook:

   ```python
   import os
   import openai
   import faiss
   import numpy as np
   import PyPDF2
   ```

4. **Set your OpenAI API key:**

   ```python
   openai.api_key = "YOUR_OPENAI_API_KEY"
   ```
   Replace `"YOUR_OPENAI_API_KEY"` with your actual API key.

---

### 2. Prepare the PDF for Embedding

1. **Upload a PDF file** to your Colab environment (for instance, `sample.pdf`).
2. **Extract the text** from the PDF using PyPDF2:

   ```python
   def extract_text_from_pdf(pdf_path):
       text = ""
       with open(pdf_path, 'rb') as f:
           pdf_reader = PyPDF2.PdfReader(f)
           for page_num in range(len(pdf_reader.pages)):
               page = pdf_reader.pages[page_num]
               text += page.extract_text() + "\n"
       return text

   pdf_path = "sample.pdf"  # Replace with the filename you uploaded
   raw_text = extract_text_from_pdf(pdf_path)
   print(f"Total length of text extracted: {len(raw_text)} characters")
   ```

---

### 3. Text Preprocessing and Chunking

- Large text passages can lead to lengthy embedding calls, and we also need chunking for efficient retrieval.

1. **Define a simple chunking function** that:
   - Breaks the text into smaller pieces (e.g., ~500 tokens or ~500-1000 characters each).
   - Ensures we do not split in the middle of sentences if possible (this can be optional and as detailed as you like).

   ```python
   import re

   def chunk_text(text, chunk_size=500, overlap=50):
       """
       Splits text into overlapping chunks of chunk_size characters.
       Overlap ensures continuity between chunks.
       """
       # Simple approach: split by whitespace or just slice in fixed sizes
       words = text.split()
       chunks = []
       current_chunk = []

       current_length = 0
       for word in words:
           if current_length + len(word) + 1 > chunk_size:
               # join current chunk
               chunks.append(" ".join(current_chunk))
               # create overlap
               overlap_words = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
               current_chunk = overlap_words[:]
               current_length = sum(len(w) for w in current_chunk) + len(current_chunk) - 1
           current_chunk.append(word)
           current_length += len(word) + 1

       # Add the last chunk
       if current_chunk:
           chunks.append(" ".join(current_chunk))

       return chunks

   chunks = chunk_text(raw_text, chunk_size=500, overlap=50)
   print(f"Number of chunks created: {len(chunks)}")
   ```

   - Feel free to adjust `chunk_size` and `overlap` to optimize performance or maintain better context.

---

### 4. Generate Embeddings and Store in FAISS

1. **Embed each chunk** using the OpenAI Embeddings endpoint (e.g., `text-embedding-ada-002`).

   ```python
   EMBEDDING_MODEL = "text-embedding-ada-002"

   def get_embedding(text, model=EMBEDDING_MODEL):
       """
       Get the embedding from OpenAI API for a given text.
       """
       response = openai.Embedding.create(
           input=[text],
           model=model
       )
       embedding = response['data'][0]['embedding']
       return np.array(embedding, dtype=np.float32)
   ```

2. **Build a FAISS index** and add all chunk embeddings:

   ```python
   # Prepare the index
   dimension = 1536  # Dimension of text-embedding-ada-002
   index = faiss.IndexFlatL2(dimension)

   # List to store the original text chunks (so we can retrieve them)
   chunk_texts = []

   # List to store embeddings
   all_embeddings = []

   for i, chunk in enumerate(chunks):
       if chunk.strip() == "":
           continue
       embedding = get_embedding(chunk)
       all_embeddings.append(embedding)
       chunk_texts.append(chunk)

   all_embeddings = np.vstack(all_embeddings)
   index.add(all_embeddings)
   print(f"Number of vectors in the FAISS index: {index.ntotal}")
   ```

---

### 5. Define a Search and Re-Rank Function

We’ll create two main functions:
1. **`search_index(query, k=5)`:** 
   - Embeds the user query.
   - Searches the FAISS index for the `k` most similar chunks.
   - Returns the top `k` chunks (and their similarity distances).

2. **`rerank_chunks(query, chunks):** 
   - Uses GPT (GPT-4 or another completion model) to re-rank the retrieved chunks based on their relevance to the query. 
   - Returns the chunks in a sorted order (most relevant first).

#### 5.1. Vector Similarity Search

```python
def search_index(query, k=5):
    """
    Embeds the query, searches the FAISS index,
    and returns the top k most similar chunks.
    """
    query_embedding = get_embedding(query)
    # Convert query embedding to shape (1, dimension) for Faiss
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((dist, chunk_texts[idx]))
    return results
```

#### 5.2. Re-Ranking with GPT

To re-rank, we can feed GPT a prompt containing the query and the list of retrieved chunks and ask GPT to order them by relevance. GPT can return the chunks in ranked order.

> **Note**: This approach uses tokens for each chunk and might become expensive if there are many long chunks. Adjust chunk sizes or prompt strategy accordingly.

```python
def rerank_chunks(query, chunks):
    """
    Given a query and a list of (distance, chunk_text) pairs,
    ask GPT to re-rank them based on relevance to the query.
    Returns the chunks in the new sorted order.
    """
    # Create a prompt with the candidate chunks
    # You can tune this prompt to match your desired style or behavior.
    prompt = (
        f"You are given a user query: {query}\n"
        f"Below are text chunks retrieved from a knowledge base. "
        f"Rank them by their relevance to the query, from most relevant to least relevant.\n\n"
    )

    for i, (dist, chunk) in enumerate(chunks, start=1):
        # Distances from Faiss are for reference. GPT doesn't have to rely on them;
        # it's basically a textual analysis. But you could provide them if you wish.
        prompt += f"Chunk #{i}: {chunk[:400]}...\n\n"

    prompt += (
        "Please output the chunk numbers in descending order of relevance. "
        "Include a brief explanation for each rank.\n"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",  # or use "gpt-3.5-turbo" if you don't have GPT-4 access
        messages=[
            {"role": "system", "content": "You are a helpful assistant that re-ranks text chunks for a query."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    ranked_output = response["choices"][0]["message"]["content"]
    # Now, parse the GPT output to extract a ranking of chunk indices
    # This depends heavily on how GPT responds. 
    # For a simpler approach, you can skip re-ranking or adopt a more direct numeric extraction strategy.
    # 
    # Example strategy: Look for lines containing "Chunk #X"
    # or a list. This is not guaranteed robust, but for demonstration:
    
    import re
    pattern = r"Chunk\s*#(\d+)"
    found_indices = re.findall(pattern, ranked_output)
    found_indices = [int(idx) - 1 for idx in found_indices if idx.isdigit()]  # convert to 0-based

    # Reorder the chunks according to GPT's found order (if some chunks are not mentioned, keep them last)
    # fallback: keep original order if GPT doesn't mention them
    index_to_chunk = {i: (dist, chunk) for i, (dist, chunk) in enumerate(chunks)}
    new_order = []
    seen = set()
    for idx in found_indices:
        if idx in index_to_chunk and idx not in seen:
            new_order.append(index_to_chunk[idx])
            seen.add(idx)

    # Add any chunks GPT didn't rank at the end
    for i, (dist, chunk) in enumerate(chunks):
        if i not in seen:
            new_order.append((dist, chunk))

    return new_order
```

---

### 6. Conversation Loop With GPT-4 for Answers

Finally, we want a simple conversation function:
1. **Takes a user query**.
2. **Performs vector search** to get the top matches.
3. **Optionally re-ranks** the top matches using GPT.
4. **Passes the top chunk(s)** into GPT-4 to generate a final answer.

```python
def generate_answer(query, k=5, use_reranking=True):
    # Step 1: Retrieve top k chunks
    initial_results = search_index(query, k)
    
    # Step 2: Re-rank them if needed
    if use_reranking:
        ranked_results = rerank_chunks(query, initial_results)
    else:
        ranked_results = initial_results
    
    # Step 3: Let's take the top result from the re-ranked list 
    # (or you could combine top few results)
    _, top_chunk = ranked_results[0]
    
    # Step 4: Ask GPT-4 to answer the query using the top chunk
    prompt = (
        f"You are a chatbot that answers questions based on provided context.\n"
        f"Context:\n{top_chunk}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    answer = response["choices"][0]["message"]["content"]
    return answer


# Simple test
query_example = "What is the main topic of the PDF?"
answer_example = generate_answer(query_example, k=5, use_reranking=True)
print("Q:", query_example)
print("A:", answer_example)
```

---

### 7. Build an Interactive Chat Loop (Optional)

You can keep prompting the model inside a loop in Colab:

```python
def chat_loop():
    print("Enter 'exit' to quit.")
    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break
        
        answer = generate_answer(query)
        print("Assistant:", answer)

# Uncomment to run an interactive loop in a Colab environment
# chat_loop()
```

**Note**: In Colab, `input()` will work if you run the cell manually. For a more polished UI, you could adapt to an IPython widget or a streamlit interface, but that would rely on additional external tools. 

---

## Submission and Further Exploration

1. **Try different values** for `k` (the number of chunks to retrieve initially). Observe how it affects the quality of answers.
2. **Experiment** with the chunking strategy to see if smaller or larger chunks give better results.
3. **Modify the re-ranking approach** to pass more or fewer chunks to GPT. Possibly embed the entire snippet of top chunks into the final prompt for a more comprehensive answer.
4. **Analyze token usage**. Keep an eye on cost and performance, especially if your PDF is large. 
5. **Discuss** the trade-offs between local search + re-ranking vs. more advanced indexing and retrieval strategies.

**Deliverables**:

1. **A completed Colab notebook** containing:
   - All the code cells from above (or your improved versions).
   - At least one example PDF (small enough to share or an open-source PDF).
   - Demonstrations of the chatbot answering queries accurately.
2. **A short written discussion** of your design choices, any errors you encountered, and how you solved them.

---

## Conclusion

This assignment demonstrates how to build a **self-contained vector-search chatbot** using FAISS, OpenAI Embeddings, and GPT-4 for answering questions on a PDF’s content. By refining chunk sizes, retrieval strategies, and re-ranking prompts, you can significantly improve the quality of answers. This pattern is a powerful approach for domain-specific question answering without needing an entire external stack beyond Python, FAISS, and OpenAI’s API. 

**Good luck, and have fun experimenting!**
