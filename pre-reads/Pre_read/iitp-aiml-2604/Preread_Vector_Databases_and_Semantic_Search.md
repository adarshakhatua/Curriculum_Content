# Pre-read: Vector Databases & Semantic Search

## Context of This Session in the Course

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 45, "rankSpacing": 65, "diagramPadding": 20}} }%%
flowchart TB
  subgraph Foundation["Foundation Built So Far"]
    P1(["Previous Module<br/><b>Module 1</b><br/><i>(NumPy/Pandas, Scikit-learn)</i><br/>Python, data wrangling, and classical ML"])
    P2(["Previous Module<br/><b>Module 2</b><br/><i>(PyTorch, XGBoost)</i><br/>ML algorithms through neural networks"])
    C1[["Current Module Until Previous Session<br/><b>Module 3</b><br/><i>(CNNs, Transformers/LLMs)</i><br/>CV, NLP, Transformers, and LLM fine-tuning"]]
  end

  CS{{Current Session<br/><b>Vector Databases &amp; Semantic Search</b><br/><i>From keywords to meaning</i><br/>Embedding models · Cosine similarity · ChromaDB · FAISS}})

  subgraph Value["Why This Matters"]
    CV(["Course Value<br/><b>Bridges ML to production retrieval</b><br/>Vector search connects deep learning to<br/>grounded retrieval, powering RAG and AI agents."])
    RV(["Real-Life Value<br/><b>Semantic search in production apps</b><br/>Enables document search, Q&amp;A systems,<br/>and AI-powered recommendation engines."])
  end

  subgraph Future["Where This Leads Next"]
    F1(["Upcoming<br/><b>RAG, Agents &amp; Gen AI</b><br/><i>(LangChain, AI Agents, Diffusion)</i><br/>RAG pipelines, AI agents, and generative models"])
  end

  P1 ==>|&nbsp;Foundation&nbsp;| P2
  P2 ==>|&nbsp;ML to DL&nbsp;| C1
  C1 ==>|&nbsp;Vector Search&nbsp;| CS
  CS ==>|&nbsp;Course Path&nbsp;| CV
  CS ==>|&nbsp;Real-Life Use&nbsp;| RV
  CV ==>|&nbsp;Build&nbsp;| F1

  classDef previous fill:#eef4ff,stroke:#4f7ccf,color:#111827,stroke-width:2px;
  classDef current  fill:#fff4d6,stroke:#d9a321,color:#111827,stroke-width:3px;
  classDef value    fill:#eaf8ef,stroke:#3f9f63,color:#111827,stroke-width:2px;
  classDef future   fill:#f4ecff,stroke:#8a61d1,color:#111827,stroke-width:2px;

  class P1,P2,C1 previous;
  class CS current;
  class CV,RV value;
  class F1 future;
  linkStyle default stroke-width:3px
```

You search for "laptop for machine learning under 1000 dollars" on an e-commerce site, and it returns products titled "budget AI workstation," "entry-level deep learning rig," and "affordable data science notebook." None of those results contain the words "laptop," "machine learning," or "dollars" — yet every one of them is exactly what you were looking for. Meanwhile, the simple keyword search buried a dozen irrelevant results between you and these perfect matches.

This happens because the gap between what users say and what documents actually contain is wider than most search engines can handle. A user asking "how do I fix a flickering screen" and a support article titled "LCD panel replacement guide for XPS 15" share almost no words in common, yet a human instantly sees they are about the same problem. Traditional keyword search fails here because it treats language as a bag of literal tokens — it does not understand that "fix," "repair," "replace," and "troubleshoot" all point toward the same intent. The more your data grows, the more this gap costs in lost information, missed recommendations, and frustrated users.

That is where **vector databases and semantic search** become essential.

---

**What if** you could build a system that reads every document in your organisation — thousands of PDFs, internal wikis, support tickets, product manuals, and code repositories — and retrieves the most relevant information based on meaning rather than exact word matches? What if a healthcare researcher could search "patients with inflammatory response who did not respond to first-line treatment" and instantly find relevant clinical trial records that use completely different terminology? This is exactly what semantic search makes possible, and it is the foundational capability behind modern AI applications like RAG (Retrieval-Augmented Generation), AI agents with memory, and enterprise knowledge assistants. The vector database and search tools you will learn in this session are the engine that turns this from a hypothetical into a buildable reality.

---

At the heart of semantic search lies a deceptively simple idea: convert text into numbers, then measure the distance between those numbers. An **embedding model** from the **sentence-transformers** library takes any piece of text — a word, a sentence, or a whole document — and maps it to a dense vector of floating-point numbers, typically 384 to 1024 dimensions long. The magic is that semantically similar texts end up with vectors that point in similar directions in this high-dimensional space. To quantify that similarity, you use **cosine similarity**, which measures the angle between two vectors: a value of 1 means identical direction (highly similar), 0 means no relationship, and −1 means opposite meaning. But storing millions of these vectors and comparing every query against every document would be impossibly slow. That is where vector databases come in. **ChromaDB** provides a lightweight, developer-friendly way to create collections, add documents with their embeddings, and query by similarity. For larger-scale needs, **FAISS** (Facebook AI Similarity Search) implements fast approximate nearest neighbour search — trading a tiny amount of accuracy for massive speed gains, making it practical to search billion-scale datasets in milliseconds. Throughout this session, you will explore how these tools work together and, just as importantly, when **semantic search** outperforms **keyword search** and when it does not.

---

In the **previous session**, you explored **LLM concepts and responsible use** — pre-training and instruction tuning, RLHF, context windows, hallucination, and the capabilities and limitations of open-source LLMs like Llama, Mistral, and Gemma. You learned that LLMs have finite context windows, that they can hallucinate convincingly when they lack the right information, and that grounding responses in external knowledge is essential for reliability. That understanding now becomes a critical building block: vector databases and semantic search are precisely the tools that solve the grounding problem. They provide the external knowledge layer that LLMs need — storing documents as searchable vectors so that a model can retrieve relevant facts before generating an answer. Without this retrieval layer, even the most capable LLM is limited to whatever fits inside its context window and whatever it memorized during training. With it, the model gains access to your entire knowledge base. This session is the bridge from understanding LLM limitations to having the practical tools to overcome them, which directly sets the stage for **RAG** in the next session.

---

In this pre-read, you will discover:

- How to **apply** sentence-transformers to convert text into dense vector embeddings
- How to **interpret** cosine similarity scores for ranking search results by meaning
- How to **build** vector collections and queries using ChromaDB and FAISS
- How to **recognise** the tradeoffs between semantic search and traditional keyword search

---

## How Embeddings Turn Meaning into Numbers

The core challenge of search is that computers do not understand language the way humans do. To a computer, the sentence "The cat sat on the mat" is just a sequence of character codes. Even after tokenization, each word is a discrete symbol — "cat" and "kitten" share no inherent relationship in the computer's representation. **Embedding models** solve this by learning to map each piece of text to a dense vector in a continuous space where semantic relationships are encoded geometrically. The sentence-transformers library provides pre-trained models that have been fine-tuned to produce embeddings where similar texts are close together and dissimilar texts are far apart. For example, a well-trained embedding model would place "How do I return a product?" and "What is your refund policy?" near each other, even though they share almost no common words, because their meanings are similar. This is not keyword overlap — it is learned semantic understanding, distilled from training on massive text corpora with objectives that force the model to recognize paraphrases, synonyms, and thematic relationships.

## Why Cosine Similarity — and Why Not Keyword Match

Once you have embeddings, you need a way to compare them. **Cosine similarity** is the standard measure for this task because it focuses purely on direction, ignoring magnitude. Imagine each embedding vector as an arrow pointing from the origin into high-dimensional space. Two documents about different aspects of the same topic will produce arrows pointing in roughly the same direction (high cosine similarity), while one about an unrelated topic will point elsewhere (low cosine similarity). This is fundamentally different from **keyword search**, which relies on exact token overlap — typically using TF-IDF or BM25 scoring. Keyword search excels at finding documents containing specific terms (like a product code or a person's name), but it fails when the query and the document use different words to express the same idea. Semantic search, powered by embeddings and cosine similarity, excels at capturing intent and meaning but can miss exact matches that have no semantic proximity. The practical insight is that these approaches are complementary rather than competing: the most robust search systems combine both, using semantic search to cast a wide net and keyword search to pin down precise terms. Understanding this tradeoff — when meaning matters more than exact words and vice versa — is what separates a naive search implementation from a thoughtful one.

## Where Vector Search Appears in Real Life

Vector search has moved from research labs into production across nearly every industry that deals with unstructured text. **E-commerce** platforms use it for semantic product search — a user searching "comfy shoes for standing all day" retrieves results about ergonomic footwear and cushioned insoles, even when those exact phrases are absent from product descriptions. **Healthcare** organisations index medical literature, clinical trial reports, and patient records as embeddings, enabling researchers to find relevant studies using natural-language descriptions of symptoms, treatments, or outcomes rather than exact ICD codes or MeSH terms. **Legal technology** companies deploy vector databases for contract analysis and case law retrieval — a lawyer can search for "force majeure clauses related to supply chain disruptions in the Asia-Pacific region" and retrieve relevant clauses from thousands of contracts written in varied language. In **enterprise knowledge management**, companies use vector search to power internal FAQs, HR policy assistants, and developer documentation search, where the same question might be phrased dozens of different ways by different employees. And in **AI-powered development**, tools like GitHub Copilot and code-search systems use embedding models to find relevant code snippets and documentation based on intent — a developer searching "how to read a CSV file and handle missing values" gets pointed to the right library documentation and example code, regardless of the exact words used in the query.

## What's Next

After this session, you will be able to:

- Generate text embeddings using the sentence-transformers library and interpret what the vector dimensions represent
- Compute cosine similarity between document embeddings to rank search results by semantic relevance
- Create a ChromaDB collection, add documents with metadata, and execute similarity queries
- Build a FAISS index for fast approximate nearest neighbour search on large document collections
- Analyse the tradeoffs between semantic search and keyword search for different query types and domains

You do not need to build a production-grade search system right now. The goal is to recognise that semantic search transforms retrieval from exact-word matching to meaning-based discovery: **vector search is how you teach a computer to understand what you meant, not just what you typed.**

---

## Interesting Questions for the Live Session

- If cosine similarity measures only the angle between vectors and ignores their magnitude, what information are we discarding, and when might that loss hurt search quality?
- FAISS sacrifices exactness for speed through approximate nearest neighbour search — what applications can tolerate approximate results, and which use cases demand exact matches?
- How would you design a hybrid search system that combines semantic search for meaning with keyword search for exact terms to handle queries containing rare technical acronyms or product codes?
- What happens to your semantic search quality when you query with text from a specialized domain (e.g., medical or legal) that your embedding model was never trained on?

By the end of this session, vector search should feel less like a black-box algorithm and more like a practical system you can design and tune: **moving from keyword matching to meaning matching — one embedding at a time.**
