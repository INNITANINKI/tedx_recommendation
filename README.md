# 🎤 TED Talks Search & Recommendation System

This is a **personal project** that allows users to **search and get recommendations for TED Talks** based on topics or talk descriptions. The system uses **Natural Language Processing (NLP)**, **sentence embeddings**, and **content-based recommendation** techniques to provide semantically relevant results.

---

## **Project Overview**

The TED Talks Search & Recommendation system provides:

1. **Search by Topic:**  
   Users can search TED Talks by keywords in the talk title, description, or presenter name.

2. **Recommendation by Description:**  
   Users can input a talk description or topic, and the system recommends **semantically similar TED Talks**.

3. **Filters & Sorting:**  
   - Filter talks by duration.  
   - Sort talks by relevance, newest, or oldest.

---

## **Key Features**

- Uses **SentenceTransformer** for semantic embeddings.  
- Calculates **cosine similarity** for content-based recommendations.  
- Implements **keyword-based search** for exact matches.  
- Precomputes embeddings and caches the model for **fast performance**. 

---

## **Tech Stack**

| Component           | Technology / Library |
|--------------------|--------------------|
| Programming Language | Python 3.9+ |
| Web Framework        | Streamlit |
| NLP / ML             | Sentence-Transformers, PyTorch, NumPy |
| Data Handling        | Pandas |

---


