# Evaluation Metrics

In real world, the product development would feature iterative testings with the target audience and sprints to refine selection of connected data sources, best data processing and prompt layer.

### RAG Model

The performance of the model was evaluated using three methods:

1. Qualitative Test Questions

We defined seven representative user questions, which can be found in 06_doc/user_queries.md.
The model’s responses were reviewed through an iterative process in which source documents were refined and the prompt template was adjusted until satisfactory accuracy and clarity were achieved.
The resulting outputs from the last iteration is available in 07_testsdemo/test_outputs.

2. Relevance Score

A relevance score measures how well the retrieved documents align semantically with the user query. In our system, this score is calculated using FAISS’s cosine-similarity distance metric, which compares the query embedding with document embeddings. Cosine similarity returns a value between –1 and 1, with higher values indicating stronger semantic alignment.

For this project, out goal was to have a FAISS cosine-similarity score of 0.5 or higher.

The threshold set for the model was a similarity score of 0.3 to ensure that only the most semantically relevant documents were passed to the model for generation (The score was reduced to 0.15 for startup questions and 0.25 for patent questions as it was found these questions needed boosting).

3. Loading Time

We set a performance target requiring the system to load and respond in under 10 seconds, including document retrieval, model inference, and any post-processing steps.

### Predictive Model

Accuracy, qualitative plausability - 
    
F1 for classification (80 considered a good score) - Add definition and our score goal

Tbc for time-series analysis




