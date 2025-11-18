from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents
query = "What was the initial capital investment used to start Infosys?"

retriever = db.as_retriever(search_kwargs={"k": 5})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Synthetic Questions: 

# 1. "What was the first project undertaken by TCS?"
# 2. "Which airline did TCS develop the world’s first fully automated reservations system for?"
# 3. "What was the name of the first indigenous computer developed by HCL?"
# 4. "Who founded HCL?"
# 5. "What was the initial capital investment used to start Infosys?"
# 6. "Which Infosys co-founder later became the Chairman of the Unique Identification Authority of India (UIDAI)?"
# 7. "What was Wipro’s original line of business before entering IT services?"
# 8. "Which year did Wipro launch its first personal computer in India?"
# 9. "Which global telecom giant did Tech Mahindra acquire in 2009 to expand its IT services?"
# 10. "In what year did Tech Mahindra become a publicly listed company?"
# 11. "LTIMindtree was formed by the merger of which two companies?"
# 12. "In which year was Mindtree originally founded before merging with LTI?"