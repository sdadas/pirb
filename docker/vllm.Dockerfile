FROM vllm/vllm-openai:v0.10.1.1

# Install Python libraries
RUN pip install scikit-learn faiss-cpu datasets sentence-transformers
