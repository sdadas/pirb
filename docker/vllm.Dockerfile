FROM vllm/vllm-openai:v0.14.0

# Install Python libraries
RUN pip install scikit-learn faiss-cpu datasets sentence-transformers
