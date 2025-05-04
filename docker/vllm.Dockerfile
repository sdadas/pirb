FROM vllm/vllm-openai:v0.8.5

# Install Python libraries
COPY requirements.txt requirements.txt
RUN pip install scikit-learn faiss-cpu datasets sentence-transformers
