# ML Snippets: Retrieval Augmented Generation

### How to Use
1. Install Dependencies
   * conda install conda-forge::llama-cpp-python
   * pip install langchain-text-splitters
   * pip install qdrant-client
3. Download Llama_cpp compatible LLM
   * https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q2_K.gguf
4. Load input data from `inputs/source.txt`
   * `python import_data.py`
5. Query LLM
   * `python query_model`
