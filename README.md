## Quick Notes

**Dataset Creation:**  
`create_dataset.py` and `summarise.py` made the [chat-langchain-v3-selected dataset](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/af1dd414-01ab-4a4e-8351-8a4ec86f7b85?tab=1) what it looks like now.

**Main Script:**  
`base_cluster.py` is the main clustering algorithm implementation. I'm now realizing I need to update the name because it's not just base clusters anymore. This is the **only file to run** to recreate results.

**Output Structure:**

- The `clustering_results/` directory contains each layer of summarization as a CSV file
- The `initial_results*.txt` files are just rough saves of my terminal output

## Usage

```python
python base_cluster.py
