from langsmith import Client
import datetime


client = Client()
examples = list(client.list_examples(dataset_name="ds-unnatural-molecule-62",limit=200))
new_dataset = "chat-langchain-v3-selected"
client.create_dataset(dataset_name=new_dataset)

client.create_examples(
    dataset_name=new_dataset,
    examples=[
        {"inputs": {**example.inputs, **example.outputs}, "metadata": example.metadata} for example in examples
    ]
)

# clio applies special handling to data such as function calls, metadata, system prompts
# this isn't identical but it's similar -- we get rid of fields such as documents, steps, etc.
new_examples = list(client.list_examples(dataset_name=new_dataset))
for example in new_examples:
    # remove documents 
    inputs = example.inputs.copy()
    if "documents" in inputs:
        del inputs["documents"]
        print(f"Removed documents from example {example.id}")
    
    if "steps" in inputs and (not inputs["steps"] or inputs["steps"] == []):
       del inputs["steps"]
   
   # cleanup within the messages array
    if "messages" in inputs and isinstance(inputs["messages"], list):
        cleaned_messages = []
        for msg in inputs["messages"]:
            if isinstance(msg, dict):
                # Keep only essential fields
                clean_msg = {
                    "type": msg.get("type"),
                    "content": msg.get("content")
                }
       
        inputs["messages"] = cleaned_messages
    
    # update each item in LS
    client.update_example(
        example_id=example.id,
        inputs=inputs,
        outputs=example.outputs,
        metadata=example.metadata
    )
    # inputs only contain "query" and "answer" keys now