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

# Fetch examples from the new dataset and update them with cleaned inputs
new_examples = list(client.list_examples(dataset_name=new_dataset))
for example in new_examples:
    # Clean the inputs by removing documents
    inputs = example.inputs.copy()
    
    # Fixed: Remove documents from the correct location
    if "documents" in inputs:
        del inputs["documents"]
        print(f"Removed documents from example {example.id}")
    
    if "steps" in inputs and (not inputs["steps"] or inputs["steps"] == []):
       del inputs["steps"]
   
   # Clean up messages array
    if "messages" in inputs and isinstance(inputs["messages"], list):
        cleaned_messages = []
        for msg in inputs["messages"]:
            if isinstance(msg, dict):
                # Keep only essential fields
                clean_msg = {
                    "type": msg.get("type"),
                    "content": msg.get("content")
                }
                # Only add if content exists and is meaningful
                if clean_msg["content"] and clean_msg["content"].strip():
                    cleaned_messages.append(clean_msg)
       
        inputs["messages"] = cleaned_messages
    
    # Update the example in the new dataset
    client.update_example(
        example_id=example.id,
        inputs=inputs,
        outputs=example.outputs,
        metadata=example.metadata
    )

print("Done cleaning up DATA!")


## TODO
# remove documents!!!

# cleaned up data e.g. for tool calling

#Removes empty steps arrays
#Strips messages down to just type and content
#Removes messages with empty/whitespace-only content
#Filters out all the metadata bloat (additional_kwargs, response_metadata, tool_calls, etc.)