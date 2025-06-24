from langsmith import Client

client = Client()

examples = list(client.list_examples(dataset_name="chat-langchain-v3-selected", limit=3)) #check first 3

for i, example in enumerate(examples):
    print(f"Example {i+1}, Example ID: {example.id}")
    print(f"Inputs keys: {list(example.inputs.keys()) if example.inputs else 'None'}")
    print(f"Outputs: {example.outputs}")
    print(f"Outputs keys: {list(example.outputs.keys()) if example.outputs else 'None'}/n")
    
    # see preview of inputs
    if example.inputs:
        print(f"Sample inputs content: {str(example.inputs)[:200]}...")