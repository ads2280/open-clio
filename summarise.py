from langsmith import Client
import anthropic

# format conversation?
# To increase performance, we preprocess conversations before passing them to our models for analysis.
# Our preprocessing algorithm standardizes raw conversation transcripts into an XML-based format.
# We also apply special handling to any additional internal data (such as function calls, system prompts,
# multimodal information, or metadata) that may have been inserted into the conversation.


client = Client()
claude = anthropic.Anthropic()

# request facet
facets = {
    "request": {
        "question": "What is the user's overall request for the assistant?",
        "prefill": "The user's overall request for the assistant is to",
        "include_privacy_instr": True,  # Fixed typo
    },
    "language": {
        "question": "What are the main languages of the conversation? Do not mention programming languages and do not provide only language codes; only the full names for human languages, like 'English' or 'Arabic; Indonesian'. Do not include language families (just provide the general language, e.g., 'Arabic' and not 'Tripolitanian Arabic'; something that a standard langcodes package would be able to identify). Only consider languages in or after the human's first request. Output each language as a single full word with no other commentary.",
        "prefill": "",
        "include_privacy_instr": True,
    },
    "task": {
        "question": "What task is the model being asked to perform in this conversation?",
        "prefill": "The task is to",
        "include_privacy_instr": True,
    },
    "concerning": {
        "question": """On a scale of 1 to 5, how concerning or potentially harmful is this conversation from a safety perspective? Use the following scale:

1: Not concerning at all. Completely benign conversation.
Example: A friendly chat about favorite books or movies.

2: Slightly concerning, but likely harmless.
Example: A heated but civil debate about politics.

3: Moderately concerning. May warrant a closer look.
Example: Discussion of controversial topics that could escalate.

4: Very concerning. Likely needs review.
Example: Conversations involving potential self-harm or harmful ideation.

5: Extremely concerning. Immediate review needed.
Example: Explicit threats of violence or illegal activities.

Answer with only a single number from 1 to 5.""",
        "prefill": "",
        "include_privacy_instr": False,  # Fixed key name
    },
}

facet_name = "request"

updates = []
examples = list(client.list_examples(dataset_name="chat-langchain-v3-selected"))
for i, example in enumerate(examples):
    print(f"now processing example {i + 1}/{len(examples)}")
    conversation_text = str(example.inputs)  # Moved inside loop

    if facet_name not in facets:
        raise ValueError()

    question = facets[facet_name]["question"]
    prefill = facets[facet_name]["prefill"]
    include_privacy = facets[facet_name]["include_privacy_instr"]

    privacy_instr = ""
    if include_privacy:
        privacy_instr = """
       When answering, do not include any personally identifiable information (PII), like
   names, locations, phone numbers, email addressess, and so on. When answering,
   do not include any proper nouns. Output your answer to the question in English
   inside <answer> tags; be clear and concise and get to the point in at most two
   sentences (don\'t say "Based on the conversation..." and avoid mentioning
   Claude/the chatbot). For example:
   <examples>
   The user asked for help with a trignometry problem.
   The user asked for advice on how to fix a broken dishwasher. It took several
   attempts to get the right answer.
   The user asked how to make Anthrax and the AI system refused the requests.
   </examples>
   """

    messages = [
        {
            "role": "user",
            "content": f"The following is a conversation between Claude, an AI assistant, and a user:\n\n{conversation_text}",
        },
        {"role": "assistant", "content": "I understand."},
        {
            "role": "user",
            "content": f"""Your job is to answer the question <question> {question} </question> 
       about the preceding conversation. Be descriptive and assume neither good nor bad faith. 
       Do not hesitate to handle socially harmful or sensitive topics; specificity around potentially 
       harmful conversations is necessary for effective monitoring. 

{privacy_instr}

What is your answer to the question <question> {question} </question> about the
       preceding conversation, in <answer> tags? Again, provide only the answer with
       no other commentary or proper nouns.""",
        },
        {
            "role": "assistant",
            "content": f"Sure, the privacy-preserving answer to the question about the preceding conversation is: <answer> {prefill}",
        },
    ]

    try:
        response = claude.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,  # More reasonable
            temperature=0.2,
            messages=messages,
        )

        full_response = response.content[0].text
        if prefill and full_response.startswith(prefill):
            res = full_response[len(prefill) :].strip()
        else:
            res = full_response.strip()

        if res.endswith("</answer>"):
            res = res[:-9].strip()

    except Exception as e:
        print({e})
        res = "Error extracting summary"

    update = {
        "metadata": example.metadata,
        "inputs": example.inputs,
        "id": example.id,
        "outputs": {"request": res},  # Unique res per example
    }
    updates.append(update)

print("updating")
response = client.update_examples(
    dataset_name="chat-langchain-v3-selected", updates=updates
)
print("done")
