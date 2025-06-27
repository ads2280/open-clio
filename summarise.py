from langsmith import Client
import anthropic

#updated for new ds
#def use claude b/c openai context window too short

# format conversation?
# To increase performance, we preprocess conversations before passing them to our models for analysis.
# Our preprocessing algorithm standardizes raw conversation transcripts into an XML-based format.
# We also apply special handling to any additional internal data (such as function calls, system prompts,
# multimodal information, or metadata) that may have been inserted into the conversation.


client = Client()
# anthropic client will be initialized automatically using ANTHROPIC_API_KEY environment variable
claude = anthropic.Anthropic()

# request facet
facets = {
    "request": {
        "question": "What is the user's overall LangChain/LangSmith support request?",
        "prefill": "The user's overall request for the assistant is to",
        "include_privacy_instr": True,  
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
        "include_privacy_instr": False,  
    },
}

facet_name = "request"

updates = []
examples = list(client.list_examples(dataset_name="unthread-data"))
for i, example in enumerate(examples):
    print(f"now processing example {i + 1}/{len(examples)}")
    conversation_text = str(example.inputs)  

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
   the AI system/chatbot). For example:
   <examples>
   The user asked for help with LangSmith tracing integration issues.
   The user asked for assistance with LangGraph platform deployment configuration. It took several
   attempts to resolve the authentication errors.
   The user requested billing information about subscription upgrades and the support agent provided pricing details.
   </examples>
   """

    messages = [
        {
            "role": "user",
            "content": f"The following is a conversation between an AI assistant and a user:\n\n{conversation_text}",
        },
        {"role": "assistant", "content": "I understand."},
        {
            "role": "user",
            "content": f"""our job is to answer the question <question> {question} </question> 
       about the preceding support conversation. Be descriptive and determine the user's specific LangChain support request. 
You are helping to categorize LangChain support interactions to improve support operations, 
identify common customer issues, and enhance service quality within the LangChain ecosystem.

Focus specifically on requests related to:

**Admin Support:**
- Billing & Refunds (plan changes, subscription issues, payment problems)
- Authentication & Access (login issues, password resets, account access)
- General Account Management (adding users, migrating projects/data, org settings)
- Data Deletion (GDPR requests, account removal, data export)
- Security, Privacy and Compliance (SOC 2, HIPAA, DPA requests, security disclosures)

**LangChain OSS Support:**
- OSS Python (LangChain Python library questions, implementation, debugging)
- OSS JavaScript (LangChain JavaScript library questions, implementation, debugging)  
- Other OSS components (LangMem and other open-source LangChain tools)

**LangSmith Support:**
- Evaluation (application scoring, performance measurement, human feedback)
- Dashboards (web application UI issues, search functionality, app problems)
- Annotations (annotation queues, human feedback workflows, review processes)
- Datasets & Experiments (dataset management, structured evaluations, testing)
- Playground (LLM playground functionality, testing interface)
- SDK (tracing integration, API usage, SDK implementation)
- Prompt Hub (prompt management, version control, collaboration)
- Automation Rules (triggered actions, webhooks, automated workflows)
- Observability (trace analysis, metrics, dashboards, alerts configuration)
- Pricing (pre-sales questions, cost estimation, billing optimization)
- Administration (org-level settings, user management, permissions)

**LangGraph Support:**
- LangGraph Platform (managed cloud platform, runtime issues, deployment, scaling, infrastructure)
- OSS Python (LangGraph Python library implementation, agent development)
- OSS JavaScript (LangGraph JavaScript library implementation, agent development)
- Studio (desktop application, graph visualization, development tools)
- Pricing (platform pricing questions, cost estimation, billing)

**Other Support:**
- Sales (self-hosting inquiries, enterprise demos, new business requests)
- Spam (unsolicited emails, marketing, non-technical inquiries)

{privacy_instr}

What is your answer to the question <question> {question} </question> about the
       preceding support conversation, in <answer> tags? Again, provide only the answer with
       no other commentary or proper nouns.""",
        },
        {
            "role": "assistant",
            "content": f"Sure, the privacy-preserving answer to the question about the preceding conversation is: <answer> {prefill}",
        },
    ]

    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,  
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
    dataset_name="unthread-data", updates=updates
)
print("done")

# has what