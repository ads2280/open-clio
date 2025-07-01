from langsmith import Client
from anthropic import AsyncAnthropic
from prompts import SUMMARIZE_INSTR
import time
from tqdm.asyncio import tqdm
import asyncio

#updated for new ds
#def use claude b/c openai context window too short

# format conversation?
# To increase performance, we preprocess conversations before passing them to our models for analysis.
# Our preprocessing algorithm standardizes raw conversation transcripts into an XML-based format.
# We also apply special handling to any additional internal data (such as function calls, system prompts,
# multimodal information, or metadata) that may have been inserted into the conversation.

# parallel
# want to see iterations per second 

client = Client()
claude = AsyncAnthropic()

dataset_name = "unthread-data"

async def process_example(example, semaphore, counter, total_examples):
    async with semaphore:
        conversation_text = str(example.inputs)

        messages = [
            {
                "role": "user",
                "content": f"The following is a conversation between an AI assistant and a user:\n\n{conversation_text}",
            },
            {
                "role": "assistant", 
                "content": "I understand."},
            {
                "role": "user",
                "content": f"{SUMMARIZE_INSTR}",
            },
            {
                "role": "assistant",
                "content": f"Sure, I'll analyze this conversation and provide a structured summary: <answer>User requested",
            },

        ]
        current_count = counter[0]
        counter[0] += 1
        print(f'processing example {current_count}/{total_examples} (ID: {example.id})')
        start_time = time.time()

        try:
            response = await claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                temperature=0.2,
                messages=messages
            )

            full_response = response.content[0].text
            # print(f'full response: {full_response}')

            # extract content between <answer> and </answer> tags
            if '</answer>' in full_response:
                # print(f'full response: {full_response}')
                end = full_response.find("</answer>")
                res = full_response[:end].strip()
            else:
                res = full_response.strip()


        except Exception as e:
            print(f'Error processing example {example.id}: {e}')
            res = "Error extracting summary"

        print(f'processed example {current_count}/{total_examples} (ID: {example.id}) in {time.time() - start_time:.2f}s')
        print(f'summary: {res}')

        return {
            "metadata": example.metadata,
            "inputs": example.inputs,
            "id": example.id,
            "outputs": {"request":res}
        }

async def main():
    # get all examples
    examples = list(client.list_examples(dataset_name=dataset_name))
    total_ex = len(examples)

    max_concurrent = 5
    semaphore = asyncio.Semaphore(max_concurrent)
    counter = [0]  # List to make it mutable across async functions

    print(f"processing {total_ex} conversations with {max_concurrent} concurrent requests...\n")

    start_time = time.time()
    tasks = [process_example(example, semaphore, counter, total_ex) for example in examples]
    updates = await asyncio.gather(*tasks) #each coroutine as separate task in list

    print(f'\nUpdating {dataset_name} dataset...')

    response = client.update_examples(
        dataset_name=dataset_name,
        updates=updates
    )

    # timing
    total_time = time.time() - start_time
    rate = total_ex / total_time if total_time > 0 else 0
    print(f'\n\nSuccess, summaries complete!')
    print(f'\n\nTiming Stats:')
    print(f'total time: {total_time:.2f}s')
    print(f'average {rate:.2f} iterations/second')

if __name__ == "__main__":
    asyncio.run(main())






