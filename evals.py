import pandas as pd
from langsmith import Client, wrappers
from typing import List, Dict, Any
import anthropic
from pydantic import BaseModel
from openevals.llm import create_llm_as_judge
from prompts import CATEGORY_RELEVANCE, HIERARCHICAL_FIT, BEST_FIT, EXCLUSIVE_FIT, DEDUPLICATE, PARTITIONS
import re


anthropic_client = wrappers.wrap_anthropic(anthropic.Anthropic())
client = Client()
dataset_name="ds-granular-pseudoscience-68"
examples = list(client.list_examples(dataset_name=dataset_name))
print(f"Loaded {len(examples)} examples from dataset '{dataset_name}")

combined_df = pd.read_csv('combined.csv')
example_to_category = dict(zip(combined_df['full_example'],combined_df['category']))
convo_to_cluster = dict(zip(combined_df['full_example'], combined_df['base_cluster_name']))
print(f"Loaded {len(example_to_category)} example->category mappings")
print(f"Loaded {len(convo_to_cluster)} convo->cluster mappings")

def category_relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate if conversation fits its assigned category"""
    summary = {reference_outputs["summary"]}
    category = {reference_outputs["category"]}
    category_relevance_prompt = CATEGORY_RELEVANCE.format(
        summary=summary, 
        category=category,
        partitions=PARTITIONS
    )

    evaluator = create_llm_as_judge(
        prompt=category_relevance_prompt,
        model="anthropic:claude-sonnet-4-20250514",
        feedback_key="category-relevance",
    )
    
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )
    return eval_result

def hierarchical_fit_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate if conversation properly belongs under its assigned base cluster"""
    
    hierarchical_fit_prompt = HIERARCHICAL_FIT.format(reference_outputs=reference_outputs)
    
    evaluator = create_llm_as_judge(
        prompt=hierarchical_fit_prompt,
        model="anthropic:claude-sonnet-4-20250514",
        feedback_key="base-cluster-relevance",
    )
    
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )
    return eval_result

def best_fit_evaluator(inputs: dict, outputs:dict, reference_outputs: dict):
    
    # there's prob a better way to do this w/ combined, uniq method
    clusters_df = pd.read_csv('level_0_clusters.csv')
    all_base_clusters = clusters_df["name"].tolist()
    all_base_clusters_text = "\n".join(all_base_clusters)

    summary = reference_outputs["summary"]
    current_cluster = reference_outputs["clustering"]["level_0"]["name"] # base only
    
    best_fit_prompt= BEST_FIT.format(summary=summary, current_cluster=current_cluster, all_base_clusters=all_base_clusters_text)

    evaluator = create_llm_as_judge(
        prompt=best_fit_prompt,
        model="anthropic:claude-sonnet-4-20250514",
        feedback_key="best-fit",
    )
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )
    return eval_result

# evaluating redundancy by asking claude to check if a convo reasonably fits in 2+ clusters
def exclusive_fit_evaluator(inputs: dict, outputs:dict, reference_outputs: dict):
    clusters_df = pd.read_csv('level_0_clusters.csv')
    all_base_clusters = clusters_df["name"].tolist()
    all_base_clusters_text = "\n".join([f"- {cluster}" for cluster in all_base_clusters])
    
    summary = reference_outputs["summary"]
    current_cluster = reference_outputs["clustering"]["level_0"]["name"] # base only
    
    redundancy_prompt = EXCLUSIVE_FIT.format(summary=summary, current_cluster=current_cluster, all_base_clusters_text=all_base_clusters_text)
    
    evaluator = create_llm_as_judge(
        prompt=redundancy_prompt,
        model="anthropic:claude-sonnet-4-20250514",
        feedback_key="exclusive-fit", 
    )
    
    eval_result = evaluator(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs
    )
    
    return eval_result

def unique_n_summary_evaluator(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    
    # get all unique clusters, same as best_fit_evaluator
    clusters_df = pd.read_csv('level_0_clusters.csv')
    all_base_clusters = clusters_df["name"].tolist()
    all_base_clusters_text = "\n".join([f"- {cluster}" for cluster in all_base_clusters])
    total_base_clusters = len(all_base_clusters)

    prompt = DEDUPLICATE.format(all_base_clusters_text=all_base_clusters_text, total_base_clusters=total_base_clusters)

    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text.strip()
        
        # extract final score,  "FINAL SCORE: X.XX"
        score = 0.0
        
        final_score_match = re.search(r'FINAL SCORE:\s*(\d*\.?\d+)', response_text, re.IGNORECASE)
        if final_score_match:
            score = float(final_score_match.group(1))
        else:
            score = -1
    except Exception as e:
        print({e})
    return {
        "key": "uniqueness_score",
        "score": float(score),
        "comment": response_text
    }
    
def dummy_target(inputs):
    return {}

print("Running xxxx Evaluation...")
category_relevance_eval = client.evaluate(
    dummy_target,
    data=client.list_examples(dataset_name="eval-test-unthread-data"),
    evaluators=[
        exclusive_fit_evaluator
    ],
    summary_evaluators=[
        unique_n_summary_evaluator
    ],
    experiment_prefix="something", 
    description="something",
    max_concurrency=2,
)
print("complete!")



#TODO - extend beyond base clusters for n,m, etc.
"""
{
  "summary": "debugging help with LangSmith SDK tracing for Python implementation",
  "category": "LangSmith product",
  "clustering": {
    "level_0": {
      "id": 5,
      "name": "Debug LangSmith Python SDK tracing integration errors"
    },
    "level_1": {
      "id": 2,
      "name": "Handle LangSmith SDK Integration and Tracing Issues"
    }
  }
}
"""