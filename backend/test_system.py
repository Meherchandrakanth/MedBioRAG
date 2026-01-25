from pipeline import MedBioRAGPipeline
import os

def test_medbiorag():
    documents = [
        "A 29-year-old man comes into his primary care physician's office with symptoms of social avoidance and detachment. Diagnoses often include Schizoid or Schizotypal personality disorders.",
        "Mitochondria are often referred to as the powerhouses of the cell. They play a crucial role in energy production through oxidative phosphorylation.",
        "Insulin is a hormone that helps glucose get into your cells to give them energy. Without insulin, too much sugar stays in your blood.",
        "Schizoid personality disorder is characterized by a lack of interest in social relationships and a tendency towards a solitary lifestyle.",
        "Recent studies suggest that mitochondria play a role in programmed cell death (apoptosis) in various plant species, including the lace plant."
    ]

    print("--- Initializing Pipeline ---")
    # API key should be set in environment variables
    api_key = os.environ.get("GITHUB_TOKEN")
    if not api_key:
        print("Warning: GITHUB_TOKEN not found in environment. Generation will fail.")
        # We can't run the actual LLM without the key, but we can verify the retrieval part.
    
    pipeline = MedBioRAGPipeline(documents, github_token=api_key)

    # Test Cases
    test_cases = [
        {
            "query": "A 29-year-old man comes into his primary care physician's ... A) Avoidant B) Schizoid C) Schizotypal D) Paranoid E) Dependent",
            "type": "mcq"
        },
        {
            "query": "Do mitochondria play a role in modelling lace plant leaves during development?",
            "type": "yes-no"
        },
        {
            "query": "How does insulin work?",
            "type": "long-form"
        }
    ]

    for tc in test_cases:
        print(f"\n{'='*20}")
        print(f"Testing Query: {tc['query'][:70]}...")
        print(f"{'='*20}")
        
        # 1. Show Retrieval Results (Re-ranked)
        retrieved_docs = pipeline.hybrid_retriever.retrieve(tc['query'], k=2)
        print("\n[Top Retrieved Documents (Re-ranked)]:")
        for i, doc in enumerate(retrieved_docs):
            print(f"{i+1}. Score ({doc['score']:.4f}): {doc['content'][:150]}...")
            
        # 2. Show Generation Results
        if api_key:
            response = pipeline.run(tc['query'], task_type=tc['type'])
            print(f"\n[Generated Answer]:\n{response}")
        else:
            print("\n[Skipping Generation - No API Key]")

if __name__ == "__main__":
    test_medbiorag()
