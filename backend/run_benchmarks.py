import pandas as pd
from typing import List, Dict

class realisticBenchmarkRunner:
    def get_benchmark_results(self) -> Dict[str, pd.DataFrame]:
        # Data provided by the user reflecting realistic MedBioRAG performance
        data = [
            ["LiveQA", "FT-GPT4o", 24.12, 6.18, 13.31, 1.63, 1.10, -46.48],
            ["LiveQA", "FT-GPT4o+MedBioRAG", 15.73, 4.58, 10.74, 1.20, 2.29, -86.99],
            ["LiveQA", "GPT4o", 26.96, 5.80, 13.42, 1.41, -2.93, -34.79],
            ["LiveQA", "GPT4o+MedBioRAG", 27.33, 6.39, 13.42, 15.29, -1.60, -29.99],
            
            ["MedicationQA", "FT-GPT4o", 24.69, 8.80, 17.61, 2.49, 8.98, -33.82],
            ["MedicationQA", "FT-GPT4o+MedBioRAG", 27.73, 15.09, 22.72, 7.24, 8.79, -33.63],
            ["MedicationQA", "GPT4o", 22.92, 13.69, 18.70, 7.89, 8.55, -6.92],
            ["MedicationQA", "GPT4o+MedBioRAG", 19.85, 4.20, 10.97, 0.98, -7.63, -33.21],
            
            ["PubMedQA", "FT-GPT4o", 35.82, 13.55, 26.09, 4.34, 35.33, -9.23],
            ["PubMedQA", "FT-GPT4o+MedBioRAG", 37.49, 14.78, 27.89, 6.11, 37.02, -3.89],
            ["PubMedQA", "GPT4o", 25.72, 9.02, 17.05, 2.48, 17.04, -9.04],
            ["PubMedQA", "GPT4o+MedBioRAG", 26.39, 9.55, 17.47, 2.73, 18.10, -7.86],
            
            ["BioASQ", "FT-GPT4o", 32.69, 16.84, 25.11, 6.52, 32.97, -2.41],
            ["BioASQ", "FT-GPT4o+MedBioRAG", 34.30, 18.81, 27.74, 6.12, 35.43, -15.44],
            ["BioASQ", "GPT4o", 13.97, 5.51, 10.08, 1.27, 0.22, -24.84],
            ["BioASQ", "GPT4o+MedBioRAG", 22.29, 8.21, 15.64, 2.27, 11.60, -12.50]
        ]
        
        columns = ["Dataset", "Model", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore", "BLEURT"]
        return pd.DataFrame(data, columns=columns)

    def display_analysis(self):
        df = self.get_benchmark_results()
        datasets = df["Dataset"].unique()
        
        print("\n" + "="*80)
        print("      REALISTIC BENCHMARK ANALYSIS: MEDBIORAG PERFORMANCE")
        print("="*80)
        
        for ds in datasets:
            print(f"\n--- {ds} ---")
            ds_df = df[df["Dataset"] == ds].drop(columns="Dataset")
            print(ds_df.to_string(index=False))
            
            # Highlight interesting deltas
            base_gpt = ds_df[ds_df["Model"] == "GPT4o"]
            rag_gpt = ds_df[ds_df["Model"] == "GPT4o+MedBioRAG"]
            
            if not base_gpt.empty and not rag_gpt.empty:
                bleu_gain = rag_gpt["BLEU"].values[0] - base_gpt["BLEU"].values[0]
                if bleu_gain > 5:
                    print(f"  [Insight] Massive BLEU jump on {ds} (+{bleu_gain:.2f}) - RAG providing exact terminology.")
                elif bleu_gain < -5:
                    print(f"  [Insight] Significant BLEU drop on {ds} ({bleu_gain:.2f}) - Context conflict or over-explanation.")

if __name__ == "__main__":
    runner = realisticBenchmarkRunner()
    runner.display_analysis()
