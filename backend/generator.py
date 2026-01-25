import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

class AnswerGenerator:
    def __init__(self, api_key: str = None):
        # GitHub Models uses an OpenAI-compatible client
        # Endpoint: https://models.inference.ai.azure.com
        token = api_key or os.environ.get("GITHUB_TOKEN") or "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"
        
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=token,
        )
        self.model = "gpt-4o"

    def generate_answer(self, query: str, context: List[str], task_type: str = "long-form") -> str:
        context_str = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context)])
        
        if task_type == "mcq":
            prompt = f"""
            Answer the following Multiple Choice Question based ONLY on the provided context.
            
            Context:
            {context_str}
            
            Question:
            {query}
            
            Provide the letter of the correct answer and a brief justification.
            Use LaTeX formatting ($...$ for inline, $$...$$ for block) for any mathematical or chemical notations.
            """
            system_role = "You are a specialized biomedical assistant focused on high-accuracy multiple-choice answering. Use LaTeX for all scientific notations."
        elif task_type == "yes-no":
            prompt = f"""
            Answer the following Yes/No question based ONLY on the provided context.
            
            Context:
            {context_str}
            
            Question:
            {query}
            
            Answer 'Yes' or 'No' and provide a brief justification.
            Use LaTeX formatting ($...$ for inline, $$...$$ for block) for any mathematical or chemical notations.
            """
            system_role = "You are a specialized biomedical assistant focused on evidence-based Yes/No answers. Use LaTeX for all scientific notations."
        else: # long-form
            prompt = f"""
            Provide a structured, detailed response to the following query based on the provided documents.
            
            Context:
            {context_str}
            
            Query:
            {query}
            
            Ensure the response is detailed and follows a logical medical structure.
            STRICT REQUIREMENT: Use LaTeX ($...$ or $$...$$) for ALL mathematical, chemical formulas, and scientific units (e.g., $15-20\\text{ g}$, $H_2O$, $C_6H_{12}O_6$).
            """
            system_role = "You are a specialized biomedical assistant providing structured, professional medical explanations. You always use LaTeX for technical notations."
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
