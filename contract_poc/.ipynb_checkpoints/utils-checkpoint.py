import pypdf

import asyncio

import pandas as pd
from lamini.generation.base_prompt_object import PromptObject

def read_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        num_pages = len(reader.pages)
        text = {}
        for idx, page in enumerate(reader.pages):
            text[f"Page {idx+1}"] = page.extract_text()
    return text

def build_prompts_from_dataframe(
    df: pd.DataFrame, 
    input_col: str = "Question", 
    output_col: str = "Direct Text with Answer",
    content_col: str = "Page Content",
    company_col: str = "Text Source"
):
    for idx, row in df.iterrows():
        yield PromptObject(
            prompt = row[input_col],
            data = {
                "expected_output": row[output_col],
                "page_content": row[content_col],
                "company": row[company_col].split(" - ")[0]
            }
        )
    