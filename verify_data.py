import os
import glob
import json
import re
from datasets import Dataset
import pypdf

SYSTEM_PROMPT = """You are a medical assistant. You will be provided with a patient case description and a clinical guideline document (PDF content).
Your task is to:
1. Identify relevant quotes from the guideline that apply to the case.
2. Formulate a recommended action based on the guideline and the case.
3. Output your reasoning, relevant quotes, and the final recommendation in the following XML format:

<analysis>
 <reasoning>
  Explain why the guideline applies to this patient.
 </reasoning>
 <relevant_quotes>
  <quote page="1">Exact text from document</quote>
 </relevant_quotes>
</analysis>
<recommended_action>
 The specific action to take.
</recommended_action>
"""

def extract_text_from_pdf(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {i+1} ---\n{page_text}"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def load_rag_dataset(data_dir):
    json_files = glob.glob(os.path.join(data_dir, "*_filtered.json"))
    data = []
    
    print(f"Found JSON files: {json_files}")
    for json_file in json_files:
        base_name = os.path.basename(json_file).replace("_filtered.json", "")
        pdf_files = glob.glob(os.path.join(data_dir, f"{base_name}*.pdf"))
        print(f"Checking for PDF for {base_name}: {pdf_files}")
        if not pdf_files:
            continue
        pdf_file = pdf_files[0]
        
        pdf_content = extract_text_from_pdf(pdf_file)
        if not pdf_content:
            print(f"Failed to extract text from {pdf_file}")
            continue
            
        with open(json_file, 'r') as f:
            cases = json.load(f)
            
        print(f"Loaded {len(cases)} cases from {json_file}")
        for i, case in enumerate(cases):
            user_content = f"Patient Description: {case['description']}\n\nClinical Guideline:\n{pdf_content[:500]}... [Truncated]"
            
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            data.append({
                "prompt": conversation, 
                "ground_truth_action": case.get('recommended_action', ''),
                "ground_truth_quotes": [r['quote'] for r in case.get('reference', [])],
                "pdf_content": pdf_content
            })
            if i == 0:
                print("--- Sample Prompt ---")
                print(user_content[:1000])
                print("--- End Sample ---")
            
    return Dataset.from_list(data)

if __name__ == "__main__":
    ds = load_rag_dataset("data")
    print(f"Total dataset size: {len(ds)}")
