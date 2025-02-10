from PyPDF2 import PdfReader
import os
import re
import unicodedata
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List


# Download sentence tokenizer
# nltk.download('punkt')
# nltk.download('punkt_tab')

def extract_text_with_pyPDF(PDF_File):
    pdf_reader = PdfReader(PDF_File)
    raw_text = ''
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # os.makedirs("extract", exist_ok=True)
    # base_name = os.path.splitext(os.path.basename(PDF_File))[0]
    # txt_path = os.path.join("extract", f"{base_name}.txt")
    # with open(txt_path, "w", encoding="utf-8") as f:
    #     f.write(raw_text)

    return raw_text


def clean_extracted_text(text):
    # 1. Normalize Unicode (fix special characters like “ ” → " ")
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove weird/non-ASCII characters (e.g., , �, etc.)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Keeps only ASCII characters

    # 3. Remove extra whitespaces & newlines
    text = re.sub(r'\s+', ' ', text).strip()

    # 4. Remove page numbers (e.g., "Page 1 of 20" or "Page 5")
    text = re.sub(r'Page\s+\d+(\s*of\s*\d+)?', '', text, flags=re.IGNORECASE)

    # 5. Remove table lines (e.g., "---------" or "=======")
    text = re.sub(r'[-=]{3,}', '', text)

    # 6. Remove references section (academic papers)
    text = re.sub(r'References\s*\n.*', '', text, flags=re.DOTALL | re.IGNORECASE)

    # 7. Merge broken sentences (joins lines that don't end with a punctuation mark)
    text = re.sub(r'(?<![\.\?!])\n(?!\n)', ' ', text)

    # 8. Preserve paragraph breaks (keeps \n\n)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # os.makedirs("extract", exist_ok=True)
    # base_name = os.path.splitext(os.path.basename("cleaned_text"))[0]
    # txt_path = os.path.join("extract", f"{base_name}.txt")
    # with open(txt_path, "w", encoding="utf-8") as f:
    #     f.write(text)

    return text


def create_chunks(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    # 1. Sentence-based (Semantic) Splitting using NLTK
    sentences = nltk.sent_tokenize(text)

    # 2. Convert list of sentences back into a single text block
    processed_text = " ".join(sentences)

    # 3. Apply LangChain's Recursive Character Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Uses character count (can be adjusted for tokens)
        separators=["\n\n", "\n", ". ", " "]  # Tries to split by meaningful boundaries
    )

    chunks = text_splitter.split_text(processed_text)

    return chunks


text_with_pyPDF = extract_text_with_pyPDF("doc_01.pdf")
cleaned_text = clean_extracted_text(text_with_pyPDF)

# Create chunks using LangChain's Recursive Splitter
chunks = create_chunks(cleaned_text, chunk_size=200, chunk_overlap=20)

# Print chunks
for idx, chunk in enumerate(chunks):
    print(f"Chunk {idx + 1}: {chunk}\n")
