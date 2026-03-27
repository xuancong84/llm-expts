#!/usr/bin/env python

# pip install docling

doc_converter = None
def convert_doc(pdf_path, output_type="markdown"):
	global doc_converter
	if doc_converter is None:
		from docling.document_converter import DocumentConverter
		doc_converter = DocumentConverter()
	doc = doc_converter.convert(pdf_path).document
	if output_type == "markdown":
		return doc.export_to_markdown()
	elif output_type == "text":
		return doc.export_to_text()
	elif output_type == "html":
		return doc.export_to_html()
	else:
		return doc

pdf_reader = None
def convert_pdf2txt(pdf_path):
	global pdf_reader
	if pdf_reader is None:
		import pypdf
		pdf_reader = pypdf.PdfReader(pdf_path)
	try:
		text = ""
		for i, page in enumerate(pdf_reader.pages):
			page_text = page.extract_text()
			if page_text:
				text += f"\n--- Page {i+1} ---\n{page_text}"
		return text
	except Exception as e:
		print(f"Error reading {pdf_path}: {e}")
		return ""

if __name__ == "__main__":
	txt = convert_doc("/home/xuancong/projects/llm-expts/data/acg-hypertension_15dec2023.pdf")
	print(txt)
	txt = convert_pdf2txt("/home/xuancong/projects/llm-expts/data/acg-hypertension_15dec2023.pdf")
	print(txt)