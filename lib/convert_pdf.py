#!/usr/bin/env python3
# pip install docling

import re, html
from typing import List


# Matches list-marker prefixes: a) a. (a) iii). iv) 1) 1. (1) etc.
_LIST_MARKER_PREFIX_RE = re.compile(
	r'^\s*'
	r'(?:'
	r'\(?[ivxlcdmIVXLCDM]+[.)]+|'   # roman: iv) iv. (iv) iii).
	r'\(?[a-zA-Z][.)]+|'             # letter: a) a. (a) B.
	r'\(?\d+[.)]+|'                  # digit:  1) 1. (1) 2).
	r'[-*•]\s'                        # bullet: - * •
	r')'
)

def _starts_with_lowercase(line: str) -> bool:
	"""Return True if the line effectively begins with a lowercase letter.

	Allows at most one optional opener (quote, paren) before the first
	letter so that e.g. '"word' or '(word' are handled correctly.
	Lines like '<!-- image -->' are NOT matched because the first letter
	is buried inside markup, not at the start.
	AND the line does NOT start with a list marker.
	"""
	if _LIST_MARKER_PREFIX_RE.match(line):
		return False
	m = re.match(r'\s*["\']?([A-Za-z])', line)
	return bool(m and m.group(1).islower())

def join_wrapped_lines(text: str) -> str:
	"""Join hard-wrapped lines back into prose.

	A line is merged onto the previous non-blank line when its first
	alphabetic character is lowercase, *unless* it starts with a list
	marker (e.g. 'a)', 'a.', 'iii).', '(iv)', '1.', '•').

	Blank lines between a continuation line and its predecessor are
	consumed (not preserved). All other blank lines are kept as-is.
	"""
	lines = text.splitlines()
	out: List[str] = []
	pending_blanks: List[str] = []  # blank lines after last non-blank in out

	for line in lines:
		if not line.strip():
			pending_blanks.append(line)
		elif _starts_with_lowercase(line) and out:
			# Continuation: discard pending blanks, merge onto previous line
			curr = line.strip()
			prev = out[-1].rstrip()
			if prev.endswith('-'):
				out[-1] = prev + curr   # keep the hyphen: "neuro-" + "osteo..." → "neuro-osteo..."
			else:
				out[-1] = prev + ' ' + curr
			pending_blanks = []
		else:
			# New line: flush pending blanks, then append
			out.extend(pending_blanks)
			pending_blanks = []
			out.append(line)

	out.extend(pending_blanks)  # flush any trailing blanks
	return "\n".join(out)

doc_converter = markdown_options = None
def convert_doc(pdf_path, exclude_ack_re=None):
	global doc_converter, markdown_options
	if doc_converter is None:
		from docling.document_converter import DocumentConverter, PdfFormatOption
		from docling.datamodel.base_models import InputFormat
		from docling_core.types.doc.document import ContentLayer
		from docling.datamodel.pipeline_options import TableStructureOptions, PdfPipelineOptions
		pipeline_options = PdfPipelineOptions()
		pipeline_options.do_table_structure = True
		pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
		doc_converter = DocumentConverter(format_options={
			InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options,)
		})
		markdown_options = {
			"traverse_pictures":True,
			"include_annotations":True,
			"included_content_layers":{
				ContentLayer.BODY,
				ContentLayer.FURNITURE,
				ContentLayer.BACKGROUND,
				ContentLayer.INVISIBLE,
				ContentLayer.NOTES,
			}
		}

	doc = doc_converter.convert(pdf_path).document
	md_pages = [doc.export_to_markdown(**markdown_options, page_no=i) for i in range(1, doc.num_pages()+1)]
	if exclude_ack_re:
		md_pages = [page for page in md_pages if not exclude_ack_re.search(page)]
	text_raw = "\n\n".join(md_pages)
	text = join_wrapped_lines(text_raw)
	text = html.unescape(text)
	text = text.replace('<!-- image -->\n', '')
	
	# Remove expert group section
	lines = text.splitlines()
	title_set = set('Dr Ms Mr Adj A/Prof Assoc Prof'.split())
	idx = next((i for i, line in enumerate(lines) if line.lower().startswith('## expert group')), -1)
	if idx >= 0:
		lines.pop(idx)
		while idx < len(lines):
			if not lines[idx].strip():
				pass
			elif lines[idx].strip().startswith('##'):
				pass
			elif lines[idx].split()[0] in title_set:
				pass
			else:
				break
			lines.pop(idx)
	text = '\n'.join(lines)
	return text, text_raw

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
	txt = convert_doc("/home/xuancong/projects/llm-expts/data/foot-assessment-in-patients-with-diabetes-mellitus-(aug-2024).pdf", do_post_process=False)
	with open("/tmp/tp1.md", "w") as f:
		f.write(txt)
	with open("/tmp/tp2.md", "w") as f:
		f.write(join_wrapped_lines(txt))
	print(txt)