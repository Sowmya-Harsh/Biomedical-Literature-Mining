# ============================
# INSTALLS (run once)
# ============================
!pip install biopython transformers sentencepiece nltk sentence-transformers gradio --quiet

# ============================
# IMPORTS
# ============================
import os
import re
from typing import List, Dict
from collections import defaultdict

import torch
from Bio import Entrez
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from sentence_transformers import SentenceTransformer, util
import gradio as gr

# ============================
# NLTK SETUP
# ============================
nltk.download("punkt")
nltk.download("stopwords")

EN_STOPWORDS = set(stopwords.words("english"))

device = "cpu"
print("Using device:", device)

# ============================
# HELPERS
# ============================
def clean_markup(text: str) -> str:
    text = re.sub(r"<sup>.*?</sup>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^\s*•\s*", "", text, flags=re.MULTILINE)
    return text

def extract_keywords(text: str, top_k: int = 15) -> List[str]:
    words = re.findall(r"[A-Za-z]+", text.lower())
    freq = defaultdict(int)
    for w in words:
        if w not in EN_STOPWORDS and len(w) > 2:
            freq[w] += 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, c in sorted_words[:top_k]]

def extract_query_keywords(query: str) -> set:
    words = query.lower().split()
    kws = set()
    for w in words:
        w = re.sub(r'[,.\'"()-]', "", w)
        if w not in EN_STOPWORDS and len(w) > 2:
            kws.add(w)
            if w.endswith("ies"):
                kws.add(w[:-3] + "y")
            elif w.endswith("es"):
                kws.add(w[:-2])
            elif w.endswith("s") and not w.endswith("ss"):
                kws.add(w[:-1])
    return kws

def analyze_title_focus(title: str, query_keywords: set) -> float:
    if not title or not query_keywords:
        return 0.0
    words = [
        w.strip(".,;:!?()[]{}")
        for w in title.lower().split()
        if w not in EN_STOPWORDS
    ]
    if not words:
        return 0.0
    matches = 0.0
    for w in words:
        if w in query_keywords:
            matches += 1.0
        else:
            for kw in query_keywords:
                if kw in w or w in kw:
                    matches += 0.5
                    break
    return min(matches / len(words), 1.0)

UNIVERSAL_IMPORTANCE = {
    "study", "research", "investigate", "examine", "analyze", "assess",
    "find", "found", "shows", "demonstrate", "reveal", "indicate",
    "suggest", "conclude", "propose", "determine", "establish",
    "associated", "linked", "relationship", "mechanism", "pathway",
    "result", "outcome", "finding", "conclusion", "implication",
    "treatment", "therapy", "intervention", "prevention", "disease",
    "important", "significant", "novel", "critical", "essential",
}

def score_sentence(sentence: str, query_keywords: set, title_focus: float) -> float:
    words = sentence.lower().split()
    important = [
        w.strip(".,;:!?()[]{}")
        for w in words
        if w not in EN_STOPWORDS and len(w) > 2
    ]
    if not important:
        return 0.0

    qm = 0.0
    for w in important:
        if w in query_keywords:
            qm += 1.0
        else:
            for kw in query_keywords:
                if kw in w or w in kw:
                    qm += 0.5
                    break

    um = sum(1 for w in important if w in UNIVERSAL_IMPORTANCE)
    length_score = 1.0 if 8 <= len(words) <= 25 else 0.5

    query_score = qm / len(important)
    uni_score = um / len(important)
    base = query_score * 0.6 + uni_score * 0.25 + length_score * 0.15

    title_boost = 1.0 + 0.3 * title_focus
    return base * title_boost

# ============================
# 1. PubMed: search, fetch, metadata
# ============================
Entrez.email = os.getenv("NCBI_EMAIL", "your_email@example.com")

def pubmed_search_ids(query: str, max_results: int = 20) -> List[str]:
    handle = Entrez.esearch(
        db="pubmed",
        term=query + " AND humans[MeSH Terms] AND english[lang]",
        retmax=max_results,
        sort="relevance"
    )
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])

def pubmed_fetch_details(pmid: str) -> Dict:
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="text")
    record = Entrez.read(handle)
    handle.close()
    if not record.get("PubmedArticle"):
        return {}
    article = record["PubmedArticle"][0]["MedlineCitation"]["Article"]
    title = article.get("ArticleTitle", "")
    journal = article.get("Journal", {}).get("Title", "")
    date = "Unknown"
    ad = article.get("ArticleDate", [])
    if ad:
        y = ad[0].get("Year", "")
        m = ad[0].get("Month", "")
        d = ad[0].get("Day", "")
        if y:
            date = f"{y}-{m:0>2}-{d:0>2}" if m and d else y
    abstract_sections = article.get("Abstract", {}).get("AbstractText", [])
    if isinstance(abstract_sections, list):
        abstract = " ".join(str(s) for s in abstract_sections)
    else:
        abstract = str(abstract_sections)
    authors_list = article.get("AuthorList", [])
    authors = []
    for a in authors_list[:3]:
        last = a.get("LastName", "")
        init = a.get("Initials", "")
        if last and init:
            authors.append(f"{last} {init}")
    if len(authors_list) > 3:
        authors.append("et al.")
    author_str = ", ".join(authors) if authors else "Unknown"
    return {
        "title": str(title),
        "journal": str(journal),
        "date": date,
        "abstract": abstract.strip(),
        "authors": author_str,
    }

rank_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)

def ranked_pubmed_results(query: str, max_results: int = 3, search_pool: int = 20):
    ids = pubmed_search_ids(query, max_results=search_pool)
    docs = []
    for pmid in ids:
        try:
            details = pubmed_fetch_details(pmid)
            abs_text = details.get("abstract", "")
            if abs_text.strip():
                docs.append((pmid, details))
        except Exception as e:
            print("Fetch error for", pmid, ":", e)
    if not docs:
        return []
    query_emb = rank_model.encode(query, convert_to_tensor=True)
    doc_embs = rank_model.encode([d[1]["abstract"] for d in docs], convert_to_tensor=True)
    scores = util.cos_sim(query_emb, doc_embs)[0]
    ranked = sorted(zip(docs, scores.cpu().tolist()), key=lambda x: x[1], reverse=True)
    return [(pmid, details, score) for ((pmid, details), score) in ranked[:max_results]]

# ============================
# 2. Abbreviation Resolver
# ============================
COMMON_ABBR = {
    "DNA": "Deoxyribonucleic Acid",
    "RNA": "Ribonucleic Acid",
    "LPS": "Lipopolysaccharide",
    "TNF": "Tumor Necrosis Factor",
    "IL": "Interleukin",
    "IFN": "Interferon",
    "PCR": "Polymerase Chain Reaction",
    "CRISPR": "Clustered Regularly Interspaced Short Palindromic Repeats",
    "GWAS": "Genome-Wide Association Study",
}

def extract_abbreviations(text: str) -> Dict[str, str]:
    pattern = r"([A-Za-z][A-Za-z\s\-]+?)\s*\(([A-Z][A-Z0-9\-]{1,15})\)"
    abbrs = {}
    for full, abbr in re.findall(pattern, text):
        full = full.strip()
        abbr = abbr.strip()
        if len(abbr) < len(full):
            abbrs[abbr] = full
    for a, f in COMMON_ABBR.items():
        if re.search(rf"\b{re.escape(a)}\b", text) and a not in abbrs:
            abbrs[a] = f
    return abbrs

def expand_abbreviations(text: str, abbrs: Dict[str, str]) -> str:
    for abbr, full in abbrs.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
    return text

# ============================
# 3. BART Summarization
# ============================
bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_model_name).to(device)

def neural_summarize(query: str, text: str, max_len: int = 160, min_len: int = 40) -> str:
    cleaned = clean_markup(text)
    combined = f"question: {query} context: {cleaned}"
    inputs = bart_tokenizer(
        combined,
        truncation=True,
        padding="longest",
        max_length=1024,
        return_tensors="pt"
    ).to(device)
    summary_ids = bart_model.generate(
        **inputs,
        num_beams=4,
        max_length=max_len,
        min_length=min_len,
        no_repeat_ngram_size=3
    )
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ============================
# 4. PROPER Biomedical NER
# ============================
ner_model_name = "sschet/biomedical-ner-all"  # real biomedical NER
ner_pipe = pipeline(
    "token-classification",
    model=ner_model_name,
    tokenizer=ner_model_name,
    aggregation_strategy="simple",
    device=-1,
)

TYPE_MAP = {
    "DISEASE": "Disease",
    "CHEMICAL": "Chemical",
    "GENE": "Gene",
    "CELL_LINE": "Organism",
    "ORGANISM": "Organism",
}

def transformer_ner(text: str) -> List[Dict]:
    ents = ner_pipe(text)
    results = []
    for e in ents:
        raw = e["entity_group"]
        # Map known labels, otherwise keep raw
        mapped = TYPE_MAP.get(raw, raw)  # e.g., Diagnostic_procedure
        results.append({
            "name": e["word"],
            "type": mapped,
            "raw_type": raw,
            "start": int(e["start"]),
            "end": int(e["end"]),
            "confidence": float(e["score"]),
        })
    return results

# ============================
# 5. Knowledge Graph (entities + CSV)
# ============================
class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}   # name -> {type, mentions}
        self.edges = []   # not used here

    def add_entity(self, name: str, etype: str):
        if name not in self.nodes:
            self.nodes[name] = {"type": etype, "mentions": 0}
        self.nodes[name]["mentions"] += 1

    def export_csv(self):
        entities_csv = "Entity,Type,Mentions\n"
        for name, data in self.nodes.items():
            entities_csv += f"{name},{data['type']},{data['mentions']}\n"
        relationships_csv = "Source,Target,RelationType,Evidence\n"
        return entities_csv, relationships_csv

kg = KnowledgeGraph()

# ============================
# 6. End-to-End for One Paper
# ============================
def process_paper(query: str, pmid: str, meta: Dict) -> Dict:
    title = meta.get("title", "")
    journal = meta.get("journal", "")
    date = meta.get("date", "")
    authors = meta.get("authors", "")
    abstract = meta.get("abstract", "")

    abbrs = extract_abbreviations(abstract)
    expanded = expand_abbreviations(abstract, abbrs)

    summary = neural_summarize(query, expanded)

    qk = extract_query_keywords(query)
    title_focus = analyze_title_focus(title, qk)

    sents = [s.strip() for s in sent_tokenize(summary) if s.strip()]
    sent_scores = [(s, score_sentence(s, qk, title_focus)) for s in sents]
    sent_scores = sorted(sent_scores, key=lambda x: x[1], reverse=True)

    entities = transformer_ner(summary)
    for e in entities:
        kg.add_entity(e["name"], e["type"])

    kws = extract_keywords(summary, top_k=15)

    return {
        "pmid": pmid,
        "title": title,
        "journal": journal,
        "date": date,
        "authors": authors,
        "summary": summary,
        "entities": entities,
        "title_focus": float(title_focus),
        "sentence_scores": sent_scores,
        "abstract": abstract,
        "abbreviations": abbrs,
        "keywords": kws,
    }


def run_pipeline_for_query(query: str, max_papers: int = 1):
    ranked = ranked_pubmed_results(query, max_results=max_papers, search_pool=20)
    out = []
    for pmid, details, score in ranked:
        rec = process_paper(query, pmid, details)
        rec["relevance_score"] = float(score)
        out.append(rec)
    return out

# ============================
# 7. Export functions for Gradio
# ============================
def download_entities():
    entities_csv, _ = kg.export_csv()
    return entities_csv.encode("utf-8")

def download_relationships():
    _, rels_csv = kg.export_csv()
    return rels_csv.encode("utf-8")

# ============================
# 8. Gradio UI
# ============================
LAST_RESULTS = []

def run_and_cache(query: str, max_papers: int):
    global LAST_RESULTS
    LAST_RESULTS = run_pipeline_for_query(query, max_papers=max_papers)
    if not LAST_RESULTS:
        return gr.update(choices=["No papers found"], value="No papers found")
    labels = [
        f"{i+1}. PMID {r['pmid']} | relevance={r.get('relevance_score',0):.3f} | title_focus={r.get('title_focus',0):.2f}"
        for i, r in enumerate(LAST_RESULTS)
    ]
    return gr.update(choices=labels, value=labels[0])

def show_paper(index_label: str):
    if not LAST_RESULTS or not index_label or index_label.startswith("No papers"):
        return "No data.", "", ""
    try:
        idx = int(index_label.split(".")[0]) - 1
    except Exception:
        idx = 0
    idx = max(0, min(idx, len(LAST_RESULTS)-1))
    rec = LAST_RESULTS[idx]

    # formatted summary block
    title = rec["title"]
    authors = rec["authors"]
    journal = rec["journal"]
    date = rec["date"]
    pmid = rec["pmid"]
    summary = rec["summary"]

    lines = []
    lines.append(f"{title}")
    lines.append(f"Authors: {authors}")
    lines.append(f"Journal: {journal}")
    lines.append(f"Date: {date}")
    lines.append(f"PMID: {pmid}")
    lines.append("")
    lines.append("📊 AI Summary")
    for s in sent_tokenize(summary):
        if s.strip():
            lines.append(f"• {s.strip()}")
    lines.append("")
    lines.append("🧬 Extracted Entities")
    if rec["entities"]:
        for e in rec["entities"]:
            lines.append(f"- {e['name']} ({e['type']}, conf={e['confidence']:.2f})")
    else:
        lines.append("No entities found.")
    lines.append("")
    lines.append("📖 Full Abstract")
    lines.append(clean_markup(rec["abstract"]))
    text_block = "\n".join(lines)

    # top sentences with scores
    sents_str = "\n".join(
        f"- [{score:.3f}] {sent}" for sent, score in rec["sentence_scores"][:5]
    )

    info_str = (
        f"Relevance score: {rec.get('relevance_score',0):.3f}\n"
        f"Title focus score: {rec.get('title_focus',0):.2f}\n"
        f"Abbreviations expanded: {len(rec['abbreviations'])}\n"
        f"Keywords: {', '.join(rec['keywords'])}"
    )

    return text_block, sents_str, info_str

with gr.Blocks() as demo:
    gr.Markdown("## PubMed AI Summaries + Knowledge Graph CSV Export")

    with gr.Row():
        query_in = gr.Textbox(
            label="Enter biomedical query",
            placeholder="e.g., AlphaFold 3 drug discovery",
            lines=1,
        )
        max_papers_in = gr.Slider(
            minimum=1, maximum=5, value=1, step=1,
            label="Number of top papers"
        )
        run_btn = gr.Button("Search & Process")

    paper_selector = gr.Dropdown(
        label="Select a paper",
        choices=["No papers yet"],
        value="No papers yet"
    )

    summary_out = gr.Textbox(label="Formatted Summary + Entities + Abstract", lines=20)
    sents_out = gr.Textbox(label="Top Sentences with Scores", lines=8)
    info_out = gr.Textbox(label="Scores & Meta Info", lines=6)

    with gr.Row():
        dl_entities_btn = gr.Button("Export Entities CSV")
        dl_rels_btn = gr.Button("Export Relationships CSV")

    run_btn.click(
        fn=run_and_cache,
        inputs=[query_in, max_papers_in],
        outputs=paper_selector,
    )

    paper_selector.change(
        fn=show_paper,
        inputs=paper_selector,
        outputs=[summary_out, sents_out, info_out],
    )

demo.launch(share=True)
