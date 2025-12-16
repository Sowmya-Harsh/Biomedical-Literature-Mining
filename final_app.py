# ============================
# SETUP
# ============================
# !pip install biopython transformers sentencepiece nltk sentence-transformers gradio --quiet

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
    AutoModelForTokenClassification,
    pipeline,
)
from sentence_transformers import SentenceTransformer, util
import gradio as gr

nltk.download("punkt")
nltk.download("stopwords")

device = "cpu"   # keep CPU to avoid GPU/RAPIDS issues in shared environments
print("Using device:", device)

# ============================
# 1. PubMed Access + Ranking
# ============================
Entrez.email = os.getenv("NCBI_EMAIL", "sowmyajanmahanthi@example.com")

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

def pubmed_fetch_abstract(pmid: str) -> str:
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
    text = handle.read()
    handle.close()
    return text

rank_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)

def ranked_pubmed_results(query: str, max_results: int = 5, search_pool: int = 25):
    ids = pubmed_search_ids(query, max_results=search_pool)
    docs = []
    for pmid in ids:
        try:
            abs_text = pubmed_fetch_abstract(pmid)
            if abs_text.strip():
                docs.append((pmid, abs_text))
        except Exception as e:
            print("Fetch error for", pmid, ":", e)

    if not docs:
        return []

    query_emb = rank_model.encode(query, convert_to_tensor=True)
    doc_embs = rank_model.encode([d[1] for d in docs], convert_to_tensor=True)
    scores = util.cos_sim(query_emb, doc_embs)[0]

    ranked = sorted(zip(docs, scores.cpu().tolist()), key=lambda x: x[1], reverse=True)
    ranked = [(pmid, abs_text, score) for ((pmid, abs_text), score) in ranked[:max_results]]
    return ranked

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
# 3. Neural Summarization (BART)
# ============================
bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_model_name).to(device)

def neural_summarize(query: str, text: str, max_len: int = 160, min_len: int = 40) -> str:
    """
    Query-conditioned abstractive summarization:
    prepend query to context so the model focuses on it.
    """
    combined = f"question: {query} context: {text}"
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
# 4. BioBERT NER
# ============================
biobert_model_name = "Ishan0612/biobert-ner-disease-ncbi"  # public BioBERT NER[web:8][web:20]
biobert_ner = pipeline(
    "token-classification",
    model=biobert_model_name,
    tokenizer=biobert_model_name,
    aggregation_strategy="simple",
    device=-1,   # CPU
)

TYPE_MAP = {
    "DISEASE": "Disease",
    "DIS": "Disease",
    "CHEMICAL": "Chemical",
    "CHEM": "Chemical",
    "GENE": "Gene",
    "PROTEIN": "Gene",
    "CELL_LINE": "Organism",
    "ORG": "Organism",
}

def biobert_ner_wrap(text: str) -> List[Dict]:
    ents = biobert_ner(text)
    results = []
    for e in ents:
        raw = e["entity_group"]
        mapped = TYPE_MAP.get(raw, raw)
        results.append({
            "name": e["word"],
            "type": mapped,
            "start": int(e["start"]),
            "end": int(e["end"]),
            "confidence": float(e["score"]),
            "sources": "BioBERT",
        })
    return results

# ============================
# 5. Simple Relation Extraction (rule-based)
# ============================
REL_PATTERNS = {
    "ASSOCIATION": [
        r"(.+?)\s+is\s+associated\s+with\s+(.+?)",
        r"(.+?)\s+is\s+related\s+to\s+(.+?)",
        r"(.+?)\s+is\s+linked\s+to\s+(.+?)",
    ],
    "CAUSES": [
        r"(.+?)\s+causes\s+(.+?)",
        r"(.+?)\s+leads\s+to\s+(.+?)",
        r"(.+?)\s+results\s+in\s+(.+?)",
    ],
    "TREATS": [
        r"(.+?)\s+treats\s+(.+?)",
        r"(.+?)\s+is\s+used\s+to\s+treat\s+(.+?)",
        r"(.+?)\s+is\s+effective\s+against\s+(.+?)",
    ],
}

def find_relations_rule_based(sentence: str, entities: List[Dict]) -> List[Dict]:
    rels = []
    for rel_type, patterns in REL_PATTERNS.items():
        for pat in patterns:
            for m in re.finditer(pat, sentence, flags=re.IGNORECASE):
                span1 = m.group(1).strip()
                span2 = m.group(2).strip()

                ent1 = None
                ent2 = None
                for e in entities:
                    if e["name"].lower() in span1.lower() and ent1 is None:
                        ent1 = e
                    elif e["name"].lower() in span2.lower() and ent2 is None:
                        ent2 = e

                if ent1 and ent2 and ent1["name"] != ent2["name"]:
                    rels.append({
                        "source": ent1["name"],
                        "target": ent2["name"],
                        "type": rel_type,
                        "evidence": sentence.strip(),
                        "confidence": 0.75,
                    })
    return rels

def extract_relations_full(text: str) -> List[Dict]:
    sentences = sent_tokenize(text)
    all_rels = []
    for sent in sentences:
        ents = biobert_ner_wrap(sent)
        if not ents:
            continue
        rels = find_relations_rule_based(sent, ents)
        all_rels.extend(rels)
    return all_rels

# ============================
# 6. Knowledge Graph
# ============================
class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}    # name -> {type, mentions}
        self.edges = []    # list of {source, target, type, evidence}

    def add_entity(self, name: str, etype: str):
        if name not in self.nodes:
            self.nodes[name] = {"type": etype, "mentions": 0}
        self.nodes[name]["mentions"] += 1

    def add_relationship(self, source: str, target: str, rel_type: str, evidence: str):
        self.edges.append({
            "source": source,
            "target": target,
            "type": rel_type,
            "evidence": evidence,
        })

kg = KnowledgeGraph()

# ============================
# 7. Keywords (optional)
# ============================
EN_STOPWORDS = set(stopwords.words("english"))

def extract_keywords(text: str, top_k: int = 20) -> List[str]:
    words = re.findall(r"[A-Za-z]+", text.lower())
    freq = defaultdict(int)
    for w in words:
        if w not in EN_STOPWORDS and len(w) > 2:
            freq[w] += 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, c in sorted_words[:top_k]]

# ============================
# 8. End-to-End for One Paper
# ============================
def process_paper(query: str, pmid: str, abstract_text: str) -> Dict:
    # 1) Abbreviation expansion
    abbrs = extract_abbreviations(abstract_text)
    expanded = expand_abbreviations(abstract_text, abbrs)

    # 2) Neural, query-aware summarization
    summary = neural_summarize(query, expanded)

    # 3) BioBERT NER on summary
    entities = biobert_ner_wrap(summary)
    for e in entities:
        kg.add_entity(e["name"], e["type"])

    # 4) Relations
    relations = extract_relations_full(summary)
    for r in relations:
        kg.add_relationship(r["source"], r["target"], r["type"], r["evidence"])

    # 5) Keywords
    kws = extract_keywords(summary, top_k=15)

    return {
        "query": query,
        "pmid": pmid,
        "summary": summary,
        "entities": entities,
        "relations": relations,
        "keywords": kws,
        "abbreviations": abbrs,
    }

def run_pipeline_for_query(query: str, max_papers: int = 3):
    results = ranked_pubmed_results(query, max_results=max_papers, search_pool=20)
    processed = []
    for pmid, abs_text, score in results:
        out = process_paper(query, pmid, abs_text)
        out["relevance_score"] = float(score)
        processed.append(out)
    return processed

# ============================
# 9. Gradio Interface (multi-paper)
# ============================
LAST_RESULTS = []

def run_and_cache(query: str, max_papers: int):
    global LAST_RESULTS
    LAST_RESULTS = run_pipeline_for_query(query, max_papers=max_papers)
    if not LAST_RESULTS:
        return gr.update(choices=["No papers found"], value="No papers found")
    labels = [
        f"{i+1}. PMID {r['pmid']} | score={r.get('relevance_score', 0):.3f}"
        for i, r in enumerate(LAST_RESULTS)
    ]
    return gr.update(choices=labels, value=labels[0])

def show_paper(index_label: str):
    if not LAST_RESULTS or not index_label or index_label.startswith("No papers"):
        return "No data.", "", "", ""
    try:
        idx = int(index_label.split(".")[0]) - 1
    except Exception:
        idx = 0
    idx = max(0, min(idx, len(LAST_RESULTS)-1))
    out = LAST_RESULTS[idx]
    pmid = out["pmid"]
    score = out.get("relevance_score", 0.0)

    summary = out["summary"]

    # Deduplicate for display by (name, type)
    seen = set()
    display_ents = []
    for e in out["entities"]:
        key = (e["name"], e["type"])
        if key in seen:
            continue
        seen.add(key)
        display_ents.append(e)

    ents_str = "\n".join(
        f"- {e['name']} ({e['type']}, conf={e['confidence']:.2f}, src={e['sources']})"
        for e in display_ents
    ) or "No entities found."

    rels_str = "\n".join(
        f"- {r['source']} --[{r['type']}]--> {r['target']}\n  evidence: {r['evidence']}"
        for r in out["relations"]
    ) or "No relations found."

    info_str = (
        f"PMID: {pmid}\n"
        f"Relevance score: {score:.3f}\n"
        f"Abbreviations expanded: {len(out['abbreviations'])}\n"
        f"Keywords: {', '.join(out['keywords'])}"
    )

    return summary, ents_str, rels_str, info_str

with gr.Blocks() as demo:
    gr.Markdown("## PubMed Knowledge Graph – BioBERT + Neural Summarization")

    with gr.Row():
        query_in = gr.Textbox(
            label="Enter biomedical query",
            placeholder="e.g., TNF alpha inflammation",
            lines=1,
        )
        max_papers_in = gr.Slider(
            minimum=1, maximum=15, value=3, step=1,
            label="Number of papers to fetch and process"
        )
        run_btn = gr.Button("Search & Process")

    paper_selector = gr.Dropdown(
        label="Select a paper",
        choices=["No papers yet"],
        value="No papers yet"
    )

    summary_out = gr.Textbox(label="Abstractive Summary", lines=6)
    ents_out = gr.Textbox(label="Extracted Entities (BioBERT)", lines=10)
    rels_out = gr.Textbox(label="Extracted Relationships", lines=10)
    info_out = gr.Textbox(label="Paper Info & Keywords", lines=6)

    run_btn.click(
        fn=run_and_cache,
        inputs=[query_in, max_papers_in],
        outputs=paper_selector,
    )

    paper_selector.change(
        fn=show_paper,
        inputs=paper_selector,
        outputs=[summary_out, ents_out, rels_out, info_out],
    )

demo.launch(share=True)
