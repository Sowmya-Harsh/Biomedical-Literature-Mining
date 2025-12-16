from Bio import Entrez
import os
from typing import List, Dict

Entrez.email = os.getenv('NCBI_EMAIL', 'sowmyajanmahanthi@gmail.com')

class PubMedSearcher:
    """Search and fetch papers from PubMed"""
    def __init__(self, email: str = None):
        if email:
            Entrez.email = email
    
    def search(self, query: str, max_results: int = 10) -> list[str]:
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            return record.get("IdList", [])
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def fetch_details(self, pmid: str) -> dict:
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=pmid,
                rettype="xml"
            )
            record = Entrez.read(handle)
            handle.close()
            
            if not record.get("PubmedArticle"):
                return None
            
            article = record["PubmedArticle"][0]["MedlineCitation"]["Article"]
            
            paper_data = {
                "pmid": pmid,
                "title": article.get("ArticleTitle", "N/A"),
                "journal": article.get("Journal", {}).get("Title", "N/A"),
                "date": self._extract_date(article),
                "authors": self._extract_authors(article),
                "abstract": self._extract_abstract(article)
            }
            
            return paper_data
        except Exception as e:
            print(f"Fetch error for {pmid}: {e}")
            return None
    
    def _extract_authors(self, article: Dict) -> str:
        authors = []
        author_list = article.get("AuthorList", [])
        for author in author_list[:3]:
            last_name = author.get("LastName", "")
            first_initial = author.get("Initials", "")
            if last_name and first_initial:
                authors.append(f"{last_name} {first_initial}")
        
        if len(author_list) > 3:
            authors.append("et al.")
        
        return ", ".join(authors) if authors else "Unknown"
    
    def _extract_abstract(self, article: Dict) -> str:
        abstract_sections = article.get("Abstract", {}).get("AbstractText", [])
        
        if isinstance(abstract_sections, list):
            abstract_text = " ".join([str(section) for section in abstract_sections])
        else:
            abstract_text = str(abstract_sections)
        
        return abstract_text.strip() if abstract_text else "No abstract available"
    
    def _extract_date(self, article: Dict) -> str:
        pub_date = article.get("ArticleDate", [])
        
        if pub_date:
            year = pub_date[0].get("Year", "")
            month = pub_date[0].get("Month", "")
            day = pub_date[0].get("Day", "")
            return f"{year}-{month:0>2}-{day:0>2}" if year else "Unknown"
        return "Unknown"
