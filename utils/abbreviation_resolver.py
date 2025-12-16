import re

class AbbreviationResolver:
    """Resolve biological abbreviations"""
    
    def resolve_from_text(self, text: str) -> dict[str, str]:
        """Extract abbreviations and their full forms from text"""
        abbreviations = {}
        
        # Pattern: Full form (Abbreviation)
        pattern = r'(\b[A-Z][a-zA-Z\s\-]+?)\s*\(([A-Z]{2,}(?:\-[A-Z]{1,})?)\)'
        
        for match in re.finditer(pattern, text):
            full_form = match.group(1).strip()
            abbr = match.group(2).strip()
            
            # Validate: abbreviation should be much shorter
            if len(abbr) < len(full_form) and len(abbr) >= 2:
                abbreviations[abbr] = full_form
        
        return abbreviations
    
    def expand_abbreviations(self, text: str, abbreviations: dict) -> str:
        """Expand abbreviations in text"""
        for abbr, full_form in abbreviations.items():
            # Replace standalone abbreviations
            pattern = r'\b' + re.escape(abbr) + r'\b(?!\s*\()'
            text = re.sub(pattern, f"{full_form} ({abbr})", text)
        
        return text
"""  
    def get_common_abbr(self) -> dict[str, str]:
            # Common biomedical abbreviations
        return {
            'DNA': 'Deoxyribonucleic Acid',
            'RNA': 'Ribonucleic Acid',
            'HbA1c': 'Hemoglobin A1c',
            'LPS': 'Lipopolysaccharide',
            'TNF': 'Tumor Necrosis Factor',
            'IL': 'Interleukin',
            'IFN': 'Interferon',
            'PCR': 'Polymerase Chain Reaction',
            'CRISPR': 'Clustered Regularly Interspaced Short Palindromic Repeats',
            'CAR-T': 'Chimeric Antigen Receptor T cell',
            'CRS': 'Cytokine Release Syndrome',
            'ALL': 'Acute Lymphoblastic Leukemia',
            'CFU': 'Colony Forming Unit',
            'OTU': 'Operational Taxonomic Unit',
            'rRNA': 'Ribosomal RNA',
            '16S': '16S ribosomal RNA',
            'qPCR': 'Quantitative PCR',
            'GWAS': 'Genome-Wide Association Study'
        }
"""
