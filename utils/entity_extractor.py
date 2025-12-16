import re

class BiomedicalEntityExtractor:
    """Extract biomedical entities using patterns"""
    
    def __init__(self):
        """Initialize with entity dictionaries"""
        
        # Known genes
        self.genes = {
            'BRCA1', 'BRCA2', 'TP53', 'EGFR', 'KRAS', 'MYC',
            'CDK4', 'PTEN', 'RB1', 'APC', 'PIK3CA', 'ERBB2',
            'ALK', 'BRAF', 'NRAS', 'HRAS', 'PTEN', 'NF1', 'FOXP3'
        }
        
        # Known diseases
        self.diseases = {
            'cancer', 'diabetes', 'leukemia', 'lymphoma', 'melanoma',
            'carcinoma', 'adenoma', 'fibrosis', 'arthritis', 'pneumonia',
            'infection', 'inflammation', 'obesity', 'hypertension',
            'heart disease', 'stroke', 'alzheimer', 'parkinson'
        }
        
        # Known bacteria/organisms
        self.organisms = {
            'Firmicutes', 'Bacteroidetes', 'Lactobacillus',
            'Clostridium', 'Streptococcus', 'Staphylococcus',
            'Escherichia coli', 'E. coli', 'Bacillus',
            'Faecalibacterium', 'Salmonella', 'Listeria',
            'Candida', 'Aspergillus'
        }
        
        # Known chemicals
        self.chemicals = {
            'Metformin', 'Aspirin', 'Warfarin', 'Lipopolysaccharide',
            'LPS', 'TNF-alpha', 'TNF', 'IL-6', 'IL-10', 'IFN-gamma'
        }
        
        # Biological processes
        self.processes = {
            'inflammation', 'apoptosis', 'differentiation', 'proliferation',
            'angiogenesis', 'necrosis', 'glycolysis', 'transcription',
            'translation', 'phosphorylation', 'ubiquitination'
        }
    
    def extract_genes(self, text: str) -> list[dict]:
        entities = []
        
        # Pattern 1: All caps with numbers (BRCA1, TP53)
        pattern1 = r'\b([A-Z]{1,4}\d+)\b'
        matches = re.finditer(pattern1, text)
        
        for match in matches:
            gene = match.group(1)
            if gene in self.genes or len(gene) <= 5:  # Likely a gene
                entities.append({
                    'name': gene,
                    'type': 'Gene',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        # Pattern 2: Known genes from dictionary
        for gene in self.genes:
            pattern = r'\b' + re.escape(gene) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'name': gene,
                    'type': 'Gene',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.95
                })
        
        return entities
    
    def extract_diseases(self, text: str) -> list[dict]:
        entities = []
        
        for disease in self.diseases:
            pattern = r'\b' + re.escape(disease) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'name': disease.lower(),
                    'type': 'Disease',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        # Pattern for -itis, -osis, -emia
        patterns = [
            (r'\b\w+itis\b', 'Disease'),
            (r'\b\w+osis\b', 'Disease'),
            (r'\b\w+emia\b', 'Disease'),
            (r'\b\w+pathy\b', 'Disease'),
            (r'\bcarcinoma\b', 'Disease'),
            (r'\badenoma\b', 'Disease'),
        ]
        
        for pattern, etype in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'name': match.group(0).lower(),
                    'type': etype,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.7
                })
        
        return entities
    
    def extract_organisms(self, text: str) -> list[dict]:
        """Extract organism/bacteria mentions"""
        entities = []
        
        for organism in self.organisms:
            pattern = r'\b' + re.escape(organism) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'name': organism,
                    'type': 'Organism',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        # Pattern for capitalized genus + species
        pattern = r'\b([A-Z][a-z]+)\s+([a-z]+)\b'
        for match in re.finditer(pattern, text):
            genus_species = f"{match.group(1)} {match.group(2)}"
            entities.append({
                'name': genus_species,
                'type': 'Organism',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.6
            })
        
        return entities
    
    def extract_chemicals(self, text: str) -> list[dict]:
        """Extract chemical/drug mentions"""
        entities = []
        
        for chemical in self.chemicals:
            pattern = r'\b' + re.escape(chemical) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'name': chemical,
                    'type': 'Chemical',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        return entities
    
    def extract_all(self, text: str) -> list[dict]:
        """Extract all entity types"""
        all_entities = []
        all_entities.extend(self.extract_genes(text))
        all_entities.extend(self.extract_diseases(text))
        all_entities.extend(self.extract_organisms(text))
        all_entities.extend(self.extract_chemicals(text))
        
        # Remove duplicates
        seen = set()
        unique_entities = []
        for entity in all_entities:
            key = (entity['name'].lower(), entity['type'], entity['start'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return sorted(unique_entities, key=lambda x: x['start'])
