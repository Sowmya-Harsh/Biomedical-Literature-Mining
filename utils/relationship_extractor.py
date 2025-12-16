import re

class RelationshipExtractor:
    """Extract relationships between entities"""
    
    def __init__(self):
        """Initialize relationship patterns"""
        
        self.relationships = {
            'ASSOCIATION': [
                r'(.+?)\s+(?:is\s+)?associated\s+with\s+(.+?)(?:[.;,])',
                r'(.+?)\s+related\s+to\s+(.+?)(?:[.;,])',
                r'(.+?)\s+linked\s+to\s+(.+?)(?:[.;,])',
                r'(.+?)\s+correlated\s+with\s+(.+?)(?:[.;,])',
            ],
            'CAUSATION': [
                r'(.+?)\s+causes?\s+(.+?)(?:[.;,])',
                r'(.+?)\s+leads?\s+to\s+(.+?)(?:[.;,])',
                r'(.+?)\s+triggers?\s+(.+?)(?:[.;,])',
                r'(.+?)\s+induces?\s+(.+?)(?:[.;,])',
            ],
            'TREATMENT': [
                r'(.+?)\s+(?:treats?|used\s+to\s+treat)\s+(.+?)(?:[.;,])',
                r'(.+?)\s+effective\s+against\s+(.+?)(?:[.;,])',
                r'(.+?)\s+therapy\s+for\s+(.+?)(?:[.;,])',
            ],
            'ENRICHMENT': [
                r'(.+?)\s+enriched\s+in\s+(.+?)(?:[.;,])',
                r'(.+?)\s+abundance\s+(?:high|increased)\s+in\s+(.+?)(?:[.;,])',
                r'(.+?)\s+elevated\s+in\s+(.+?)(?:[.;,])',
            ],
            'DEPLETION': [
                r'(.+?)\s+depleted\s+in\s+(.+?)(?:[.;,])',
                r'(.+?)\s+reduced\s+in\s+(.+?)(?:[.;,])',
                r'(.+?)\s+decreased\s+in\s+(.+?)(?:[.;,])',
                r'(.+?)\s+absent\s+in\s+(.+?)(?:[.;,])',
            ],
        }
    
    def extract_relationships(self, text: str) -> list[dict]:
        """Extract relationships from text"""
        relationships = []
        
        for rel_type, patterns in self.relationships.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity1 = match.group(1).strip()
                    entity2 = match.group(2).strip()
                    
                    # Clean up entities
                    entity1 = self._clean_entity(entity1)
                    entity2 = self._clean_entity(entity2)
                    
                    if entity1 and entity2:
                        relationships.append({
                            'source': entity1,
                            'target': entity2,
                            'relation_type': rel_type,
                            'evidence': match.group(0).strip(),
                            'confidence': 0.75
                        })
        
        return relationships
    
    def _clean_entity(self, entity: str) -> str:
        """Clean extracted entity name"""
        # Remove articles
        entity = re.sub(r'\b(?:the|a|an)\s+', '', entity, flags=re.IGNORECASE)
        # Remove extra whitespace
        entity = ' '.join(entity.split())
        # Remove trailing/leading punctuation
        entity = entity.strip('.,;:')
        
        return entity if len(entity) > 2 else ''
