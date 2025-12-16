class KnowledgeGraph:
    """Build and manage biomedical knowledge graph"""
    
    def __init__(self):
        self.nodes = {}  # entity_name -> {type, mentions}
        self.edges = []  # list of relationships
    
    def add_entity(self, name: str, entity_type: str):
        """Add entity to knowledge graph"""
        if name not in self.nodes:
            self.nodes[name] = {
                'type': entity_type,
                'mentions': 0
            }
        self.nodes[name]['mentions'] += 1
    
    def add_relationship(self, source: str, target: str, rel_type: str, evidence: str):
        """Add relationship to knowledge graph"""
        self.edges.append({
            'source': source,
            'target': target,
            'type': rel_type,
            'evidence': evidence
        })
    
    def get_entity_summary(self, entity_name: str) -> dict:
        """Get summary for entity"""
        if entity_name not in self.nodes:
            return None
        
        # Find all relationships
        outgoing = [e for e in self.edges if e['source'] == entity_name]
        incoming = [e for e in self.edges if e['target'] == entity_name]
        
        return {
            'name': entity_name,
            'type': self.nodes[entity_name]['type'],
            'mentions': self.nodes[entity_name]['mentions'],
            'related_to': len(outgoing),
            'related_by': len(incoming),
            'connections': {
                'outgoing': outgoing,
                'incoming': incoming
            }
        }
    
    def export_csv(self) -> tuple[str, str]:
        """Export as CSV"""
        # Entities CSV
        entities_csv = "Entity,Type,Mentions\n"
        for name, data in self.nodes.items():
            entities_csv += f'"{name}",{data["type"]},{data["mentions"]}\n'
        
        # Relationships CSV
        relationships_csv = "Source,Target,RelationType,Evidence\n"
        for edge in self.edges:
            relationships_csv += f'"{edge["source"]}","{edge["target"]}",{edge["type"]},'
            relationships_csv += f'"{edge["evidence"].replace(chr(34), chr(34)+chr(34))}"\n'
        
        return entities_csv, relationships_csv
    
    def export_json(self) -> dict:
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'statistics': {
                'total_entities': len(self.nodes),
                'total_relationships': len(self.edges),
                'entity_types': list(set(node['type'] for node in self.nodes.values()))
            }
        }
