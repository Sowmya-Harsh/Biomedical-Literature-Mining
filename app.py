from flask import Flask, render_template, request, jsonify
from utils.pubmed import PubMedSearcher
from utils.entity_extractor import BiomedicalEntityExtractor
from utils.abbreviation_resolver import AbbreviationResolver
from utils.relationship_extractor import RelationshipExtractor
from utils.knowledge_graph import KnowledgeGraph
from utils.text_summarizer_dynamic import TextSummarizer  # DYNAMIC + TITLE-AWARE version
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['NCBI_EMAIL'] = os.getenv('NCBI_EMAIL', 'sowmyajanmahanthi@gmail.com')

# Initialize tools
searcher = PubMedSearcher(app.config['NCBI_EMAIL'])
entity_extractor = BiomedicalEntityExtractor()
abbr_resolver = AbbreviationResolver()
rel_extractor = RelationshipExtractor()
summarizer = TextSummarizer()
knowledge_graph = KnowledgeGraph()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """Search PubMed and extract entities + SUMMARIZE (query-aware + title-aware)"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        num_papers = int(data.get('num_papers', 10))
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        pmids = searcher.search(query, max_results=num_papers)
        
        papers_with_mining = []
        
        for pmid in pmids:
            paper_data = searcher.fetch_details(pmid)
            
            if paper_data:
                abstract = paper_data['abstract']
                title = paper_data.get('title', '')
                
                # ========== EXTRACT ENTITIES ==========
                entities = entity_extractor.extract_all(abstract)
                
                # ========== RESOLVE ABBREVIATIONS ==========
                abbreviations = abbr_resolver.resolve_from_text(abstract)
                
                # ========== EXTRACT RELATIONSHIPS ==========
                relationships = rel_extractor.extract_relationships(abstract)
                
                # ========== GENERATE DYNAMIC SUMMARY (CONTEXT-AWARE + TITLE-AWARE) ==========
                # Pass query AND title to the summarizer
                summary_result = summarizer.summarize(abstract, query=query, title=title, num_sentences=3)
                summary_bullets = summarizer.get_summary_bullets(abstract, query=query, title=title, num_bullets=3)
                findings = summarizer.extract_findings(abstract, query=query, title=title)
                methodology = summarizer.extract_methodology(abstract, query=query, title=title)
                implications = summarizer.extract_implications(abstract, query=query, title=title)
                
                # Get title focus score (how much does paper focus on query)
                title_focus = summary_result['title_focus']
                
                # Add to knowledge graph
                for entity in entities:
                    knowledge_graph.add_entity(entity['name'], entity['type'])
                
                for rel in relationships:
                    knowledge_graph.add_relationship(
                        rel['source'], rel['target'], 
                        rel['relation_type'], rel['evidence']
                    )
                
                # ========== ADD ALL TO PAPER DATA ==========
                paper_data['entities'] = entities
                paper_data['abbreviations'] = abbreviations
                paper_data['relationships'] = relationships
                
                # Summary data (DYNAMIC + TITLE-AWARE based on query)
                paper_data['summary_bullets'] = summary_bullets
                paper_data['key_phrases'] = summary_result['key_phrases']
                paper_data['methodology'] = methodology
                paper_data['findings'] = findings
                paper_data['implications'] = implications
                paper_data['query_used'] = query  # Store query for reference
                paper_data['title_focus'] = title_focus  # NEW: How focused is this paper
                
                # Relevance indicator based on title focus
                if title_focus > 0.7:
                    paper_data['relevance'] = 'HIGHLY FOCUSED'
                elif title_focus > 0.4:
                    paper_data['relevance'] = 'RELEVANT'
                else:
                    paper_data['relevance'] = 'TANGENTIAL'
                
                papers_with_mining.append(paper_data)
        
        return jsonify({
            'success': True,
            'papers': papers_with_mining,
            'total': len(papers_with_mining),
            'query': query,  # Return query in response
            'knowledge_graph_stats': {
                'total_entities': len(knowledge_graph.nodes),
                'total_relationships': len(knowledge_graph.edges)
            }
        })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/knowledge-graph', methods=['GET'])
def get_knowledge_graph():
    """Get current knowledge graph"""
    return jsonify(knowledge_graph.export_json())

@app.route('/api/export', methods=['POST'])
def export_data():
    """Export mining results"""
    try:
        data = request.get_json()
        format_type = data.get('format', 'json')
        
        if format_type == 'json':
            return jsonify(knowledge_graph.export_json())
        elif format_type == 'csv':
            entities_csv, rel_csv = knowledge_graph.export_csv()
            return jsonify({
                'entities': entities_csv,
                'relationships': rel_csv
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
