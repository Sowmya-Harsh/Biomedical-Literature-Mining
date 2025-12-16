const searchInput = document.getElementById('searchQuery');
const numPapersSelect = document.getElementById('numPapers');
const searchBtn = document.getElementById('searchBtn');
const loadingEl = document.getElementById('loading');
const errorEl = document.getElementById('error');
const resultsSection = document.getElementById('resultsSection');
const papersList = document.getElementById('papersList');
const emptyState = document.getElementById('emptyState');
const exportBtn = document.getElementById('exportBtn');

const modal = document.getElementById('paperModal');
const modalCloseBtn = document.getElementById('modalCloseBtn');
const modalTitle = document.getElementById('modalTitle');
const modalAuthors = document.getElementById('modalAuthors');
const modalJournal = document.getElementById('modalJournal');
const modalDate = document.getElementById('modalDate');
const modalPMID = document.getElementById('modalPMID');
const modalSummary = document.getElementById('modalSummary');
const modalAbstract = document.getElementById('modalAbstract');
const modalPubmedLink = document.getElementById('modalPubmedLink');
const modalCopyBtn = document.getElementById('modalCopyBtn');
const modalMethodology = document.getElementById('modalMethodology');
const modalFindings = document.getElementById('modalFindings');
const modalImplications = document.getElementById('modalImplications');
const modalEntities = document.getElementById('modalEntities');
const modalRelationships = document.getElementById('modalRelationships');

let currentPapers = [];

// Events
searchBtn.addEventListener('click', handleSearch);
searchInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') handleSearch();
});
exportBtn.addEventListener('click', handleExport);
modalCloseBtn.addEventListener('click', () => modal.classList.add('hidden'));
modal.addEventListener('click', (e) => {
  if (e.target === modal) modal.classList.add('hidden');
});

async function handleSearch() {
  const query = searchInput.value.trim();
  const numPapers = parseInt(numPapersSelect.value, 10);

  if (!query) {
    showError('Please enter a search query.');
    return;
  }

  errorEl.classList.add('hidden');
  loadingEl.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  emptyState.classList.add('hidden');
  papersList.innerHTML = '';
  currentPapers = [];

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, num_papers: numPapers })
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Search failed');
    }

    currentPapers = data.papers || [];
    if (currentPapers.length === 0) {
      showError('No papers found for this query.');
      return;
    }

    renderPapers(currentPapers);
    resultsSection.classList.remove('hidden');
  } catch (err) {
    showError(err.message);
  } finally {
    loadingEl.classList.add('hidden');
  }
}

function getRelevanceBadgeColor(relevance) {
  switch(relevance) {
    case 'HIGHLY FOCUSED':
      return '#10b981'; // green
    case 'RELEVANT':
      return '#f59e0b'; // amber
    case 'TANGENTIAL':
      return '#ef4444'; // red
    default:
      return '#6b7280'; // gray
  }
}

function renderPapers(papers) {
  papersList.innerHTML = '';
  papers.forEach((paper, index) => {
    const card = document.createElement('div');
    card.className = 'paper-card';

    // Use NLP summary bullets instead of abstract
    const summaryBullets = paper.summary_bullets || [];
    const keyPhrases = paper.key_phrases || [];
    const relevance = paper.relevance || 'TANGENTIAL';
    const titleFocus = (paper.title_focus * 100).toFixed(0);

    // Badge color based on relevance
    const badgeColor = getRelevanceBadgeColor(relevance);

    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
        <div class="paper-title">${escapeHtml(paper.title || 'No title')}</div>
        <span style="
          background-color: ${badgeColor}; 
          color: white; 
          padding: 4px 10px; 
          border-radius: 4px; 
          font-size: 11px; 
          font-weight: bold;
          white-space: nowrap;
          margin-left: 8px;
        ">
          ${relevance} (${titleFocus}%)
        </span>
      </div>
      
      <div class="paper-authors">${escapeHtml(paper.authors || '')}</div>
      <div class="paper-meta">
        <span>${escapeHtml(paper.journal || '')}</span> ·
        <span>${escapeHtml(paper.date || '')}</span> ·
        <span>PMID: ${escapeHtml(paper.pmid || '')}</span>
      </div>
      
      <div class="paper-summary-title">📊 AI Summary</div>
      <div class="summary-bullets-preview">
        ${summaryBullets.slice(0, 2).map(bullet => `<div class="bullet">${escapeHtml(bullet)}</div>`).join('')}
      </div>

      <div class="paper-key-phrases">
        <strong>Key Topics:</strong>
        <div class="phrases-tags">
          ${keyPhrases.slice(0, 3).map(phrase => `<span class="tag">${escapeHtml(phrase)}</span>`).join('')}
        </div>
      </div>

      <div class="paper-entities-preview">
        <strong>${paper.entities ? paper.entities.length : 0}</strong> entities extracted · 
        <strong>${paper.relationships ? paper.relationships.length : 0}</strong> relationships found
      </div>
    `;

    card.addEventListener('click', () => openModal(paper));
    papersList.appendChild(card);
  });
}

function openModal(paper) {
  modalTitle.textContent = paper.title || '';
  modalAuthors.textContent = paper.authors || '';
  modalJournal.textContent = paper.journal || '';
  modalDate.textContent = paper.date || '';
  modalPMID.textContent = paper.pmid || '';

  // ===== NLP SUMMARY SECTION =====
  const summaryBullets = paper.summary_bullets || [];
  modalSummary.innerHTML = '';
  summaryBullets.forEach(bullet => {
    const div = document.createElement('div');
    div.className = 'summary-bullet';
    div.textContent = bullet;
    modalSummary.appendChild(div);
  });

  // ===== METHODOLOGY =====
  modalMethodology.textContent = paper.methodology || 'Not available';

  // ===== FINDINGS =====
  const findings = paper.findings || [];
  modalFindings.innerHTML = '';
  findings.forEach(finding => {
    const li = document.createElement('li');
    li.textContent = finding;
    modalFindings.appendChild(li);
  });

  // ===== IMPLICATIONS =====
  modalImplications.textContent = paper.implications || 'Not available';

  // ===== ABSTRACT (at bottom) =====
  modalAbstract.textContent = paper.abstract || 'No abstract available';

  // ===== ENTITIES =====
  const entities = paper.entities || [];
  modalEntities.innerHTML = '';
  
  // Group entities by type
  const entityTypes = {};
  entities.forEach(ent => {
    if (!entityTypes[ent.type]) entityTypes[ent.type] = [];
    entityTypes[ent.type].push(ent);
  });

  for (const [type, ents] of Object.entries(entityTypes)) {
    const typeDiv = document.createElement('div');
    typeDiv.className = 'entity-type-group';
    typeDiv.innerHTML = `<strong>${type}:</strong>`;
    
    const tagsDiv = document.createElement('div');
    tagsDiv.className = 'entity-tags';
    ents.slice(0, 5).forEach(ent => {
      const tag = document.createElement('span');
      tag.className = 'entity-tag';
      tag.textContent = ent.name;
      tagsDiv.appendChild(tag);
    });
    
    typeDiv.appendChild(tagsDiv);
    modalEntities.appendChild(typeDiv);
  }

  // ===== RELATIONSHIPS =====
  const relationships = paper.relationships || [];
  modalRelationships.innerHTML = '';
  relationships.slice(0, 5).forEach(rel => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${escapeHtml(rel.source)}</strong> 
                     <span class="rel-type">${escapeHtml(rel.relation_type)}</span> 
                     <strong>${escapeHtml(rel.target)}</strong>`;
    modalRelationships.appendChild(li);
  });

  // ===== PUBMED LINK & COPY =====
  if (paper.pmid) {
    modalPubmedLink.href = `https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/`;
  } else {
    modalPubmedLink.href = '#';
  }

  modalCopyBtn.onclick = () => {
    const text = [
      `Title: ${paper.title}`,
      `Authors: ${paper.authors}`,
      `Journal: ${paper.journal}`,
      `Relevance: ${paper.relevance || 'TANGENTIAL'} (${(paper.title_focus * 100).toFixed(0)}% title focus)`,
      ``,
      `Summary:`,
      ...summaryBullets,
      ``,
      `Methodology: ${paper.methodology}`,
      ``,
      `Key Findings:`,
      ...findings,
      ``,
      `Implications: ${paper.implications}`
    ].join('\n');

    navigator.clipboard.writeText(text).then(() => {
      alert('Summary copied to clipboard!');
    });
  };

  modal.classList.remove('hidden');
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  emptyState.classList.remove('hidden');
}

async function handleExport() {
  if (!currentPapers.length) {
    showError('No data to export.');
    return;
  }

  // Create CSV with all extracted data
  const rows = ['Title,Authors,Journal,PMID,Relevance,TitleFocus,Summary,Methodology,Findings,Implications,Entities,Relationships'];
  
  currentPapers.forEach(paper => {
    const summary = (paper.summary_bullets || []).join('; ');
    const methodology = paper.methodology || '';
    const findings = (paper.findings || []).join('; ');
    const implications = paper.implications || '';
    const entities = (paper.entities || []).map(e => e.name).join('; ');
    const relationships = (paper.relationships || []).map(r => `${r.source}→${r.relation_type}→${r.target}`).join('; ');
    const relevance = paper.relevance || 'UNKNOWN';
    const titleFocus = (paper.title_focus * 100).toFixed(0);
    
    const row = [
      escapeCSV(paper.title || ''),
      escapeCSV(paper.authors || ''),
      escapeCSV(paper.journal || ''),
      paper.pmid || '',
      relevance,
      titleFocus,
      escapeCSV(summary),
      escapeCSV(methodology),
      escapeCSV(findings),
      escapeCSV(implications),
      escapeCSV(entities),
      escapeCSV(relationships)
    ].map(field => `"${field}"`).join(',');
    
    rows.push(row);
  });

  const csv = rows.join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'pubmed_mining_results.csv';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function escapeHtml(str) {
  if (str == null) return '';
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

function escapeCSV(str) {
  if (!str) return '';
  return String(str).replace(/"/g, '""');
}