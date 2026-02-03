const API_BASE = '';
let currentResponse = null;

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(btn => btn.classList.remove('active'));
    
    document.getElementById(tabName + 'Tab').classList.add('active');
    event.target.classList.add('active');
    
    if (tabName === 'upload') loadPapersList();
    else if (tabName === 'evaluate') setupEvaluationForm();
    else if (tabName === 'stats') loadStatistics();
}

async function updateStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        
        document.getElementById('statusText').textContent = 
            data.vectorstore_loaded ? 'System Ready' : 'Indexing Required';
        document.getElementById('paperCount').textContent = 
            `${data.papers_count} documents indexed`;
        document.getElementById('evalCount').textContent = 
            `${data.evaluations_count} evaluations completed`;
    } catch (error) {
        console.error('Status check failed:', error);
    }
}

async function uploadFiles() {
    const fileInput = document.getElementById('fileInput');
    const files = fileInput.files;
    
    if (files.length === 0) {
        alert('Please select PDF files to upload');
        return;
    }
    
    const statusDiv = document.getElementById('ingestionStatus');
    statusDiv.innerHTML = '<div class="loading">Uploading documents...</div>';
    
    let successCount = 0;
    let failCount = 0;
    
    for (let file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(`${API_BASE}/api/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error('Upload failed');
            
            successCount++;
            statusDiv.innerHTML += `<div class="success">✓ ${file.name}</div>`;
        } catch (error) {
            failCount++;
            statusDiv.innerHTML += `<div class="error">✗ ${file.name} - Upload failed</div>`;
        }
    }
    
    if (successCount > 0) {
        statusDiv.innerHTML = `<div class="success">Successfully uploaded ${successCount} document(s)</div>` + statusDiv.innerHTML;
    }
    if (failCount > 0) {
        statusDiv.innerHTML = `<div class="error">Failed to upload ${failCount} document(s)</div>` + statusDiv.innerHTML;
    }
    
    fileInput.value = '';
    loadPapersList();
    updateStatus();
}

async function loadPapersList() {
    try {
        const response = await fetch(`${API_BASE}/api/papers`);
        const data = await response.json();
        
        const container = document.getElementById('uploadedPapers');
        if (data.count === 0) {
            container.innerHTML = '<div class="info">No documents uploaded. Please upload PDF files to begin.</div>';
        } else {
            container.innerHTML = `
                <h3>Uploaded Documents (${data.count})</h3>
                <ul>${data.papers.map(p => `<li>${p}</li>`).join('')}</ul>
            `;
        }
    } catch (error) {
        console.error('Failed to load papers:', error);
    }
}

async function ingestDocuments() {
    const statusDiv = document.getElementById('ingestionStatus');
    statusDiv.innerHTML = '<div class="loading">Indexing documents... This process may take several minutes depending on the number of documents.</div>';
    
    try {
        const response = await fetch(`${API_BASE}/api/ingest`, { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'success') {
            statusDiv.innerHTML = `
                <div class="success">
                    <strong>Indexing Complete</strong><br>
                    Documents Processed: ${data.processed_documents}<br>
                    Text Chunks Created: ${data.total_chunks}<br>
                    Status: Ready for queries
                </div>
            `;
        } else {
            statusDiv.innerHTML = `<div class="error">Indexing failed: ${data.message}</div>`;
        }
        
        updateStatus();
    } catch (error) {
        statusDiv.innerHTML = `<div class="error">System error: ${error.message}</div>`;
    }
}

async function queryRAG() {
    const question = document.getElementById('questionInput').value.trim();
    const topK = parseInt(document.getElementById('topKInput').value);
    
    if (!question) {
        alert('Please enter a research question');
        return;
    }
    
    const resultsDiv = document.getElementById('queryResults');
    resultsDiv.innerHTML = '<div class="loading">Executing query across indexed documents...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, top_k: topK })
        });
        
        const data = await response.json();
        currentResponse = data;
        
        let html = `
            <div class="result-card">
                <p><strong>Response ID:</strong> <span class="response-id">${data.response_id}</span></p>
                
                <h3>Query</h3>
                <p>${data.question}</p>
                
                <h3>System Response</h3>
                <p>${data.answer}</p>
                
                <h3>Source Citations (${data.cited_sources.length})</h3>
        `;
        
        data.cited_sources.forEach((source, idx) => {
            html += `
                <div class="source-card">
                    <strong>Source ${idx + 1}:</strong> ${source.document}
                    ${source.page ? ` | Page ${source.page}` : ''}
                    <p style="margin-top: 8px; font-size: 13px; color: var(--text-secondary);">${source.chunk_content.substring(0, 300)}...</p>
                </div>
            `;
        });
        
        html += `
                <div style="margin-top: 24px;">
                    <button onclick="showTab('evaluate')" class="btn btn-primary">Evaluate Response</button>
                </div>
            </div>
        `;
        
        resultsDiv.innerHTML = html;
        
    } catch (error) {
        resultsDiv.innerHTML = `<div class="error">Query execution failed: ${error.message}</div>`;
    }
}

function setupEvaluationForm() {
    if (!currentResponse) {
        document.getElementById('evaluationForm').style.display = 'none';
        document.getElementById('noResponseMessage').style.display = 'block';
        return;
    }
    
    document.getElementById('evaluationForm').style.display = 'block';
    document.getElementById('noResponseMessage').style.display = 'none';
    
    document.getElementById('evalResponseId').textContent = currentResponse.response_id;
    document.getElementById('evalQuestion').textContent = currentResponse.question;
    document.getElementById('evalAnswer').textContent = currentResponse.answer;
    
    const sourcesHtml = currentResponse.cited_sources.map((s, idx) => `
        <div class="source-card">
            <strong>${idx + 1}. ${s.document}</strong> ${s.page ? `| Page ${s.page}` : ''}
        </div>
    `).join('');
    
    document.getElementById('evalSources').innerHTML = sourcesHtml;
}

async function submitEvaluation() {
    if (!currentResponse) {
        alert('No response available for evaluation');
        return;
    }
    
    const relevance = parseInt(document.getElementById('relevanceScore').value);
    const hallucination = parseInt(document.getElementById('hallucinationScore').value);
    const completeness = parseInt(document.getElementById('completenessScore').value);
    const faithfulness = parseInt(document.getElementById('faithfulnessScore').value);
    const notes = document.getElementById('evalNotes').value;
    
    const citedSourcesPlain = currentResponse.cited_sources.map(source => ({
        document: source.document,
        page: source.page,
        chunk_content: source.chunk_content,
        relevance_score: source.relevance_score
    }));
    
    const evaluation = {
        response_id: currentResponse.response_id,
        question: currentResponse.question,
        answer: currentResponse.answer,
        cited_sources: citedSourcesPlain,
        relevance_score: relevance,
        hallucination_score: hallucination,
        completeness_score: completeness,
        faithfulness_score: faithfulness,
        notes: notes || null
    };
    
    try {
        const response = await fetch(`${API_BASE}/api/evaluate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(evaluation)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(JSON.stringify(errorData));
        }
        
        const data = await response.json();
        
        alert(`Evaluation saved successfully\nTotal evaluations: ${data.total_evaluations}`);
        
        // Reset form
        document.getElementById('relevanceScore').value = 3;
        document.getElementById('hallucinationScore').value = 3;
        document.getElementById('completenessScore').value = 3;
        document.getElementById('faithfulnessScore').value = 3;
        document.getElementById('evalNotes').value = '';
        currentResponse = null;
        
        document.getElementById('evaluationForm').style.display = 'none';
        document.getElementById('noResponseMessage').style.display = 'block';
        
        updateStatus();
        
    } catch (error) {
        console.error('Evaluation error:', error);
        alert(`Evaluation submission failed: ${error.message}`);
    }
}

async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE}/api/evaluations`);
        const data = await response.json();
        
        const statsDiv = document.getElementById('statsContent');
        
        if (data.total_evaluations === 0) {
            statsDiv.innerHTML = '<div class="info">No evaluation data available. Complete query evaluations to view analytics.</div>';
            return;
        }
        
        let html = `
            <div class="stat-box">
                <h3>Summary Statistics</h3>
                <p><strong>Total Evaluations:</strong> ${data.total_evaluations}</p>
                <p><strong>Average Relevance Score:</strong> ${data.avg_relevance.toFixed(2)} / 5.0</p>
                <p><strong>Average Citation Accuracy:</strong> ${data.avg_hallucination.toFixed(2)} / 5.0</p>
                ${data.avg_completeness ? `<p><strong>Average Completeness:</strong> ${data.avg_completeness.toFixed(2)} / 5.0</p>` : ''}
                ${data.avg_faithfulness ? `<p><strong>Average Faithfulness:</strong> ${data.avg_faithfulness.toFixed(2)} / 5.0</p>` : ''}
            </div>
            
            <h3>Evaluation History</h3>
        `;
        
        data.evaluations.forEach((eval, idx) => {
            html += `
                <div class="eval-item">
                    <p><strong>Evaluation #${idx + 1}</strong> | ID: <span class="response-id">${eval.response_id}</span></p>
                    <p><strong>Query:</strong> ${eval.question}</p>
                    <p><strong>Metrics:</strong> 
                       Relevance: ${eval.relevance_score}/5 | 
                       Accuracy: ${eval.hallucination_score}/5
                       ${eval.completeness_score ? ` | Completeness: ${eval.completeness_score}/5` : ''}
                       ${eval.faithfulness_score ? ` | Faithfulness: ${eval.faithfulness_score}/5` : ''}
                    </p>
                    ${eval.notes ? `<p><strong>Notes:</strong> ${eval.notes}</p>` : ''}
                    <p style="font-size: 13px; color: var(--text-secondary); margin-top: 8px;">
                        Sources: ${eval.cited_sources.map(s => s.document).join(', ')}
                    </p>
                </div>
            `;
        });
        
        statsDiv.innerHTML = html;
        
    } catch (error) {
        document.getElementById('statsContent').innerHTML = 
            `<div class="error">Failed to load analytics: ${error.message}</div>`;
    }
}

async function clearAllData() {
    if (!confirm('⚠️ WARNING: This will permanently delete all uploaded documents and the vector store. This action cannot be undone. Continue?')) {
        return;
    }
    
    if (!confirm('Are you absolutely sure? All documents and evaluations will be lost.')) {
        return;
    }
    
    const statusDiv = document.getElementById('clearStatus');
    statusDiv.innerHTML = '<div class="loading">Clearing all data...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/api/clear-all`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            statusDiv.innerHTML = `
                <div class="success">
                    <strong>Data Cleared</strong><br>
                    ${data.documents_removed} documents removed<br>
                    Vector store deleted
                </div>
            `;
            
            // Refresh the page
            setTimeout(() => {
                location.reload();
            }, 2000);
        } else {
            statusDiv.innerHTML = `<div class="error">Failed to clear data</div>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    updateStatus();
    loadPapersList();
    setInterval(updateStatus, 30000);
});
