const API_BASE = '';

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(btn => btn.classList.remove('active'));

    document.getElementById(tabName + 'Tab').classList.add('active');

    document.querySelectorAll('.tab').forEach(btn => {
        const onclickAttr = btn.getAttribute('onclick') || '';
        if (onclickAttr.includes(`'${tabName}'`)) {
            btn.classList.add('active');
        }
    });

    if (tabName === 'upload') loadPapersList();
}

async function updateStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();

        document.getElementById('statusText').textContent =
            data.vectorstore_loaded ? 'System Ready' : 'Indexing Required';
        document.getElementById('paperCount').textContent =
            `${data.papers_count} documents indexed`;
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

        html += `</div>`;
        resultsDiv.innerHTML = html;

    } catch (error) {
        resultsDiv.innerHTML = `<div class="error">Query execution failed: ${error.message}</div>`;
    }
}

async function clearAllData() {
    if (!confirm('⚠️ WARNING: This will permanently delete all uploaded documents and the vector store. This action cannot be undone. Continue?')) {
        return;
    }

    if (!confirm('Are you absolutely sure? All documents will be lost.')) {
        return;
    }

    const statusDiv = document.getElementById('clearStatus');
    statusDiv.innerHTML = '<div class="loading">Clearing all data...</div>';

    try {
        const response = await fetch(`${API_BASE}/api/clear-all`, { method: 'POST' });
        const data = await response.json();

        if (data.status === 'success') {
            statusDiv.innerHTML = `
                <div class="success">
                    <strong>Data Cleared</strong><br>
                    ${data.documents_removed} documents removed<br>
                    Vector store deleted
                </div>
            `;
            setTimeout(() => location.reload(), 2000);
        } else {
            statusDiv.innerHTML = `<div class="error">Failed to clear data</div>`;
        }
    } catch (error) {
        statusDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
}

async function showStats() {
    try {
        const response = await fetch(`${API_BASE}/api/vectorstore-stats`);
        const data = await response.json();

        const statsDiv = document.getElementById('perfStats');

        if (data.status === 'loaded') {
            statsDiv.innerHTML = `
                <div class="stat-box" style="margin-top: 16px;">
                    <p><strong>Total Chunks:</strong> ${data.total_chunks}</p>
                    <p><strong>Collection:</strong> ${data.collection_name}</p>
                    <p><strong>Embedding Model:</strong> ${data.embedding_model}</p>
                    <p><strong>Query Cache Size:</strong> ${data.cache_size} / ${data.max_workers * 25}</p>
                    <p><strong>Parallel Workers:</strong> ${data.max_workers}</p>
                    <p><strong>Batch Size:</strong> ${data.batch_size}</p>
                </div>
            `;
        } else {
            statsDiv.innerHTML = '<p class="help-text">No statistics available. Index documents first.</p>';
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    updateStatus();
    loadPapersList();
    setInterval(updateStatus, 30000);
});