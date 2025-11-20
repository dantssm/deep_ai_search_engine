let ws = null; 
let wsReady = false; 
let currentSessionId = null;
let currentPlan = null;
let currentResult = null;
let currentDepth = 'standard';
let logsVisible = false;

document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    setupButtons();
});

function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/search`);

    ws.onopen = () => {
        console.log('Connected to server');
        wsReady = true;
    };

    ws.onclose = () => {
        console.log('Disconnected. Reconnecting in 2 seconds...');
        wsReady = false;
        currentSessionId = null;
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (error) => {
        console.error('Connection error:', error);
        wsReady = false;
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
    };
}

function handleServerMessage(data) {
    switch (data.type) {
        case 'session_id':
            currentSessionId = data.session_id;
            console.log('Session ID received:', currentSessionId);
            break;

        case 'plan_generated':
        case 'plan_refined':
            currentPlan = data.plan;
            renderPlan(data.plan);
            if (data.type === 'plan_refined') {
                document.getElementById('refineInputContainer').classList.remove('active');
            }
            break;

        case 'status':
            updateStatusLabel(data.message);
            addLogEntry(data.message);
            break;

        case 'synthesis_chunk':
            appendToOutput(data.chunk);
            break;

        case 'complete':
            currentResult = data.result;
            updateStatusLabel('Research Complete');
            renderFinalResult(data.result);
            document.getElementById('exportBtn').classList.remove('hidden');
            break;

        case 'error':
            alert(`Error: ${data.message}`);
            updateStatusLabel('Error occurred');
            break;
    }
}

function sendToServer(messageObject) {
    if (!ws || !wsReady) {
        alert('Not connected to server. Please wait...');
        return;
    }
    ws.send(JSON.stringify(messageObject));
}

function setupButtons() {
    document.getElementById('createPlanBtn').onclick = handleCreatePlan;
    document.getElementById('mainInput').onkeypress = (e) => {
        if (e.key === 'Enter') handleCreatePlan();
    };
    document.getElementById('depthBtn').onclick = () => toggleDepth('depthBtn');
    document.getElementById('surpriseBtn').onclick = fillRandomTopic;
    document.getElementById('aboutBtn').onclick = () => showScreen('about');

    document.getElementById('editPlanBtn').onclick = () => showScreen('main');
    document.getElementById('initializeBtn').onclick = startResearch;
    document.getElementById('depthBtnPlan').onclick = () => toggleDepth('depthBtnPlan');
    document.getElementById('refineBtn').onclick = () => {
        document.getElementById('refineInputContainer').classList.toggle('active');
    };
    document.getElementById('sendRefineBtn').onclick = handleRefinePlan;

    document.getElementById('toggleLogsBtn').onclick = toggleLogsPanel;
    document.getElementById('minimizeLogsBtn').onclick = closeLogsPanel;

    document.getElementById('exportBtn').onclick = downloadReport;

    document.getElementById('closeAboutBtn').onclick = () => {
        showScreen('main');
    };
}

function showScreen(screenName) {
    document.querySelectorAll('.screen').forEach(el => el.classList.remove('active'));
    
    const screenId = screenName + 'Screen';
    const screenEl = document.getElementById(screenId);
    if (screenEl) {
        screenEl.classList.add('active');
    }

    const query = document.getElementById('mainInput').value;
    if (screenName === 'plan') document.getElementById('planInput').value = query;
    if (screenName === 'research') document.getElementById('researchInput').value = query;
}

function toggleDepth(buttonId) {
    currentDepth = (currentDepth === 'standard') ? 'deep' : 'standard';
    
    document.getElementById('depthBtn').textContent = currentDepth;
    document.getElementById('depthBtnPlan').textContent = currentDepth;
}

function fillRandomTopic() {
    const topics = [
        "How does quantum entanglement work?",
        "Future of fusion energy",
        "History of the internet",
        "What causes Northern Lights?",
        "CRISPR gene editing explained",
        "How do neural networks learn?",
        "Impact of climate change on oceans"
    ];
    const random = topics[Math.floor(Math.random() * topics.length)];
    document.getElementById('mainInput').value = random;
}

function handleCreatePlan() {
    const query = document.getElementById('mainInput').value.trim();
    if (!query) return alert('Please enter a topic.');

    sendToServer({
        type: 'create_plan',
        query: query,
        depth: currentDepth
    });

    showScreen('plan');
    document.getElementById('planText').innerHTML = '<em>Generating research plan...</em>';
}

function handleRefinePlan() {
    const feedback = document.getElementById('refineInput').value.trim();
    const query = document.getElementById('mainInput').value;

    if (!feedback) return;

    sendToServer({
        type: 'refine_plan',
        query: query,
        depth: currentDepth,
        feedback: feedback,
        current_plan: currentPlan
    });

    document.getElementById('planText').innerHTML = '<em>Refining plan...</em>';
    document.getElementById('refineInput').value = '';
}

function startResearch() {
    if (!currentPlan) return;

    showScreen('research');
    updateStatusLabel('Starting research...');
    document.getElementById('outputContent').textContent = '';

    sendToServer({
        type: 'execute_research',
        plan: currentPlan,
        enable_streaming: true
    });
}

function renderPlan(plan) {
    let html = '';
    
    if (plan.reasoning) {
        html += `<div class="plan-section"><strong>Strategy:</strong><br>${escapeHtml(plan.reasoning)}</div>`;
    }
    
    if (plan.sub_topics && plan.sub_topics.length) {
        html += '<div class="plan-section"><strong>Sub-topics:</strong><ul>';
        plan.sub_topics.forEach(topic => {
            html += `<li>${escapeHtml(topic)}</li>`;
        });
        html += '</ul></div>';
    }
    
    document.getElementById('planText').innerHTML = html;
}

function renderFinalResult(result) {
    const outputDiv = document.getElementById('outputContent');
    
    let text = result.report_text || '';
    const sources = result.sources || [];
    const citations = result.citations || [];

    text = text.replace(/\[([0-9,\s]+)\]/g, (match, ids) => {
        const idList = ids.split(',').map(s => s.trim());
        if (idList.length === 1) return match;
        return idList.map(id => `[${id}]`).join('');
    });

    const sourceMap = {};
    sources.forEach(s => sourceMap[s.id] = s);

    text = text.replace(/\[(\d+)\]/g, (match, id) => {
        const s = sourceMap[id];
        if (s) {
            return `<a href="${s.url}" target="_blank" class="source-link" title="${s.title}">[${id}]</a>`;
        }
        return match;
    });

    text = text
        .replace(/^###### (.+)$/gm, '<h6>$1</h6>')
        .replace(/^##### (.+)$/gm, '<h5>$1</h5>')
        .replace(/^#### (.+)$/gm, '<h4>$1</h4>')
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^# (.+)$/gm, '<h1>$1</h1>')
        .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>');

    const lines = text.split('\n');
    let inList = false;
    let processedLines = [];
    
    for (let line of lines) {
        const trimmed = line.trim();
        
        if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
            if (!inList) {
                processedLines.push('<ul>');
                inList = true;
            }
            processedLines.push(`<li>${trimmed.substring(2)}</li>`);
        } else {
            if (inList) {
                processedLines.push('</ul>');
                inList = false;
            }
            if (trimmed) {
                processedLines.push(line);
            }
        }
    }
    
    if (inList) {
        processedLines.push('</ul>');
    }
    
    text = processedLines.join('\n\n');

    text = text
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');

    let refsHtml = '';
    if (citations.length > 0) {
        refsHtml = '<div class="references-section"><h3>References</h3>';
        [...new Set(citations)].sort((a,b) => a-b).forEach(id => {
            const s = sourceMap[id];
            if (s) {
                refsHtml += `
                    <div class="reference-item">
                        <span class="ref-id">[${id}]</span>
                        <a href="${s.url}" target="_blank" class="ref-link">${s.title || 'Untitled'}</a>
                    </div>`;
            }
        });
        refsHtml += '</div>';
    }

    outputDiv.innerHTML = `
        <div class="result-header">Research Complete</div>
        <div class="result-content"><p>${text}</p></div>
        ${refsHtml}
    `;
}

function appendToOutput(textChunk) {
    const el = document.getElementById('outputContent');
    el.textContent += textChunk;
    el.scrollTop = el.scrollHeight;
}

function addLogEntry(msg) {
    const logs = document.getElementById('logsContent');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logs.appendChild(entry);
    logs.scrollTop = logs.scrollHeight;
}

function updateStatusLabel(msg) {
    document.getElementById('statusBar').textContent = `Status: ${msg}`;
}

function toggleLogsPanel() {
    logsVisible = !logsVisible;
    const panel = document.getElementById('logsPanel');
    const btn = document.getElementById('toggleLogsBtn');
    
    if (logsVisible) {
        panel.classList.add('visible');
        btn.textContent = 'Hide Logs';
    } else {
        panel.classList.remove('visible');
        btn.textContent = 'Show Logs';
    }
}

function closeLogsPanel() {
    logsVisible = false;
    document.getElementById('logsPanel').classList.remove('visible');
    document.getElementById('toggleLogsBtn').textContent = 'Show Logs';
}

function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

async function downloadReport() {
    if (!currentSessionId) {
        alert('No active session. Please complete a research first.');
        return;
    }
    
    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `research_report_${new Date().toISOString().slice(0,10)}.md`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } else {
            const errorText = await response.text();
            console.error('Export failed:', errorText);
            alert(`Export failed: ${response.status} - ${errorText}`);
        }
    } catch (e) {
        console.error('Export error:', e);
        alert(`Export error: ${e.message}`);
    }
}