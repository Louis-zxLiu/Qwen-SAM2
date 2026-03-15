const screenshot = document.getElementById('screenshot');
const overlay = document.getElementById('overlay');
const tooltip = document.getElementById('tooltip');
const tooltipContent = document.getElementById('tooltip-content');
const pluginActions = document.getElementById('plugin-actions');
const loading = document.getElementById('loading');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');

// Settings Elements
const settingsPanel = document.getElementById('settings-panel');
const settingsToggleBtn = document.getElementById('settings-toggle-btn');
const apiKeyInput = document.getElementById('api-key-input');
const baseUrlInput = document.getElementById('base-url-input');
const saveSettingsBtn = document.getElementById('save-settings-btn');
const closeSettingsBtn = document.getElementById('close-settings-btn');

// Window Controls
const hideBtn = document.getElementById('hide-btn');
const closeBtn = document.getElementById('close-btn');

let currentImageBase64 = null;
let currentBBox = null;
let chatHistory = [];

// Load Settings from LocalStorage
const savedApiKey = localStorage.getItem('qwen_api_key');
const savedBaseUrl = localStorage.getItem('qwen_base_url');

if (savedApiKey) apiKeyInput.value = savedApiKey;
if (savedBaseUrl) baseUrlInput.value = savedBaseUrl;

// Fetch global config from backend on startup
async function syncConfig() {
    try {
        const res = await fetch('http://localhost:8000/system/config');
        if (res.ok) {
            const config = await res.json();
            if (config.api_key) {
                apiKeyInput.value = config.api_key;
                localStorage.setItem('qwen_api_key', config.api_key);
            }
            if (config.base_url) {
                baseUrlInput.value = config.base_url;
                localStorage.setItem('qwen_base_url', config.base_url);
            }
            if (config.qwen_model) {
                localStorage.setItem('qwen_model', config.qwen_model);
            }
            console.log("Config synced from backend:", config);
        }
    } catch (e) {
        console.error("Failed to sync config:", e);
    }
}

// Initial Sync
syncConfig();

// Settings UI Logic
settingsToggleBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Prevent clicking through to image
    settingsPanel.style.display = settingsPanel.style.display === 'block' ? 'none' : 'block';
});

closeSettingsBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    settingsPanel.style.display = 'none';
});

saveSettingsBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const apiKey = apiKeyInput.value.trim();
    const baseUrl = baseUrlInput.value.trim();
    
    localStorage.setItem('qwen_api_key', apiKey);
    localStorage.setItem('qwen_base_url', baseUrl);
    
    settingsPanel.style.display = 'none';
    alert('Settings Saved!');
});

// Window Control Logic
hideBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    window.electronAPI.hideWindow();
});

closeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    window.electronAPI.hideWindow();
    window.electronAPI.resetCapture(); // Tell main process next Alt+X should capture
    clearOverlay(); // Clear content on close
});

// Prevent clicks inside settings panel from triggering image click
settingsPanel.addEventListener('click', (e) => {
    e.stopPropagation();
});

// Handle Capture Result from Main Process
window.electronAPI.onCaptureResult((imageData) => {
    console.log("Received capture result");
    screenshot.src = imageData;
    currentImageBase64 = imageData.split(',')[1]; // Remove header
    
    // Reset UI
    clearOverlay();
    chatHistory = []; // Reset chat history on new screenshot
    currentBBox = null;
});

// Handle Esc Key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        window.electronAPI.hideWindow();
        clearOverlay();
    }
});

function clearOverlay() {
    overlay.innerHTML = ''; // Remove all SVG paths
    tooltip.style.display = 'none';
    loading.style.display = 'none';
    loading.classList.remove('active');
    tooltipContent.innerHTML = '';
    pluginActions.innerHTML = '';
    pluginActions.style.display = 'none';
    chatInput.value = '';
}

// Handle Click
screenshot.addEventListener('click', async (e) => {
    if (!currentImageBase64) return;
    
    // Ignore clicks if settings panel is open
    if (settingsPanel.style.display === 'block') return;
    
    // Ignore clicks on tooltip itself (handled by stopPropagation on tooltip)
    
    // Show loading
    loading.style.left = (e.clientX - 20) + 'px';
    loading.style.top = (e.clientY - 20) + 'px';
    loading.style.display = 'block';
    
    // Calculate coordinates relative to original image size
    const scaleX = screenshot.naturalWidth / screenshot.clientWidth;
    const scaleY = screenshot.naturalHeight / screenshot.clientHeight;
    
    const clickX = Math.round(e.offsetX * scaleX);
    const clickY = Math.round(e.offsetY * scaleY);
    
    console.log(`Click at (${clickX}, ${clickY}) with scale (${scaleX}, ${scaleY})`);

    try {
        const apiKey = localStorage.getItem('qwen_api_key') || '';
        const baseUrl = localStorage.getItem('qwen_base_url') || '';
        const qwenModel = localStorage.getItem('qwen_model') || 'Qwen/Qwen2-VL-7B-Instruct';

        const response = await fetch('http://localhost:8000/analyze/screen', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: currentImageBase64,
                click_x: clickX,
                click_y: clickY,
                mode: 'identify', // Or 'segment'
                api_key: apiKey,
                base_url: baseUrl,
                qwen_model: qwenModel
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Analysis Result:", data);
        
        // Hide loading
        loading.style.display = 'none';
        
        // Draw Mask
        if (data.svg_path) {
            drawMask(data.svg_path, scaleX, scaleY);
        }
        
        // Store BBox for chat
        if (data.bbox) {
            currentBBox = data.bbox;
        }

        // Show Tooltip
        if (data.description) {
            showTooltip(e.clientX, e.clientY, data.description);
            // Add initial description to chat history
            chatHistory = [
                { role: 'user', content: 'Identify this object.' },
                { role: 'assistant', content: data.description }
            ];
            
            // Render Plugins
            if (data.plugins && data.plugins.length > 0) {
                renderPlugins(data.plugins, data.description);
            }
        }

    } catch (error) {
        console.error("Analysis failed:", error);
        loading.style.display = 'none';
        showTooltip(e.clientX, e.clientY, "Error: " + error.message);
    }
});

// Chat Logic
async function sendChat() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    // Display user message immediately
    appendChatMessage('user', text);
    chatInput.value = '';
    
    try {
        const apiKey = localStorage.getItem('qwen_api_key') || '';
        const baseUrl = localStorage.getItem('qwen_base_url') || '';
        const qwenModel = localStorage.getItem('qwen_model') || 'Qwen/Qwen2-VL-7B-Instruct';
        
        // Prepare context
        // We need to send the image crop again or handle it statefully in backend.
        // For simplicity, let's re-send the request to a new chat endpoint
        
        const response = await fetch('http://localhost:8000/analyze/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: currentImageBase64,
                bbox: currentBBox,
                messages: [...chatHistory, { role: 'user', content: text }],
                api_key: apiKey,
                base_url: baseUrl,
                qwen_model: qwenModel
            })
        });
        
        const data = await response.json();
        
        if (data.reply) {
            appendChatMessage('assistant', data.reply);
            chatHistory.push({ role: 'user', content: text });
            chatHistory.push({ role: 'assistant', content: data.reply });
        }
        
    } catch (e) {
        appendChatMessage('assistant', "Error: " + e.message);
    }
}

function appendChatMessage(role, text) {
    const div = document.createElement('div');
    div.className = `chat-message ${role === 'user' ? 'user' : 'ai'}`;
    div.textContent = text;
    tooltipContent.appendChild(div);
    // Scroll to bottom
    const tooltip = document.getElementById('tooltip');
    tooltip.scrollTop = tooltip.scrollHeight;
}

sendBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    sendChat();
});

chatInput.addEventListener('keydown', (e) => {
    e.stopPropagation(); // Allow typing
    if (e.key === 'Enter') {
        sendChat();
    }
});

// Prevent clicks inside tooltip from closing it or triggering image click
tooltip.addEventListener('click', (e) => {
    e.stopPropagation();
});

function renderPlugins(plugins, description) {
    pluginActions.innerHTML = '';
    pluginActions.style.display = 'flex';
    
    plugins.forEach(p => {
        const btn = document.createElement('button');
        btn.className = 'plugin-btn';
        btn.innerHTML = `${p.icon} ${p.name}`;
        btn.title = p.description;
        
        btn.onclick = async (e) => {
            e.stopPropagation();
            appendChatMessage('user', `Run plugin: ${p.name}`);
            
            try {
                const response = await fetch('http://localhost:8000/plugin/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        plugin_id: p.id,
                        context: {
                            description: description,
                            // Add more context if needed
                        }
                    })
                });
                
                const result = await response.json();
                if (result.status === 'success') {
                    appendChatMessage('assistant', `Plugin Result: ${result.result}`);
                } else {
                    appendChatMessage('assistant', `Plugin Error: ${result.message}`);
                }
            } catch (err) {
                appendChatMessage('assistant', `Error: ${err.message}`);
            }
        };
        
        pluginActions.appendChild(btn);
    });
}

function drawMask(svgPathData, scaleX, scaleY) {
    // Create SVG Path
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", svgPathData);
    path.setAttribute("class", "mask-path");
    
    path.setAttribute("transform", `scale(${1/scaleX}, ${1/scaleY})`);
    
    overlay.innerHTML = '';
    overlay.appendChild(path);
}

function showTooltip(x, y, text) {
    tooltipContent.innerHTML = ''; // Clear previous content
    const initialMsg = document.createElement('div');
    initialMsg.className = 'chat-message ai';
    initialMsg.innerHTML = text; // Allow HTML if backend returns it, otherwise textContent
    tooltipContent.appendChild(initialMsg);
    
    tooltip.style.display = 'flex'; // Use flex for column layout
    
    // Position logic to keep inside screen
    const rect = tooltip.getBoundingClientRect();
    let left = x + 20;
    let top = y + 20;
    
    if (left + rect.width > window.innerWidth) {
        left = x - rect.width - 20;
    }
    if (top + rect.height > window.innerHeight) {
        top = y - rect.height - 20;
    }
    
    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
}
