const screenshot = document.getElementById('screenshot');
const overlay = document.getElementById('overlay');
const tooltip = document.getElementById('tooltip');
const loading = document.getElementById('loading');

// Settings Elements
const settingsPanel = document.getElementById('settings-panel');
const settingsToggleBtn = document.getElementById('settings-toggle-btn');
const apiKeyInput = document.getElementById('api-key-input');
const baseUrlInput = document.getElementById('base-url-input');
const saveSettingsBtn = document.getElementById('save-settings-btn');
const closeSettingsBtn = document.getElementById('close-settings-btn');

let currentImageBase64 = null;

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
}

// Handle Click
screenshot.addEventListener('click', async (e) => {
    if (!currentImageBase64) return;

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
        // Add model_name support if saved (currently not in settings UI but might come from global config)
        // Ideally we should save model_name in localStorage too when syncing config.
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
        
        // Show Tooltip
        if (data.description) {
            showTooltip(e.clientX, e.clientY, data.description);
        }

    } catch (error) {
        console.error("Analysis failed:", error);
        loading.style.display = 'none';
        showTooltip(e.clientX, e.clientY, "Error: " + error.message);
    }
});

function drawMask(svgPathData, scaleX, scaleY) {
    // Create SVG Path
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", svgPathData);
    path.setAttribute("class", "mask-path");
    
    // Apply transform to scale mask back to screen coordinates
    // SVG coordinate system matches the image natural size if we don't scale.
    // Wait, the SVG is overlaid on the screen (CSS pixels).
    // The path data is in image pixels (natural size).
    // So we need to scale the path DOWN to screen size.
    // transform="scale(1/scaleX, 1/scaleY)"
    
    path.setAttribute("transform", `scale(${1/scaleX}, ${1/scaleY})`);
    
    // Clear previous paths? Maybe keep them? Let's clear for now.
    overlay.innerHTML = '';
    overlay.appendChild(path);
}

function showTooltip(x, y, text) {
    tooltip.innerHTML = text; // Allow HTML/Markdown if parsed
    tooltip.style.display = 'block';
    
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
