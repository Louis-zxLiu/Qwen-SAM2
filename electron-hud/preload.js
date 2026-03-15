const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  onCaptureResult: (callback) => ipcRenderer.on('capture-result', (_event, value) => callback(value)),
  hideWindow: () => ipcRenderer.send('hide-window'),
  analyzeScreen: async (data) => {
    // We can use fetch/axios here or expose it. 
    // Since we enabled nodeIntegration: false, we can't require axios in renderer.
    // But browser fetch API works fine.
    // Or we can proxy via main process.
    // Let's use browser fetch in renderer.
  }
});
