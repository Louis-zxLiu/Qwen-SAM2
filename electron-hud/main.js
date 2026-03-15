const { app, BrowserWindow, globalShortcut, desktopCapturer, screen, ipcMain } = require('electron');
const path = require('path');

let mainWindow;

function createWindow() {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;

  mainWindow = new BrowserWindow({
    width,
    height,
    frame: false,
    transparent: true,
    fullscreen: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false
    }
  });

  // Ensure window is always on top, even during focus loss
  mainWindow.setAlwaysOnTop(true, 'screen-saver');
  mainWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

  mainWindow.loadFile('index.html');
  // mainWindow.webContents.openDevTools({ mode: 'detach' });

  // Initially hide
  mainWindow.hide();
}

let hasCaptured = false;

// IPC to reset capture state (called when user explicitly closes or starts new)
ipcMain.on('reset-capture', () => {
  hasCaptured = false;
});

app.on('browser-window-blur', () => {
  // Only show if window was already visible (prevents showing on startup focus loss)
  if (mainWindow && mainWindow.isVisible()) {
    mainWindow.showInactive();
  }
});

app.whenReady().then(() => {
  createWindow();
  
  const ret = globalShortcut.register('Alt+X', () => {
    if (!hasCaptured || (mainWindow && mainWindow.isVisible())) {
      // If we haven't captured yet OR the window is already visible, capture NEW
      captureAndShow();
      hasCaptured = true;
    } else {
      // If window is hidden and we have an old capture, just RESTORE visibility
      mainWindow.show();
      mainWindow.focus();
    }
  });

  if (!ret) {
    console.log('Registration failed');
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});

ipcMain.on('hide-window', () => {
  if (mainWindow) mainWindow.hide();
});

async function captureAndShow() {
  if (!mainWindow) return;

  try {
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width, height } = primaryDisplay.size; // Use full size, not workAreaSize

    // Get sources
    const sources = await desktopCapturer.getSources({ 
      types: ['screen'], 
      thumbnailSize: { width, height } 
    });

    // Find primary screen (usually first or matching id)
    // On Windows, 'screen:0:0' is usually primary
    const source = sources.find(s => s.display_id === primaryDisplay.id.toString()) || sources[0];

    if (source) {
      const image = source.thumbnail.toDataURL();
      mainWindow.webContents.send('capture-result', image);
      mainWindow.show();
      mainWindow.focus();
    }
  } catch (e) {
    console.error("Capture failed:", e);
  }
}
