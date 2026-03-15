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
      webSecurity: false // Allow loading local resources if needed
    }
  });

  mainWindow.loadFile('index.html');
  // mainWindow.webContents.openDevTools({ mode: 'detach' });

  // Initially hide
  mainWindow.hide();
}

app.whenReady().then(() => {
  createWindow();

  // Register Alt+X shortcut
  const ret = globalShortcut.register('Alt+X', () => {
    captureAndShow();
  });

  if (!ret) {
    console.log('Registration failed');
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
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
