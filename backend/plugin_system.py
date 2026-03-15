import os
import json
import importlib.util
import traceback

class PluginManager:
    def __init__(self, plugin_dir=None):
        if plugin_dir is None:
            # Default to the 'plugins' directory in the same folder as this file
            self.plugin_dir = os.path.join(os.path.dirname(__file__), "plugins")
        else:
            self.plugin_dir = plugin_dir
        self.plugins = {} # id -> {manifest, module}
        self.load_plugins()

    def load_plugins(self):
        """Scan plugin directory and load all valid plugins."""
        if not os.path.exists(self.plugin_dir):
            print(f"[PluginManager] Plugin directory not found: {self.plugin_dir}")
            return

        print(f"[PluginManager] Scanning plugins in {self.plugin_dir}...")
        
        for item in os.listdir(self.plugin_dir):
            item_path = os.path.join(self.plugin_dir, item)
            if os.path.isdir(item_path):
                manifest_path = os.path.join(item_path, "manifest.json")
                main_py_path = os.path.join(item_path, "main.py")
                
                if os.path.exists(manifest_path) and os.path.exists(main_py_path):
                    try:
                        # Load Manifest
                        with open(manifest_path, 'r', encoding='utf-8') as f:
                            manifest = json.load(f)
                        
                        plugin_id = manifest.get("id")
                        if not plugin_id:
                            print(f"[PluginManager] Skipping {item}: Missing 'id' in manifest")
                            continue
                            
                        # Load Module
                        spec = importlib.util.spec_from_file_location(f"plugins.{plugin_id}", main_py_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        if not hasattr(module, "run"):
                            print(f"[PluginManager] Skipping {item}: Missing 'run' function in main.py")
                            continue
                            
                        self.plugins[plugin_id] = {
                            "manifest": manifest,
                            "module": module
                        }
                        print(f"[PluginManager] Loaded: {manifest.get('name')} ({plugin_id})")
                        
                    except Exception as e:
                        print(f"[PluginManager] Error loading {item}: {e}")
                        traceback.print_exc()

    def match_plugins(self, text_description):
        """Return a list of plugins that match the description."""
        matched = []
        text = text_description.lower()
        
        for p_id, plugin in self.plugins.items():
            manifest = plugin["manifest"]
            triggers = manifest.get("triggers", {})
            
            # 1. Keyword Match
            keywords = triggers.get("keywords", [])
            for kw in keywords:
                if kw.lower() in text:
                    matched.append({
                        "id": p_id,
                        "name": manifest.get("name"),
                        "icon": manifest.get("icon", "🔌"),
                        "description": manifest.get("description")
                    })
                    break # Matched one keyword is enough
            
            # 2. Always available (universal plugins)
            if triggers.get("universal", False):
                 matched.append({
                        "id": p_id,
                        "name": manifest.get("name"),
                        "icon": manifest.get("icon", "🔌"),
                        "description": manifest.get("description")
                    })
        
        # Remove duplicates
        unique_matched = []
        seen = set()
        for m in matched:
            if m['id'] not in seen:
                unique_matched.append(m)
                seen.add(m['id'])
                
        return unique_matched

    def execute_plugin(self, plugin_id, context):
        """Execute a specific plugin."""
        if plugin_id not in self.plugins:
            return {"status": "error", "message": "Plugin not found"}
            
        try:
            print(f"[PluginManager] Executing {plugin_id}...")
            result = self.plugins[plugin_id]["module"].run(context)
            return {"status": "success", "result": result}
        except Exception as e:
            print(f"[PluginManager] Execution failed: {e}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
