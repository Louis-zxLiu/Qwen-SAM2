import subprocess
import os
import sys

def check_dependencies():
    print("[INFO] Checking Environment Dependencies...")
    try:
        import transformers
        import tokenizers
        import huggingface_hub
        from packaging import version
        
        # Check transformers
        v_trans = version.parse(transformers.__version__)
        if v_trans < version.parse("4.45.0"):
            print(f"[ERROR] Transformers version ({transformers.__version__}) is too low. Need >= 4.45.0")
            return False
            
        # Check tokenizers compatibility
        # Transformers 4.49.0 (and similar) strictly requires tokenizers>=0.22.0,<=0.23.0
        v_tok = version.parse(tokenizers.__version__)
        if v_tok > version.parse("0.23.0") or v_tok < version.parse("0.22.0"):
            print(f"[ERROR] Tokenizers version ({tokenizers.__version__}) is incompatible.")
            print("[ERROR] Transformers requires tokenizers>=0.22.0,<=0.23.0")
            print("[ACTION] Please run: pip install 'tokenizers>=0.22.0,<=0.23.0' --force-reinstall")
            return False
            
        # Check huggingface_hub compatibility
        try:
            from huggingface_hub import is_offline_mode
        except ImportError:
            print(f"[ERROR] huggingface_hub version ({huggingface_hub.__version__}) is too old.")
            print("[ACTION] Please run: pip install --upgrade huggingface_hub>=0.23.0")
            return False
            
        print(f"[INFO] Environment Check Passed (Transformers: {transformers.__version__}, Tokenizers: {tokenizers.__version__}, HF_Hub: {huggingface_hub.__version__})")
    except ImportError as e:
        print(f"[WARNING] Dependency missing: {e}")
        print("[ACTION] Please run: pip install -r backend/requirements.txt")
    
    return True

def main():
    print("========================================")
    print("One-Click Launch Script")
    print("========================================")
    
    # Check dependencies before starting
    if not check_dependencies():
        choice = input("Dependencies might be missing or outdated. Continue anyway? (y/n) [default: n]: ").strip().lower()
        if choice != 'y':
            sys.exit(1)
    
    current_dir = os.getcwd()

    # 0. Set Environment Variables
    print("[INFO] Configuring Environment...")
    # HF Cache Directory
    hf_cache_dir = os.path.join(current_dir, "hf_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir
    print(f"[ENV] HF_HOME set to: {hf_cache_dir}")
    
    # HF Mirror Selection
    print("Choose HuggingFace Download Source / 选择下载源:")
    print("1. HF-Mirror (Recommended for CN users / 国内推荐)")
    print("2. Official HuggingFace (Global / 官方源)")
    choice = input("Enter choice (1/2) [default: 1]: ").strip()

    hf_endpoint_cmd = ""
    if choice == '2':
        print("[ENV] Using Official HuggingFace Endpoint")
    else:
        print("[ENV] Using HF Mirror: https://hf-mirror.com")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        hf_endpoint_cmd = "set HF_ENDPOINT=https://hf-mirror.com&& "

    # 1. Start Backend
    print("[INFO] Starting Backend Server...")
    backend_dir = os.path.join(current_dir, "backend")
    
    # Check for venv python
    venv_python = os.path.join(backend_dir, "venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        python_exec = venv_python
    else:
        python_exec = "python"
        print("[WARNING] Virtual environment not found, using system python.")

    # Command to run uvicorn
    # Pass environment variables to the new process
    # Note: 'start' command in Windows doesn't easily inherit current process envs if not explicitly set in the command line or if the shell is new.
    # However, setting os.environ here affects the Popen call. But since we use 'start cmd /k', it opens a NEW console.
    # We need to set the env vars inside the new console command.
    
    backend_cmd_str = f'set HF_HOME={hf_cache_dir}&& {hf_endpoint_cmd}{python_exec} -m uvicorn main:app --reload --host 0.0.0.0 --port 8000'
    backend_cmd = f'start "Backend Server" cmd /k "{backend_cmd_str}"'
    subprocess.Popen(backend_cmd, shell=True, cwd=backend_dir)

    # 2. Start Frontend
    print("[INFO] Starting Frontend Server...")
    frontend_dir = os.path.join(current_dir, "frontend")
    # Command to run npm dev
    frontend_cmd = 'start "Frontend Server" cmd /k "npm run dev"'
    subprocess.Popen(frontend_cmd, shell=True, cwd=frontend_dir)

    print("\n[SUCCESS] Servers are launching in separate windows.")
    print("Frontend: http://localhost:5173")
    print("Backend: http://localhost:8000")

if __name__ == "__main__":
    main()
