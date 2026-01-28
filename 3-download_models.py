#!/usr/bin/env python3
"""
Indian Equity Intelligence - Model Download Script

This script helps you download the required ML models based on your PC specifications.
Run with: python download_models.py
"""

import os
import sys
import subprocess
from pathlib import Path


def get_system_info():
    """Get system memory and GPU info."""
    info = {"ram_gb": 8, "gpu": None, "gpu_vram_gb": 0}
    
    try:
        import psutil
        info["ram_gb"] = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = "nvidia"
            info["gpu_vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info["gpu"] = "apple_silicon"
            info["gpu_vram_gb"] = info["ram_gb"]  # Shared memory on Apple Silicon
    except ImportError:
        pass
    
    return info


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 60)
    print("  Indian Equity Intelligence - Model Downloader")
    print("=" * 60)


def print_system_info(info):
    """Print detected system info."""
    print(f"\n📊 Detected System:")
    print(f"   RAM: {info['ram_gb']:.1f} GB")
    if info['gpu']:
        print(f"   GPU: {info['gpu'].replace('_', ' ').title()}")
        if info['gpu'] == 'nvidia':
            print(f"   VRAM: {info['gpu_vram_gb']:.1f} GB")
    else:
        print("   GPU: Not detected (will use CPU)")


def get_user_choice():
    """Get user's choice for model configuration."""
    print("\n" + "-" * 60)
    print("Choose your setup based on your PC specs:\n")
    print("  [1] 🚀 FULL     - All models (16GB+ RAM, GPU recommended)")
    print("                   Chronos-T5-Base + LightGBM + Qwen2.5-3B")
    print()
    print("  [2] 💻 STANDARD - Balanced setup (8-16GB RAM)")
    print("                   Chronos-T5-Small + LightGBM + Qwen2.5-3B")
    print()
    print("  [3] 🪶 LITE     - Minimal setup (4-8GB RAM)")
    print("                   Chronos-T5-Tiny + LightGBM (no LLM)")
    print()
    print("  [4] 🎯 CUSTOM   - Choose individual models")
    print()
    print("  [0] ❌ EXIT     - Cancel and exit")
    print("-" * 60)
    
    while True:
        choice = input("\nEnter your choice [1-4, 0 to exit]: ").strip()
        if choice in ['0', '1', '2', '3', '4']:
            return choice
        print("Invalid choice. Please enter 1, 2, 3, 4, or 0.")


def get_custom_choices():
    """Get custom model choices from user."""
    choices = {"forecaster": None, "classifier": "lightgbm", "explainer": None}
    
    print("\n--- Layer 1: Time-Series Forecaster ---")
    print("  [1] Chronos-T5-Base   (500MB, best accuracy)")
    print("  [2] Chronos-T5-Small  (200MB, balanced)")
    print("  [3] Chronos-T5-Tiny   (100MB, fastest)")
    print("  [4] Lag-Llama         (1GB, probabilistic)")
    print("  [0] Skip (no forecasting)")
    
    fc = input("Choose forecaster [1-4, 0 to skip]: ").strip()
    forecaster_map = {'1': 'chronos-t5-base', '2': 'chronos-t5-small', 
                      '3': 'chronos-t5-tiny', '4': 'lag-llama', '0': None}
    choices["forecaster"] = forecaster_map.get(fc, 'chronos-t5-base')
    
    print("\n--- Layer 2: Classifier ---")
    print("  [1] LightGBM  (fast, recommended)")
    print("  [2] CatBoost  (better with categories)")
    print("  Classifier is trained locally - no download needed!")
    
    cl = input("Choose classifier [1-2]: ").strip()
    choices["classifier"] = "catboost" if cl == '2' else "lightgbm"
    
    print("\n--- Layer 3: LLM Explainer ---")
    print("  [1] Qwen2.5-3B   (2GB, recommended)")
    print("  [2] Qwen2.5-7B   (4.5GB, better quality)")
    print("  [0] Skip (use rule-based explanations)")
    
    ex = input("Choose explainer [1-2, 0 to skip]: ").strip()
    explainer_map = {'1': 'qwen2.5-3b', '2': 'qwen2.5-7b', '0': None}
    choices["explainer"] = explainer_map.get(ex, 'qwen2.5-3b')
    
    return choices


def get_preset_config(choice, system_info):
    """Get preset configuration based on choice."""
    if choice == '1':  # Full
        return {
            "forecaster": "chronos-t5-base",
            "classifier": "lightgbm",
            "explainer": "qwen2.5-3b"
        }
    elif choice == '2':  # Standard
        return {
            "forecaster": "chronos-t5-small",
            "classifier": "lightgbm", 
            "explainer": "qwen2.5-3b"
        }
    elif choice == '3':  # Lite
        return {
            "forecaster": "chronos-t5-tiny",
            "classifier": "lightgbm",
            "explainer": None
        }
    return {}


def create_directories():
    """Create model directories."""
    dirs = [
        "models/forecaster/chronos-t5-base",
        "models/forecaster/chronos-t5-small", 
        "models/forecaster/chronos-t5-tiny",
        "models/forecaster/lag-llama",
        "models/classifier/lightgbm",
        "models/classifier/catboost",
        "models/explainer/qwen2.5-3b",
        "models/explainer/qwen2.5-7b",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("\n✓ Created model directories")


def download_forecaster(model_name):
    """Download forecaster model."""
    if not model_name:
        return True
    
    model_map = {
        "chronos-t5-base": "amazon/chronos-t5-base",
        "chronos-t5-small": "amazon/chronos-t5-small",
        "chronos-t5-tiny": "amazon/chronos-t5-tiny",
        "lag-llama": "time-series-foundation-models/Lag-Llama",
    }
    
    repo = model_map.get(model_name)
    if not repo:
        return False
    
    local_dir = f"models/forecaster/{model_name}"
    
    print(f"\n📥 Downloading {model_name}...")
    print(f"   From: huggingface.co/{repo}")
    print(f"   To: {local_dir}")
    print("   This may take a few minutes...")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Download with explicit parameters
        result = snapshot_download(
            repo_id=repo,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        # Verify files were downloaded
        dir_path = Path(local_dir)
        files = list(dir_path.glob("*"))
        if files:
            print(f"✓ Downloaded {model_name} ({len(files)} files)")
            return True
        else:
            print(f"⚠ Download completed but no files found in {local_dir}")
            return False
            
    except ImportError:
        print(f"✗ huggingface_hub not installed")
        print(f"   Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        print(f"\n  Manual download command:")
        print(f"  huggingface-cli download {repo} --local-dir {local_dir}")
        return False


def download_explainer(model_name):
    """Download LLM explainer model."""
    if not model_name:
        return True
    
    model_map = {
        "qwen2.5-3b": {
            "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "file": "qwen2.5-3b-instruct-q4_k_m.gguf",
        },
        "qwen2.5-7b": {
            "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "file": "qwen2.5-7b-instruct-q4_k_m.gguf",
        },
    }
    
    info = model_map.get(model_name)
    if not info:
        return False
    
    local_dir = f"models/explainer/{model_name}"
    local_file = f"{local_dir}/model.gguf"
    
    # Create directory
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading {model_name}...")
    print(f"   From: huggingface.co/{info['repo']}")
    print(f"   To: {local_file}")
    print("   This is a large file (~2GB), please wait...")
    
    try:
        from huggingface_hub import hf_hub_download
        
        downloaded_path = hf_hub_download(
            repo_id=info['repo'],
            filename=info['file'],
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print(f"   Downloaded to: {downloaded_path}")
        
        # Rename to model.gguf for consistency
        downloaded = Path(local_dir) / info['file']
        target = Path(local_file)
        
        if downloaded.exists():
            if not target.exists():
                downloaded.rename(target)
                print(f"✓ Downloaded and renamed to {target}")
            else:
                print(f"✓ Downloaded {model_name}")
            return True
        elif target.exists():
            print(f"✓ Model already exists at {target}")
            return True
        else:
            # Check if file is in cache subdirectory
            for f in Path(local_dir).rglob("*.gguf"):
                if not target.exists():
                    import shutil
                    shutil.copy2(f, target)
                    print(f"✓ Copied from cache to {target}")
                    return True
            
            print(f"⚠ Download completed but file not found at expected location")
            return False
            
    except ImportError:
        print(f"✗ huggingface_hub not installed")
        print(f"   Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        print(f"\n  Manual download command:")
        print(f"  mkdir -p {local_dir}")
        print(f"  wget https://huggingface.co/{info['repo']}/resolve/main/{info['file']} -O {local_file}")
        return False


def update_config(config):
    """Update settings.yaml with chosen models."""
    config_path = Path("config/settings.yaml")
    
    if not config_path.exists():
        print("⚠ config/settings.yaml not found, skipping config update")
        return
    
    try:
        import yaml
        
        with open(config_path, 'r') as f:
            settings = yaml.safe_load(f)
        
        # Update ML config
        if 'ml_config' not in settings:
            settings['ml_config'] = {}
        
        ml = settings['ml_config']
        ml['enabled'] = True
        ml['auto_initialize'] = True  # Auto-init on startup
        
        if 'forecaster' not in ml:
            ml['forecaster'] = {}
        ml['forecaster']['model'] = config.get('forecaster')
        
        if 'classifier' not in ml:
            ml['classifier'] = {}
        ml['classifier']['model'] = config.get('classifier')
        
        if 'explainer' not in ml:
            ml['explainer'] = {}
        ml['explainer']['model'] = config.get('explainer')
        
        with open(config_path, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False, sort_keys=False)
        
        print("\n✓ Updated config/settings.yaml")
        
    except Exception as e:
        print(f"⚠ Could not update config: {e}")


def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Checking dependencies...")
    
    packages = ["huggingface_hub", "pyyaml"]
    
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"   Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])
    
    print("✓ Dependencies ready")


def print_summary(config):
    """Print download summary."""
    print("\n" + "=" * 60)
    print("  SETUP COMPLETE")
    print("=" * 60)
    print("\n📋 Configuration:")
    print(f"   Forecaster: {config.get('forecaster') or 'Disabled'}")
    print(f"   Classifier: {config.get('classifier') or 'Disabled'}")
    print(f"   Explainer:  {config.get('explainer') or 'Rule-based (no LLM)'}")
    
    print("\n🚀 Next steps:")
    print("   1. Install requirements: pip install -r requirements.txt")
    print("   2. Run the app:          streamlit run app.py")
    print("   3. ML will auto-initialize when you start the app!")
    print()


def main():
    print_header()
    
    # Check if running from correct directory
    if not Path("app.py").exists():
        print("\n❌ Error: Please run this script from the project root directory")
        print("   cd /path/to/stockIt_by_RTR && python download_models.py")
        sys.exit(1)
    
    install_dependencies()
    
    system_info = get_system_info()
    print_system_info(system_info)
    
    choice = get_user_choice()
    
    if choice == '0':
        print("\n👋 Cancelled. No models downloaded.")
        sys.exit(0)
    
    if choice == '4':
        config = get_custom_choices()
    else:
        config = get_preset_config(choice, system_info)
    
    # Create directories
    create_directories()
    
    # Download models
    success = True
    
    if config.get('forecaster'):
        if not download_forecaster(config['forecaster']):
            success = False
    
    if config.get('explainer'):
        if not download_explainer(config['explainer']):
            success = False
    
    # Classifier is trained locally, no download needed
    print(f"\n✓ Classifier ({config.get('classifier', 'lightgbm')}) will be trained locally")
    
    # Update config
    update_config(config)
    
    # Print summary
    print_summary(config)
    
    if not success:
        print("⚠ Some downloads failed. See manual commands above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
