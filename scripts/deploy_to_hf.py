"""
Hugging Face Spaces Automated Deployment Script for AquaGuard RT-DETR
====================================================================
Uploads model weights, Gradio UI, reasoning logic, sample images, and space configuration
directly to a Hugging Face Space repository.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]

def deploy(space_name: str, hf_token: str = None):
    print("=" * 60)
    print(f" Deploying AquaGuard SeaClear RT-DETR to Hugging Face Spaces ")
    print("=" * 60)
    
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        print("\n[ERROR] No Hugging Face token provided.")
        print("Please provide your Hugging Face write token.")
        print("Get your token at: https://huggingface.co/settings/tokens")
        return False
        
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{space_name}"
    
    print(f"\nAuthenticated as: {username}")
    print(f"Target Space Repo: https://huggingface.co/spaces/{repo_id}")
    
    # 1. Create or verify Space repo
    print("\n[1/3] Checking Hugging Face Space repository...")
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print("  Space repository already exists.")
    except Exception:
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk="gradio",
                private=False,
                exist_ok=True,
                token=token
            )
            print("  Space repository created.")
        except Exception as e:
            print(f"  [NOTE] Proceeding with file upload to existing space: {e}")
    
    # 2. Upload main application files & test samples
    files_to_upload = [
        ("app_ui.py", "app.py"),                     # Main Gradio entrypoint
        ("app_ui.py", "app_ui.py"),                  # UI module
        ("app/reasoning.py", "reasoning.py"),        # Framework-free reasoning engine
        ("app/reasoning.py", "app/reasoning.py"),    # Package path fallback
        ("requirements.txt", "requirements.txt"),    # Dependencies
        ("weights/best.pt", "weights/best.pt"),      # Fine-tuned RT-DETR-L checkpoint
        ("data/yolo/images/test/110.jpg", "data/yolo/images/test/110.jpg"),
        ("data/yolo/images/test/1876.jpg", "data/yolo/images/test/1876.jpg"),
        ("data/yolo/images/test/431.jpg", "data/yolo/images/test/431.jpg"),
        ("data/yolo/images/test/1946.jpg", "data/yolo/images/test/1946.jpg"),
        ("data/yolo/images/test/1360.jpg", "data/yolo/images/test/1360.jpg"),
        ("data/yolo/images/test/Cam1_16_26_03_10_11_2020.mp4_00248.jpg", "data/yolo/images/test/Cam1_16_26_03_10_11_2020.mp4_00248.jpg"),
    ]
    
    print("\n[2/3] Uploading Space application assets, test samples, and model weights...")
    for local_rel, space_rel in files_to_upload:
        local_path = ROOT / local_rel
        if not local_path.exists():
            print(f"  [WARNING] Local file not found: {local_path}, skipping.")
            continue
        print(f"  Uploading {local_rel} -> {space_rel}...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=space_rel,
            repo_id=repo_id,
            repo_type="space",
            token=token
        )
        
    # 3. Create Space README with Gradio metadata
    print("\n[3/3] Uploading Space configuration metadata...")
    space_readme = f"""---
title: AquaGuard SeaClear RT-DETR
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
pinned: false
license: mit
short_description: Underwater marine debris detection & reasoning with RT-DETR
---

# 🌊 AquaGuard: SeaClear RT-DETR Underwater Debris Detection & Reasoning

Fine-tuned **RT-DETR-L** model (92.32% mAP@50) and framework-free Python reasoning engine for benthic marine cleanup.
"""
    api.upload_file(
        path_or_fileobj=space_readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        token=token
    )
    
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Space Deployed Successfully!")
    print(f"Live URL: https://huggingface.co/spaces/{repo_id}")
    print("=" * 60)
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        space_name = sys.argv[1]
        token = sys.argv[2] if len(sys.argv) > 2 else None
        deploy(space_name, token)
    else:
        print("Usage: python scripts/deploy_to_hf.py <SPACE_NAME> [HF_TOKEN]")
        print("Example: python scripts/deploy_to_hf.py aquaguard-rtdetr hf_xxxxxxxxxxxxxxxxxxxxxxxx")
