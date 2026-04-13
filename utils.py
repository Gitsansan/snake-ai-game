import torch
import os
from config import MODEL_DIR

def save_model(model, episode):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = f"{MODEL_DIR}/model_{episode}.pth"
    torch.save(model.state_dict(), path)
    print(f"Saved: {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Loaded: {path}")