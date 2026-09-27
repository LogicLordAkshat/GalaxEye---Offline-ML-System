import time
import torch
from typing import Dict, Tuple, Any
from app.ml.model import model_manager
from app.core.logging import logger

class InferenceEngine:
    """
    Offline inference engine executing deterministic PyTorch forward pass on CPU.
    Extracts top prediction, runner-up class, and confidence margin.
    """
    def __init__(self):
        pass

    def predict(self, input_tensor: torch.Tensor) -> dict[str, Any]:
        """
        Executes forward pass for a single preprocessed image tensor (1, 3, 64, 64).
        Returns top prediction, runner-up prediction, margin, distribution, and latency.
        """
        if not model_manager.is_loaded or model_manager.model is None:
            raise RuntimeError("Model is not loaded. Call model_manager.load_model() during startup.")

        t_start = time.perf_counter()
        
        with torch.no_grad():
            logits = model_manager.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
            
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            
            top_idx = sorted_indices[0].item()
            top_prob = float(sorted_probs[0].item())
            predicted_class = model_manager.idx_to_class[top_idx]
            
            # Top-2 runner-up extraction
            if len(sorted_probs) > 1:
                top2_idx = sorted_indices[1].item()
                top2_prob = float(sorted_probs[1].item())
                top_2_class = model_manager.idx_to_class[top2_idx]
                confidence_margin = round(top_prob - top2_prob, 4)
            else:
                top_2_class = predicted_class
                top2_prob = 0.0
                confidence_margin = round(top_prob, 4)
            
            # Full distribution
            prob_dict: dict[str, float] = {
                model_manager.idx_to_class[i]: round(float(probabilities[i].item()), 4)
                for i in range(len(model_manager.classes))
            }

        inference_latency_ms = (time.perf_counter() - t_start) * 1000.0
        
        return {
            "predicted_class": predicted_class,
            "confidence": round(top_prob, 4),
            "top_2_class": top_2_class,
            "top_2_confidence": round(top2_prob, 4),
            "confidence_margin": confidence_margin,
            "probabilities": prob_dict,
            "inference_latency_ms": round(inference_latency_ms, 2)
        }

inference_engine = InferenceEngine()
