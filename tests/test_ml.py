import torch
import pytest
from app.ml.model import model_manager, EuroSatCNN
from app.ml.inference import inference_engine

def test_model_loading():
    model_manager.load_model()
    assert model_manager.is_loaded is True
    assert model_manager.model is not None
    assert isinstance(model_manager.model, EuroSatCNN)
    assert len(model_manager.classes) == 7
    assert len(model_manager.checksum) == 64

def test_deterministic_forward_inference():
    if not model_manager.is_loaded:
        model_manager.load_model()
        
    dummy_input = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    
    result1 = inference_engine.predict(dummy_input)
    result2 = inference_engine.predict(dummy_input)
    
    assert result1["predicted_class"] == result2["predicted_class"]
    assert result1["confidence"] == result2["confidence"]
    assert result1["probabilities"] == result2["probabilities"]
    assert result1["predicted_class"] in model_manager.classes
    assert 0.0 <= result1["confidence"] <= 1.0
    
    # Check probabilities sum to ~1.0
    prob_sum = sum(result1["probabilities"].values())
    assert pytest.approx(prob_sum, abs=1e-2) == 1.0

def test_inference_latency_measurement():
    if not model_manager.is_loaded:
        model_manager.load_model()
        
    dummy_input = torch.rand((1, 3, 64, 64), dtype=torch.float32)
    result = inference_engine.predict(dummy_input)
    
    assert "inference_latency_ms" in result
    assert result["inference_latency_ms"] >= 0.0
