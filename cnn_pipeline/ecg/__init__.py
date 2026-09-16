"""
ECG Specialist Pipeline Module
"""
from .ecg_inference import predict_ecg, load_ecg_model

__all__ = ["predict_ecg", "load_ecg_model"]
