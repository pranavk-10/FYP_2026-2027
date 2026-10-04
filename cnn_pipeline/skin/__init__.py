"""
Skin Lesion (Dermoscopy) Specialist Pipeline Module
"""
from .skin_inference import predict_skin, load_skin_model

__all__ = ["predict_skin", "load_skin_model"]
