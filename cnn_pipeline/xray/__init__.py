# cnn_pipeline/xray - Chest X-Ray DenseNet121 Specialist (Ojas's Model)
# Framework: TensorFlow/Keras
# Dataset: NIH ChestX-ray14 (28,461 samples, 20 disease labels)
# Architecture: DenseNet121 → GlobalAvgPool → Dense(256) → BN → Dropout → Dense(20, sigmoid)

from .xray_inference import predict_xray
