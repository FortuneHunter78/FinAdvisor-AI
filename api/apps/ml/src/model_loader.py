import pickle
from io import BytesIO
import onnxruntime as ort
from src.s3.s3 import MinioAsyncStorage


class ModelLoader:
    def __init__(self, storage: MinioAsyncStorage, model_key: str):
        self.storage = storage
        self.model_key = model_key
        self.model = None

    async def load_model(self):
        buffer: BytesIO = await self.storage.download_file(self.model_key)
        buffer.seek(0)
        self.session = ort.InferenceSession(buffer.read())

    def get_model(self):
        if self.model is None:
            raise RuntimeError("Model not loaded yet")
        return self.model
