import aioboto3
from io import BytesIO


class MinioAsyncStorage:
    def __init__(self, endpoint_url: str, access_key: str, secret_key: str, bucket: str):
        self.endpoint_url = endpoint_url
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.session = aioboto3.Session()

    async def download_file(self, key: str) -> BytesIO:
        async with self.session.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name="us-east-1"
        ) as s3:
            buffer = BytesIO()
            await s3.download_fileobj(self.bucket, key, buffer)
            buffer.seek(0)
            return buffer