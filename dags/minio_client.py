from typing import Literal
from minio import Minio
import pandas as pd
import os
from io import BytesIO
import pathlib


class MinioClient:
    def __init__(
        self, minio_host: str, minio_port: str, access_key: str, secret_key: str
    ) -> None:
        self.client = Minio(
            endpoint=f"{minio_host}:{minio_port}",
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )
        self.bucket = os.getenv("MINIO_BUCKET")

    def save_dataframe(self, df: pd.DataFrame, filename: str):
        buff = BytesIO()
        df.to_csv(buff)

        self.client.put_object(
            bucket_name=self.bucket,
            object_name=filename,
            data=buff,
            length=len(buff.getvalue()),
        )

    def upload_object(
        self,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ):
        self.client.fput_object(
            bucket_name=self.bucket,
            object_name=object_name,
            file_path=file_path,
            content_type=content_type,
        )

    def download_object(self, object_name: str, file_path: str):
        self.client.fget_object(
            bucket_name=self.bucket, object_name=object_name, file_path=file_path
        )

    def read_object(self, object_name: str, object_type: Literal["dataframe", "json"]):
        try:
            stat = self.client.stat_object(
                bucket_name=self.bucket, object_name=object_name
            )
            response = self.client.get_object(
                bucket_name=self.bucket, object_name=object_name, length=stat.size
            )
            if object_type == "dataframe":
                return response.data
            elif object_type == "json":
                return response.json()
        finally:
            response.close()
            response.release_conn()

    def read_dataframe(self, object_name: str) -> pd.DataFrame:
        data = self.read_object(object_name=object_name, object_type="dataframe")
        df = pd.read_csv(BytesIO(data))
        return df

    def save_and_upload_dataframe(self, df: pd.DataFrame | pd.Series, filename: str):
        pth = pathlib.Path().resolve() / filename
        df.to_csv(str(pth), index=False)
        self.upload_object(filename, str(pth), "text/csv")
