from minio import Minio
import pandas as pd
import os
from io import BytesIO
import pathlib

ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")
BUCKET = os.getenv("MINIO_BUCKET")
MINIO_HOST = os.getenv("MINIO_HOST", default="localhost")
MINIO_PORT = os.getenv("MINIO_PORT")

client = Minio(
    endpoint=f"{MINIO_HOST}:{MINIO_PORT}",
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False,
)


def save_dataframe(
    df: pd.DataFrame,
    filename: str,
):
    buff = BytesIO()
    df.to_csv(buff)

    client.put_object(
        bucket_name=BUCKET,
        object_name=filename,
        data=buff,
        length=len(buff.getvalue()),
    )


def upload_object(object_name: str, file_path: str):
    client.fput_object(bucket_name=BUCKET, object_name=object_name, file_path=file_path)

def download_object(object_name:str,file_path:str):
    client.fget_object(bucket_name=BUCKET,object_name=object_name,file_path=file_path)

def read_dataframe(object_name: str) -> pd.DataFrame:
    try:
        stat = client.stat_object(bucket_name=BUCKET, object_name=object_name)
        response = client.get_object(
            bucket_name=BUCKET, object_name=object_name, length=stat.size
        )
        df = pd.read_csv(BytesIO(response.data))
        return df
    finally:
        response.close()
        response.release_conn()


def save_and_upload_object(df: pd.DataFrame, filename: str):
    pth = pathlib.Path().resolve() / filename
    df.to_csv(str(pth), index=False)
    upload_object(filename, str(pth))
