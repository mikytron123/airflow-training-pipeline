import pandas as pd
from airflow.decorators import dag, task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import uniform, randint
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)
from minio_client import MinioClient
from airflow.models.taskinstance import TaskInstance
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin
import joblib
import pathlib
import mlflow
import numpy as np
import os
import json

ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")
MINIO_HOST = os.getenv("MINIO_HOST", default="localhost")
MINIO_PORT = os.getenv("MINIO_PORT")


def score_models(
    model: ClassifierMixin, X: pd.DataFrame, y: np.ndarray
) -> tuple[float, float, float, float]:
    y_pred = model.predict(X)
    mod_f1_score = f1_score(y_true=y, y_pred=y_pred, average="micro")
    mod_precision_score = precision_score(y_true=y, y_pred=y_pred, average="micro")
    mod_recall_score = recall_score(y_true=y, y_pred=y_pred, average="micro")
    mod_accuracy_score = balanced_accuracy_score(
        y_true=y,
        y_pred=y_pred,
    )
    return mod_f1_score, mod_precision_score, mod_recall_score, mod_accuracy_score


minio_client = MinioClient(MINIO_HOST, MINIO_PORT, ACCESS_KEY, SECRET_KEY)


@dag(
    schedule=None,
    catchup=False,
    tags=["sklearn"],
)
def train_model_pipeline():
    """ """

    @task()
    def load_data() -> dict[str, str]:
        """ """
        X, y = fetch_openml(data_id=179, return_X_y=True)
        df = pd.concat([X, y], axis="columns")
        print(df.shape)
        filename = "raw_data.csv"
        minio_client.save_and_upload_dataframe(df, filename)
        data_type_dict = dict(
            numeric_features=["fnlwgt"],
            categorical_features=[
                "age",
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "capitalgain",
                "capitalloss",
                "hoursperweek",
            ],
        )
        data_type_filename = "data_type.json"
        pth = pathlib.Path().resolve() / data_type_filename

        with open(data_type_filename, "w") as file:
            json.dump(data_type_dict, file, indent=4)

        minio_client.upload_object("data_type.json", str(pth), "application/json")
        return {"data": filename, "data_type": data_type_filename}

    @task(
        multiple_outputs=True,
    )
    def preprocess(df_dict: dict) -> dict[str, str]:
        """ """

        df = minio_client.read_dataframe(df_dict["data"])

        data_type_dict = minio_client.read_object(df_dict["data_type"], "json")
        categorical_features = data_type_dict["categorical_features"]
        numeric_features = data_type_dict["numeric_features"]

        df = df.dropna(subset=categorical_features + numeric_features)
        df = df[df["workclass"] != "nan"]
        y = df["class"]
        X = df.drop(columns=["class"])[numeric_features + categorical_features]

        X_filename = "X.csv"
        y_filename = "y.csv"
        minio_client.save_and_upload_dataframe(X, X_filename)
        minio_client.save_and_upload_dataframe(y, y_filename)

        return {"X": X_filename, "y": y_filename, "data_type": df_dict["data_type"]}

    @task()
    def train_test_splitter(data_dict: dict) -> dict[str, str]:
        """ """
        X_filename = data_dict["X"]
        y_filename = data_dict["y"]
        X = minio_client.read_dataframe(X_filename)
        y = minio_client.read_dataframe(y_filename)
        lbl = LabelBinarizer()
        y = lbl.fit_transform(y["class"]).ravel()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        X_train_filename = "X_train.csv"
        X_test_filename = "X_test.csv"
        y_train_filename = "y_train.csv"
        y_test_filename = "y_test.csv"

        minio_client.save_and_upload_dataframe(X_train, X_train_filename)
        minio_client.save_and_upload_dataframe(X_test, X_test_filename)
        minio_client.save_and_upload_dataframe(pd.Series(y_train), y_train_filename)
        minio_client.save_and_upload_dataframe(pd.Series(y_test), y_test_filename)
        return {
            "X_train": X_train_filename,
            "X_test": X_test_filename,
            "y_train": y_train_filename,
            "y_test": y_test_filename,
            "data_type": data_dict["data_type"],
        }

    @task
    def svm(data_dict: dict) -> dict[str, str]:
        data_type_dict = minio_client.read_object(data_dict["data_type"], "json")
        categorical_features = data_type_dict["categorical_features"]
        numeric_features = data_type_dict["numeric_features"]

        X_train_filename = data_dict["X_train"]
        y_train_filename = data_dict["y_train"]
        X_train = minio_client.read_dataframe(X_train_filename)
        y_train = minio_client.read_dataframe(y_train_filename)

        if isinstance(y_train, pd.DataFrame):
            y_train = y_train[y_train.columns[0]].to_numpy()
        elif isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(
            categories="auto", drop=None, handle_unknown="error"
        )
        mod = SVC(
            C=0.8,
            class_weight="balanced",
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", mod)])

        dist = dict(classifier__C=uniform(loc=0.1, scale=1))

        mod_cv = RandomizedSearchCV(clf, dist, n_iter=1, cv=5, verbose=3)
        mod_cv.fit(X_train, y_train)

        mod_filename = "SVC_cv.joblib"
        mod_pth = pathlib.Path().resolve() / mod_filename
        joblib.dump(mod_cv, str(mod_pth))
        minio_client.upload_object(mod_filename, str(mod_pth))
        return {"model": "SVC", "mod_filename": mod_filename}

    @task
    def decision_tree(data_dict: dict) -> dict[str, str]:
        mlflow.set_tracking_uri(uri="http://mlflow:5000")

        mlflow.set_experiment(experiment_name="Training")
        mlflow.sklearn.autolog()

        data_type_dict = minio_client.read_object(data_dict["data_type"], "json")
        categorical_features = data_type_dict["categorical_features"]

        X_train_filename = data_dict["X_train"]
        y_train_filename = data_dict["y_train"]
        X_test_filename = data_dict["X_test"]
        y_test_filename = data_dict["y_test"]

        X_train = minio_client.read_dataframe(X_train_filename)
        y_train = minio_client.read_dataframe(y_train_filename)
        X_test = minio_client.read_dataframe(X_test_filename)
        y_test = minio_client.read_dataframe(y_test_filename)

        if isinstance(y_train, pd.DataFrame):
            y_train = y_train[y_train.columns[0]].to_numpy()
        elif isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()

        if isinstance(y_test, pd.DataFrame):
            y_test = y_test[y_test.columns[0]].to_numpy()
        elif isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()

        categorical_transformer = OneHotEncoder(
            categories="auto", drop=None, handle_unknown="error"
        )
        mod = DecisionTreeClassifier(
            class_weight="balanced",
        )
        with mlflow.start_run(run_name="decision tree"):
            preprocessor = ColumnTransformer(
                transformers=[
                    # ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ],
                remainder="passthrough",
                force_int_remainder_cols=False,
            )

            clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", mod)])

            dist = dict(
                classifier__min_samples_split=randint(low=2, high=15),
                classifier__min_samples_leaf=randint(low=1, high=15),
            )
            mlflow.log_params(
                {
                    "classifier__min_samples_split": "randint(low=2, high=15)",
                    "classifier__min_samples_leaf": "randint(low=1,high=15)",
                }
            )

            mod_cv = RandomizedSearchCV(clf, dist, n_iter=1, cv=5, verbose=3)
            mod_cv.fit(X_train, y_train)

            best_mod = mod_cv.best_estimator_

            mod_f1_score, mod_precision_score, mod_recall_score, mod_accuracy_score = (
                score_models(best_mod, X_test, y_test)
            )
            mlflow.log_metric(key="test_f1_score", value=mod_f1_score)
            mlflow.log_metric(key="test_precision_score", value=mod_precision_score)
            mlflow.log_metric(key="test_recall_score", value=mod_recall_score)
            mlflow.log_metric(
                key="test_balanced_accuracy_score", value=mod_accuracy_score
            )

        mod_filename = "decision_tree_cv.joblib"
        mod_pth = pathlib.Path().resolve() / mod_filename
        joblib.dump(mod_cv.best_estimator_, str(mod_pth))
        minio_client.upload_object(mod_filename, str(mod_pth))
        return {"model": "decision_tree", "mod_filename": mod_filename}

    @task
    def results(task_instance: TaskInstance | None = None):
        xcom_vals = task_instance.xcom_pull(task_ids=["decision_tree"])
        X_test = minio_client.read_dataframe("X_test.csv")
        y_test = minio_client.read_dataframe("y_test.csv").to_numpy().ravel()
        for mod_dict in xcom_vals:
            mod_filename = mod_dict["mod_filename"]
            mod_name = mod_dict["model"]
            pth = pathlib.Path() / mod_filename
            minio_client.download_object(mod_filename, str(pth))
            mod = joblib.load(str(pth))

            y_pred = mod.predict(X_test)

            test_f1_score = f1_score(y_true=y_test, y_pred=y_pred)
            print(f"this is test score for {mod_name=} {test_f1_score=}")

    raw_filename = load_data()
    data_dict = preprocess(raw_filename)
    split_data = train_test_splitter(data_dict)
    split_data >> ([decision_tree(split_data)]) >> results()
    # load_data() >> preprocess >> train_test_splitter >> svm


train_model_pipeline()
