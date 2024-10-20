import pandas as pd
from airflow.decorators import dag, task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import uniform,randint
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score
from utils import download_object, read_dataframe, save_and_upload_object, upload_object
from airflow.models.taskinstance import TaskInstance
from sklearn.tree import DecisionTreeClassifier
import joblib
import pathlib

@dag(
    schedule=None,
    catchup=False,
    tags=["sklearn"],
)
def train_model_pipeline():
    """ """

    @task()
    def load_data() -> str:
        """ """
        X, y = fetch_openml(data_id=179, return_X_y=True)
        df = pd.concat([X, y], axis="columns")
        print(df.shape)
        filename = "raw_data.csv"
        save_and_upload_object(df, filename)
        # save_dataframe(df, filename)
        return filename

    @task(
        multiple_outputs=True,
    )
    def preprocess(df_filename: str) -> dict[str, str]:
        """ """
        numeric_features = ["fnlwgt"]
        categorical_features = [
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
        ]

        df = read_dataframe(df_filename)

        df = df.dropna(subset=categorical_features + numeric_features)
        df = df[df["workclass"] != "nan"]
        y = df["class"]
        X = df.drop(columns=["class"])[numeric_features+categorical_features]
        X_filename = "X.csv"
        y_filename = "y.csv"
        save_and_upload_object(X, X_filename)
        save_and_upload_object(y, y_filename)

        return {"X": X_filename, "y": y_filename}

    @task()
    def train_test_splitter(data_dict: dict) -> dict[str, str]:
        """ """
        X_filename = data_dict["X"]
        y_filename = data_dict["y"]
        X = read_dataframe(X_filename)
        y = read_dataframe(y_filename)
        lbl = LabelBinarizer()
        y = lbl.fit_transform(y["class"]).ravel()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        X_train_filename = "X_train.csv"
        X_test_filename = "X_test.csv"
        y_train_filename = "y_train.csv"
        y_test_filename = "y_test.csv"

        save_and_upload_object(X_train, X_train_filename)
        save_and_upload_object(X_test, X_test_filename)
        save_and_upload_object(pd.Series(y_train), y_train_filename)
        save_and_upload_object(pd.Series(y_test), y_test_filename)
        return {
            "X_train": X_train_filename,
            "X_test": X_test_filename,
            "y_train": y_train_filename,
            "y_test": y_test_filename,
        }

    @task
    def svm(data_dict:dict)->dict[str,str]:
        numeric_features = ["fnlwgt"]
        categorical_features = [
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
        ]
        # xcom_value = task_instance.xcom_pull(task_ids="train_test_splitter")
        # print(xcom_value)
        # data_dict = {"X_train": "X_train.csv", "y_train": "y_train.csv"}
        X_train_filename = data_dict["X_train"]
        y_train_filename = data_dict["y_train"]
        X_train = read_dataframe(X_train_filename)
        y_train = read_dataframe(y_train_filename)
        if isinstance(y_train,pd.DataFrame):
            y_train = y_train[y_train.columns[0]].to_numpy()
        elif isinstance(y_train,pd.Series):
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

        mod_cv = RandomizedSearchCV(clf, dist, n_iter=1, cv=5,verbose=3)
        mod_cv.fit(X_train, y_train)

        mod_filename = "SVC_cv.joblib"
        mod_pth = pathlib.Path().resolve() / mod_filename
        joblib.dump(mod_cv,str(mod_pth))
        upload_object(mod_filename,str(mod_pth))
        return {"model":"SVC","mod_filename":mod_filename}

    @task
    def decision_tree(data_dict:dict)->dict[str,str]:
        numeric_features = ["fnlwgt"]
        categorical_features = [
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
        ]
        X_train_filename = data_dict["X_train"]
        y_train_filename = data_dict["y_train"]

        X_train = read_dataframe(X_train_filename)
        y_train = read_dataframe(y_train_filename)

        if isinstance(y_train,pd.DataFrame):
            y_train = y_train[y_train.columns[0]].to_numpy()
        elif isinstance(y_train,pd.Series):
            y_train = y_train.to_numpy()

        categorical_transformer = OneHotEncoder(
            categories="auto", drop=None, handle_unknown="error"
        )
        mod = DecisionTreeClassifier(
            class_weight="balanced",
        )

        preprocessor = ColumnTransformer(
            transformers=[
                # ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],remainder="passthrough"
        )

        clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", mod)])

        dist = dict(classifier__min_samples_split=randint(low=2, high=15),
                    classifier__min_samples_leaf=randint(low=1,high=15),
                    )

        mod_cv = RandomizedSearchCV(clf, dist, n_iter=1, cv=5, verbose=3)
        mod_cv.fit(X_train, y_train)

        mod_filename = "decision_tree_cv.joblib"
        mod_pth = pathlib.Path().resolve() / mod_filename
        joblib.dump(mod_cv,str(mod_pth))
        upload_object(mod_filename,str(mod_pth))
        return {"model":"decision_tree","mod_filename":mod_filename}


    @task
    def results(task_instance: TaskInstance | None = None):
        xcom_vals = task_instance.xcom_pull(task_ids=["decision_tree"])
        # print(xcom_vals)
        # print(xcom_vals[0])
        # print(type(xcom_vals[0]))
        X_test = read_dataframe("X_test.csv")
        y_test = read_dataframe("y_test.csv").to_numpy().ravel()
        for mod_dict in xcom_vals:
            mod_filename = mod_dict["mod_filename"]
            mod_name = mod_dict["model"] 
            pth = pathlib.Path() / mod_filename 
            download_object(mod_filename,str(pth))
            mod = joblib.load(str(pth))

            y_pred = mod.predict(X_test)

            test_f1_score = f1_score(y_true=y_test,y_pred=y_pred)
            print(f"this is test score for {mod_name=} {test_f1_score=}")

        



        

    raw_filename = load_data()
    data_dict = preprocess(raw_filename)
    split_data = train_test_splitter(data_dict)
    split_data >> ([decision_tree(split_data)]) >> results()
    # load_data() >> preprocess >> train_test_splitter >> svm


train_model_pipeline()

# @task()
# def load(metadata: dict):
#     """ """
