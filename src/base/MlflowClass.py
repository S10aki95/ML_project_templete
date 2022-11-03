# Setting
import sys
import os
import sqlite3


# Manage experiments
import mlflow
from mlflow.tracking import MlflowClient

# hydraのファイルの操作用
from omegaconf import DictConfig, ListConfig

sys.dont_write_bytecode = True


class MlflowBase:
    def __init__(self, DB_DIR_PATH):
        # mlflow setting
        self.DB_DIR_PATH = DB_DIR_PATH
        self.DB_PATH = DB_DIR_PATH + "mlruns.db"

        try:
            # DBを構築する
            # self.build_DB()
            # mlrunsを置く場所の指定
            # print(self.DB_DIR_PATH)
            # mlflow.set_tracking_uri(f"sqlite:///{self.DB_PATH}")
            mlflow.set_tracking_uri(self.DB_DIR_PATH)
        except:
            Exception("DB error")

        # トラッキングクライアントの作成
        self.client = MlflowClient()

    def start_experiment(self, EXPERIMENT_NAME):

        # 実験の初期化
        try:
            self.experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(
                EXPERIMENT_NAME
            ).experiment_id

        self.experiment = mlflow.get_experiment(self.experiment_id)
        print("New experiment started")
        print(f"Name: {self.experiment.name}")
        print(f"Experiment_id: {self.experiment.experiment_id}")
        print(f"Artifact Location: {self.experiment.artifact_location}")

        # 実験の開始
        self.run = self.client.create_run(self.experiment_id)
        self.run_id = self.run.info.run_id

    def build_DB(self):
        # DBの作成
        if "mlruns.db" not in os.listdir(self.DB_DIR_PATH):
            os.makedirs(os.path.dirname(self.DB_PATH), exist_ok=True)
            self.conn = sqlite3.connect(self.DB_PATH)
        else:
            pass

    def terminate_experiment(self):
        self.client.set_terminated(self.run_id)
        self.run_id = None


class MlflowToDB(MlflowBase):
    def __init__(self, DB_DIR_PATH) -> None:
        super().__init__(DB_DIR_PATH)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)

    def log_artifacts(self, local_path, artifact_name):
        self.client.log_artifact(self.run_id, local_path, artifact_path=artifact_name)

    def log_text(self, text, text_file_name):
        self.client.log_text(self.run_id, text, text_file_name)

    def log_image(self, image, image_file_name):
        self.client.log_figure(self.run_id, image, image_file_name)


class Environments(MlflowToDB):
    def __init__(self, DB_DIR_PATH) -> None:
        super().__init__(DB_DIR_PATH)

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f"{parent_name}.{k}", v)
                else:
                    self.client.log_param(self.run_id, f"{parent_name}.{k}", v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f"{parent_name}.{i}", v)
