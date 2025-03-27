import os
import shutil
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple

import bentoml
import mlflow
import numpy.typing as npt
import pandas as pd
from catboost import CatBoostClassifier, Pool
from dotenv import load_dotenv
from mlflow.entities import Run
from mlflow.models import infer_signature
from tqdm.auto import tqdm

from utils.dates import DateValues

# .env 파일 로드
load_dotenv()

artifacts_path = os.getenv("ARTIFACTS_PATH")

TARGET_NAME = "credit_score"


class Trainer:
    """모델 학습 클래스

    Args:
        model_name (str): 모델명
            해당 이름으로 아티팩트 폴더 아래 관련 객체들이 저장됩니다.
        base_dt (str, optional): 해당 값이 없는 경우 오늘 날짜로 대체됩니다.
    """

    __DROP_COLS = ["base_dt", "id", "customer_id", "date"]

    __PARAMS_CANDIDATES = {
        "depth": [7, 8, 9],
        "rsm": [0.8, 0.9, 1.0],
        "l2_leaf_reg": [3, 5, 7],
    }

    __TEXT_COLS = ["type_of_loan", "payment_behaviour"]

    __CATEGORICAL_COLS = ["occupation", "credit_mix", "payment_of_min_amount"]

    def __init__(
        self,
        model_name: str,
        base_dt: str = DateValues.get_current_date(),
    ):
        self._model_name = model_name
        self._base_dt = base_dt
        self._preprocessing_path = os.path.join(
            artifacts_path,
            "preprocessing",
            self._model_name,
            self._base_dt,
        )
        self._model_path = os.path.join(
            artifacts_path,
            "models",
            self._model_name,
            self._base_dt,
        )
        # MLflow 실험 설정
        self._experiment_name = (
            f"training-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
        )
       
        self.is_trained = False
        self._make_dirs()

    def train(self):
        x_train, y_train, x_val, y_val = self._load_data()

        train_pool = self._create_pool(x=x_train, y=y_train)
        val_pool = self._create_pool(x=x_val, y=y_val)

        param_set = self._get_params_set(self.__PARAMS_CANDIDATES)

        mlflow.set_experiment(self._experiment_name)

        for i, params in enumerate(tqdm(param_set)):
            run_name = f"Run {i}"
            artifacts_path = os.path.join(
                self._model_path, run_name.lower().replace(" ", "_")
            )

            with mlflow.start_run(run_name=run_name):
                cls = CatBoostClassifier(
                    **params,
                    loss_function="MultiClassOneVsAll",
                    learning_rate=0.3,
                    iterations=2000,
                    thread_count=-1,
                    random_seed=42,
                    verbose=50,
                    custom_metric=["F1", "Accuracy"],
                    text_processing={
                        "tokenizers": [
                            {
                                "tokenizer_id": "Space",
                                "separator_type": "ByDelimiter",
                                "delimiter": " ",
                            },
                            {
                                "tokenizer_id": "Comma",
                                "separator_type": "ByDelimiter",
                                "delimiter": ",",
                            },
                            {
                                "tokenizer_id": "Underscore",
                                "separator_type": "ByDelimiter",
                                "delimiter": "_",
                            },
                        ],
                        "dictionaries": [
                            {
                                "dictionary_id": "BiGram",
                                "occurence_lower_bound": 1,
                            },
                            {
                                "dictionary_id": "Word",
                                "occurence_lower_bound": 1,
                            },
                        ],
                    },
                )
                cls.fit(
                    train_pool,
                    eval_set=val_pool,
                    early_stopping_rounds=50,
                )

                # MLflow logging
                # TODO: 1. estimator_name 을 태그로 저장
                mlflow.set_tag(
                    "estimator_name", cls.__class__.__name__
                )
                # TODO: 2. 파라미터 로깅
                mlflow.log_params(
                    {key: cls.get_all_params()[key] for key in params}
                )
                # TODO: 3. Early stopping한 모델의 최종 iterations를 파라미터로 저장
                mlflow.log_params(
                    {"iterations": cls.best_iteration_}
                )
                # TODO: 4. self._parse_score_dict를 이용해 검증셋에 대한 스코어를 메트릭으로 저장
                mlflow.log_metrics(                    
                    self._parse_score_dict(
                        cls.get_best_score().get("validation")
                    )
                )
                # TODO: 5. signature를 포함하여 모델 정보 로깅
                mlflow.catboost.log_model(
                    cls,
                    "CatBoostClassifier",
                    signature=infer_signature(x_train, cls.predict(x_train)),
                )
                # TODO: 6. 모델 객체 저장
                mlflow.catboost.save_model(
                    cls,
                    path=artifacts_path,
                )

        self.is_trained = True

    def get_best_model_info(self) -> Run:
        """실험에서 가장 성능이 좋은 모델의 정보를 반환합니다.
        성능 기준은 Accuracy입니다.


        Raises:
            AttributeError: 학습이 되지 않았다면 오류 발생
            AttributeError: 실험 정보가 없다면 오류 발생

        Returns:
            Run: 모델 정보
        """
        if not self.is_trained:
            raise AttributeError(
                "학습이 진행되지 않았습니다. 실험 결과를 가져올 수 없습니다."
            )

        # TODO: 최적 모델 탐색
        # mlflow.search_runs 메서드를 이용해
        # metrics.Accuracy를 내림차순으로 정렬하여 맨 위의 데이터를 best_run_df에 저장
        best_run_df = mlflow.search_runs(
            experiment_names=[self._experiment_name],
            order_by=["metrics.Accuracy DESC"],
            max_results=1,
        )

        if len(best_run_df) == 0:
            raise AttributeError(
                f"Found no runs for experiment '{self._experiment_name}'"
            )

        return mlflow.get_run(best_run_df.at[0, "run_id"])

    def save_model_with_bentoml(self, model_info: Run) -> None:
        """BentoML로 서빙하도록 모델과 메타데이터를 저장합니다.

        Args:
            model_info (Run): MLflow에서 얻은 모델 정보
        """
        model_uri = f"{model_info.info.artifact_uri}/CatBoostClassifier"
        model_params = model_info.data.params

        # TODO: 모델 저장
        # 1. Model URI로부터 모델을 불러오기
        # 2. 파라미터 정보를 메타데이터로 저장
        bentoml.catboost.save_model(
            name="credit_score_classifier",
            model=mlflow.catboost.load_model(model_uri),
            signatures={"predict": {"batchable": True, "batch_dim": 0}},
            metadata=model_params,
        )



    def _make_dirs(self) -> None:
        """기존 모델 경로가 존재하면 제거합니다.
        저장될 경로가 존재하지 않으면 해당 폴더를 생성합니다."""
        if os.path.exists(self._model_path):
            shutil.rmtree(self._model_path)
        if not os.path.isdir(self._model_path):
            os.makedirs(self._model_path)

    def _load_data(
        self,
    ) -> Tuple[pd.DataFrame, npt.NDArray, pd.DataFrame, npt.NDArray]:
        """데이터를 불러옵니다.
        불러온 데이터를 `x_train, y_train, x_val, y_val` 로 반환합니다.

        Returns:
            Tuple[pd.DataFrame, npt.NDArray, pd.DataFrame, npt.NDArray]: _description_
        """
        train = pd.read_csv(
            os.path.join(
                self._preprocessing_path, f"{self._model_name}_train.csv"
            )
        )
        val = pd.read_csv(
            os.path.join(
                self._preprocessing_path, f"{self._model_name}_val.csv"
            )
        )

        return (
            train.drop([TARGET_NAME] + self.__DROP_COLS, axis=1),
            train[TARGET_NAME].to_numpy(),
            val.drop([TARGET_NAME] + self.__DROP_COLS, axis=1),
            val[TARGET_NAME].to_numpy(),
        )

    def _create_pool(self, x: pd.DataFrame, y: npt.NDArray) -> Pool:
        """Catboost에 사용할 Pool을 생성합니다.
        범주형 변수와 텍스트 변수를 명시합니다.

        Args:
            x (pd.DataFrame): 피처 데이터
            y (npt.NDArray): 타겟값

        Returns:
            Pool
        """
        return Pool(
            data=x,
            label=y,
            cat_features=self.__CATEGORICAL_COLS,
            text_features=self.__TEXT_COLS,
        )

    @staticmethod
    def _get_params_set(params: Dict) -> List:
        """하이퍼파라미터 후보 딕셔너리를 리스트로 변환합니다.

        Args:
            params (Dict): 하이퍼파라미터 후보 딕셔너리

        Returns:
            List: 하이퍼파라미터 후보 리스트
        """
        params_keys = params.keys()
        params_values = [
            params[key] if isinstance(params[key], list) else [params[key]]
            for key in params_keys
        ]
        return [
            dict(zip(params_keys, combination))
            for combination in product(*params_values)
        ]

    @staticmethod
    def _parse_score_dict(score_dict: Dict) -> Dict:
        """Catboost 모델 결과 딕셔너리를 파싱합니다.
        MLflow에서는 딕셔너리 키에 `=`를 허용하지 않습니다.
        해당 문자를 공백으로 치환합니다.

        Args:
            score_dict (Dict): 모델 학습 결과

        Returns:
            Dict: 키 변환 후 모델 학습 결과
        """
        return {k.replace("=", " "): v for k, v in score_dict.items()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="An argument parser for training."
    )

    # TODO: 코드 작성
    # 1. 본 파일을 실행할 때는 두 개의 인자를 받음
    # 2. model_name은 문자열로 받으며, 기본값은 "credit_score_classification"
    # 3. base_dt는 문자열을 받으며 기본값은 DateValues.get_current_date()
    parser.add_argument(
        "--model_name", type=str, default="credit_score_classification"
    )
    parser.add_argument(
        "--base_dt", type=str, default=DateValues.get_current_date()
    )

    args = parser.parse_args()

    trainer = Trainer(model_name=args.model_name, base_dt=args.base_dt)
    trainer.train()
    model_info = trainer.get_best_model_info()
    trainer.save_model_with_bentoml(model_info=model_info)
