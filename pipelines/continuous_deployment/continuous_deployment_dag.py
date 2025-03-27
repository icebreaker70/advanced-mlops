from datetime import datetime
from typing import List

import bentoml
import pendulum
import requests
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.models import Variable

from utils.callbacks import failure_callback, success_callback


local_timezone = pendulum.timezone("Asia/Seoul")
airflow_dags_path = Variable.get("AIRFLOW_DAGS_PATH")


def get_branch_by_api_status() -> List[str] | str:
    try:
        response = requests.get("http://localhost:3000/healthz")
        if response.status_code == 200:
            return [
                "get_deployed_model_creation_time",
                "get_latest_trained_model_creation_time",
            ]
        else:
            return "deploy_new_model"
    except Exception as e:
        print(f"API 통신이 이루어지지 않았습니다.: {e}")
        return "deploy_new_model"


def get_deployed_model_creation_time() -> datetime | None:
    """이미 배포된 모델의 `creation_time`을 조회합니다."""
    try:
        response = requests.post("http://localhost:3000/metadata")
        if response.status_code == 200:
            return datetime.strptime(
                response.json().get("creation_time"), "%Y-%m-%dT%H:%M:%S.%fZ"
            )
        else:
            print(
                f"`creation_time`을 불러올 수 없습니다.: {response.status_code}"
            )
            return None
    except Exception as e:
        print(f"배포된 모델의 API를 받아오지 못했습니다.: {e}")
        return None


def get_latest_trained_model_creation_time() -> datetime | None:
    """로컬 저장소에 저장된 최신 학습 모델의 `creation_time` 조회합니다."""
    try:
        bento_model = bentoml.models.get("credit_score_classifier:latest")
        return bento_model.info.creation_time.replace(tzinfo=None)
    except Exception as e:
        print(f"Error getting latest trained model creation time: {e}")
        return None


def decide_model_update(ti):
    """
    현재 배포된 모델과 로컬 최신 학습 모델의 creation_time 비교.
    배포된 모델이 오래되었으면 새로운 모델을 배포하도록 결정.
    """
    api_status = ti.xcom_pull(task_ids="get_branch_by_api_status")

    if api_status == "deploy_new_model":
        return "deploy_new_model"

    deployed_creation_time = ti.xcom_pull(
        task_ids="get_deployed_model_creation_time"
    )
    trained_creation_time = ti.xcom_pull(
        task_ids="get_latest_trained_model_creation_time"
    )

    print("deployed_creation_time", deployed_creation_time)
    print("trained_creation_time", trained_creation_time)

    if deployed_creation_time is None:
        print("There is no deployed model!")
        return "deploy_new_model"

    if (
        trained_creation_time is not None
        and trained_creation_time > deployed_creation_time
    ):
        print("Deployed model is already out-of-date.")
        return "deploy_new_model"

    print("Skip deployment.")
    return "skip_deployment"


with DAG(
    dag_id="credit_score_classification_cd",
    default_args={
        "owner": "user",
        "depends_on_past": False,
        "email": ["otzslayer@gmail.com"],
        "on_failure_callback": failure_callback,
        "on_success_callback": success_callback,
    },
    description="A DAG for continuous deployment",
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=local_timezone),
    catchup=False,
    tags=["lgcns", "mlops"],
) as dag:
    # API 상태 체크 결과 가져오기
    get_api_status_task = BranchPythonOperator(
        task_id="get_branch_by_api_status",
        python_callable=get_branch_by_api_status,
        provide_context=True,
    )

    # 현재 컨테이너에서 실행 중인 모델의 creation_time 가져오기
    get_deployed_model_creation_time_task = PythonOperator(
        task_id="get_deployed_model_creation_time",
        python_callable=get_deployed_model_creation_time,
    )

    # 로컬에서 최신 학습된 모델의 creation_time 가져오기
    get_latest_trained_model_creation_time_task = PythonOperator(
        task_id="get_latest_trained_model_creation_time",
        python_callable=get_latest_trained_model_creation_time,
    )

    # 모델을 업데이트할지 결정
    decide_update_task = BranchPythonOperator(
        task_id="decide_update",
        python_callable=decide_model_update,
        provide_context=True,
    )

    # 새로운 모델을 배포
    deploy_new_model_task = BashOperator(
        task_id="deploy_new_model",
        bash_command=f"cd {airflow_dags_path}/api/docker &&"
        "docker compose up --build --detach",
    )

    # 배포를 건너뛸 경우 실행할 더미 태스크
    skip_deployment_task = PythonOperator(
        task_id="skip_deployment",
        python_callable=lambda: print("No new model to deploy"),
    )

    # DAG 실행 순서 정의
    # 1️⃣ API가 정상 동작하지 않으면 즉시 배포
    get_api_status_task >> deploy_new_model_task

    # 2️⃣ API가 정상 동작하면 모델 생성 시간 비교 후 업데이트 결정
    (
        get_api_status_task
        >> [
            get_deployed_model_creation_time_task,
            get_latest_trained_model_creation_time_task,
        ]
        >> decide_update_task
    )

    # 3️⃣ decide_update_task의 결과에 따라 모델 배포 여부 결정
    decide_update_task >> [deploy_new_model_task, skip_deployment_task]