import os
import time
import warnings

import bentoml
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from api.src.db import get_db
from api.src.models import CreditPredictionApiLog
from api.src.schemas import Features, Response
from utils.dates import DateValues

warnings.filterwarnings(action="ignore")

# .env 파일 로드
load_dotenv()

MODEL_NAME = "credit_score_classification"
BASE_DT = DateValues.get_current_date()

artifacts_path = os.getenv("ARTIFACTS_PATH")
encoder_path = os.path.join(
    artifacts_path, "preprocessing", MODEL_NAME, BASE_DT, "encoders"
)


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class CreditScoreClassifier:
    def __init__(self, db: Session = next(get_db())):
        self.db = db
        self.bento_model = bentoml.models.get("credit_score_classifier:latest")
        self.robust_scalers = joblib.load(
            os.path.join(encoder_path, "robust_scaler.joblib")
        )
        self.model = bentoml.catboost.load_model(self.bento_model)

    @bentoml.api
    def predict(self, data: Features) -> Response:
        start_time = time.time()
        df = pd.DataFrame([data.model_dump()])
        customer_id = df.pop("customer_id")
        
        # TODO: RobustScaler 적용
        for col, scaler in self.robust_scalers.items():
            df[col] = scaler.transform(df[[col]])        

        # TODO: 모델 추론 결과로 확률값과 예측 레이블을 저장
        prob = np.max(self.model.predict(df, prediction_type="Probability"))
        label = self.model.predict(df, prediction_type="Class").item()
        
        elapsed_ms = (time.time() - start_time) * 1000

        record = CreditPredictionApiLog(
            # TODO: 기본값이 존재하지 않는 컬럼에 적절한 값 매핑
            customer_id=customer_id.item(),
            features=data.model_dump(),
            prediction=label,
            confidence=prob,
            elapsed_ms=elapsed_ms,            
        )
        
        # TODO: 로깅할 값을 테이블에 적재 후 커밋
        self.db.add(record)
        self.db.commit()

        return Response(customer_id=customer_id, predict=label, confidence=prob)

    @bentoml.api(route="/metadata", output_spec=dict)
    def metadata(self):
        """현재 컨테이너에서 서빙 중인 모델의 메타데이터 반환"""
        return {
            "model_name": self.bento_model.tag.name,
            "model_version": self.bento_model.tag.version,
            "params": self.bento_model.info.metadata,
            "creation_time": self.bento_model.info.creation_time,
        }
