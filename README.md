# 🎬 Netflix Director Style Analysis (Open Source Format)

## ✅ Function Definition

이 오픈소스 프로젝트는 Netflix 콘텐츠의 감독별 스타일을 분석하고, 
이를 기반으로 장르 예측 모델을 학습하여 새로운 콘텐츠의 장르를 예측하는 기능을 제공합니다.

- Netflix 메타데이터 전처리 및 통합 장르 생성
- 감독별 평균 콘텐츠 특성 기반 KMeans 클러스터링
- 콘텐츠 특성 기반 Random Forest / Decision Tree 분류기 학습
- 하이퍼파라미터 튜닝 (GridSearch)
- 모델 평가 및 시각화 (Confusion Matrix, F1-score 등)
- 교차검증 기반 성능 분석

## ✅ Architecture

```
📁 netflix_director_style/
├── preprocessing.py             # 원본 csv 전처리 → 학습 가능한 최종 CSV 생성
├── Clustering_KMeans.py         # 감독별 평균 특성 기반 KMeans 클러스터링
├── Classification_DecisionTree.py  # 콘텐츠 기반 장르 분류 (결정 트리)
├── Random_Forest.py             # 콘텐츠 기반 장르 분류 (랜덤 포레스트)
├── GridSearch_Tuning.py         # 랜덤 포레스트 파라미터 튜닝
├── cross_validation.py          # 결정 트리 교차 검증 (F1-macro)
├── Evaluation_Metrics.py        # 분류/클러스터링 평가 함수 정의 및 시각화
├── netflix_titles.csv           # 원본 데이터
├── netflix_preprocessed_final.csv  # 전처리된 최종 입력 데이터
```

### 데이터 흐름 구조

```
netflix_titles.csv
        ↓  (preprocessing.py)
netflix_preprocessed_final.csv
        ↓
 ┌────────────────────┬──────────────────────────┐
 │ Clustering_KMeans  │ Classification_DecisionTree / Random_Forest
 └────────────────────┴──────────────────────────┘
        ↓
Evaluation_Metrics.py, cross_validation.py, GridSearch_Tuning.py
```

## ✅ GitHub Repository
> https://github.com/sunwoo202335248/netflix-director-style

---

각 파이썬 파일은 함수 기반으로 작성되어 있으며, 독립 실행 또는 모듈 호출로도 사용 가능합니다.
