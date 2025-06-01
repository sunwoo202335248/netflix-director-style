# 🎬 Netflix Director Style Analysis

## ✅ Function Definition

이 프로젝트는 Netflix 콘텐츠의 감독별 스타일을 분석하고, 
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

## 데이터 흐름 구조

```
[netflix_titles.csv]
        ↓ (preprocessing.py)
[netflix_preprocessed_final.csv]
        ↓
    ┌──────────────────────────────┐
    │      Clustering_KMeans.py    │
    └──────────────────────────────┘
                  ↓
    ┌──────────────────────────────┐
    │   Classification Models      │
    │   - DecisionTree             │
    │   - RandomForest             │
    └──────────────────────────────┘
                  ↓
    ┌──────────────────────────────┐
    │   Evaluation_Metrics.py      │
    │   GridSearch_Tuning.py       │
    │   cross_validation.py        │
    └──────────────────────────────┘
```

## ✅ Execution-Based Architecture

```
[netflix_titles.csv]
        ↓ (1. Preprocessing.py)
[netflix_preprocessed_final.csv]
        ↓
    ┌────────────────────────────────────┐
    │     2-1. GridSearch_Tuning.py      │  ← Decision Tree 하이퍼파라미터 최적화
    └────────────────────────────────────┘
                  ↓
    ┌────────────────────────────────────┐
    │   2-2. Classification_DecisionTree │  ← 결정 트리 모델 학습 및 평가
    └────────────────────────────────────┘
                  ↓
    ┌────────────────────────────────────┐
    │     2-3. Random_Forest.py          │  ← 랜덤 포레스트 모델 학습 및 평가
    └────────────────────────────────────┘
                  ↓
    ┌────────────────────────────────────┐
    │     3. cross_validation.py         │  ← 모델 일반화 성능 교차 검증
    └────────────────────────────────────┘
                  ↓
    ┌────────────────────────────────────┐
    │     4. Clustering_KMeans.py        │  ← 감독 스타일 클러스터링
    └────────────────────────────────────┘
                  ↓
    ┌────────────────────────────────────┐
    │     5. Evaluation_Metrics.py       │  ← 성능 평가 함수 (분류/클러스터링)
    └────────────────────────────────────┘
```

## ✅ 실행 순서 요약표 

| 단계 | 스크립트                         | 목적                                          |
|------|----------------------------------|-----------------------------------------------|
| 1    | Preprocessing.py                | 원본 데이터 전처리 및 저장                    |
| 2-1  | GridSearch_Tuning.py            | Decision Tree 하이퍼파라미터 최적화           |
| 2-2  | Classification_DecisionTree.py  | Decision Tree 모델 평가                       |
| 2-3  | Random_Forest.py                | RandomForest 최종 모델 학습 및 평가           |
| 3    | cross_validation.py             | 교차 검증 통한 모델 일반화 성능 확인          |
| 4    | Clustering_KMeans.py            | KMeans로 감독 스타일 시각적 분류              |
| 5    | Evaluation_Metrics.py           | 분류/클러스터링 성능 평가 함수 정의           |



