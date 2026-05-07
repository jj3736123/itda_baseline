# ITDA GNN Baseline Project

## 1. Project Overview

이 프로젝트는 Kaggle의 YelpZip dataset을 활용하여 조직적 리뷰 어뷰징을 탐지하기 위한 baseline model을 구축한 작업이다. 최종 모델 개발보다는 딥러닝 기반 분류 모델과 GNN 기반 분류 모델의 전체 흐름을 이해하고, MLP baseline과 GraphSAGE baseline의 성능을 비교하는 것을 목표로 한다.

본 프로젝트에서는 review 하나를 하나의 node로 정의하고, 각 review의 feature와 review 간 edge 정보를 활용하여 label을 예측한다.

- `label = 0`: 정상 리뷰
- `label = 1`: 사기 또는 스팸 리뷰

---

## 2. Dataset

사용 데이터는 YelpZip dataset이다. 원본 데이터는 약 60만 개 이상의 review로 구성되어 있으며, 주요 column은 다음과 같다.

| Column | 설명 |
|---|---|
| `user_id` | 리뷰를 작성한 사용자 ID |
| `prod_id` | 리뷰 대상 식당 또는 상품 ID |
| `rating` | 사용자가 부여한 별점 |
| `label` | 정상 리뷰와 사기 리뷰를 구분하는 target variable |
| `date` | 리뷰 작성 날짜 |
| `text` | 리뷰 본문 |
| `tag` | 원본 데이터에 포함된 tag 정보 |

원본 YelpZip dataset의 `label`은 사기 리뷰가 `-1`, 정상 리뷰가 `1`로 되어 있는데, 이는 각각 1과 0으로 바꿨다.

`tag` column은 `label`과 거의 동일한 정보를 포함하고 있어 data leakage 가능성이 있다고 판단하여 model feature에서 제외하였다.

---

## 3. Sampling and Split

전체 데이터를 그대로 사용하지 않고, product 중심의 subgraph sampling을 수행하였다. 단순 random sampling은 graph의 연결 구조를 약화시킬 수 있으므로, 리뷰 수가 많은 `prod_id`를 기준으로 상위 product의 review를 모아 총 15,000개의 review node를 구성하였다.

sampling 이후 같은 subgraph 안에서 train, valid, test split을 수행하였다.

| Split | 비율 |
|---|---:|
| `train` | 64% |
| `valid` | 16% |
| `test` | 20% |

split 과정에서는 `random_state = 42`를 사용하여 재현성을 확보하였다.

---

## 4. Node Feature Design

각 review node에 대해 다음과 같은 feature를 생성하였다.

| Feature | 설명 |
|---|---|
| `rating` | 리뷰 별점 |
| `text_length` | 리뷰 본문의 문자 수 |
| `word_count` | 리뷰 본문의 단어 수 |
| `user_review_count` | 해당 사용자가 sampled subgraph 안에서 작성한 리뷰 수 |
| `prod_review_count` | 해당 product가 sampled subgraph 안에서 받은 리뷰 수 |
| `days_from_start` | sampled data의 가장 이른 날짜로부터 해당 리뷰 작성일까지 지난 일수 |

`user_id`와 `prod_id`는 숫자형 ID이지만 크기 자체에 의미가 없으므로 node feature로 직접 사용하지 않았다. 대신 `user_review_count`, `prod_review_count`처럼 해석 가능한 파생변수로 변환하였다.

---

## 5. Edge Design

GraphSAGE baseline에서는 두 종류의 edge를 사용하였다.

### 5.1 R-U-R Edge

`R-U-R`은 Review-User-Review relation을 의미한다. 같은 `user_id`가 작성한 review node들을 서로 연결하였다.

즉, 같은 사용자가 여러 개의 리뷰를 작성한 경우, 해당 리뷰들은 같은 사용자라는 관계를 기준으로 연결된다.

### 5.2 Product-Time Edge

`Product-Time` edge는 본 프로젝트에서 설계한 custom relation이다. 같은 `prod_id`에 대해 30일 이내에 작성된 review node들을 서로 연결하였다.

이 edge는 특정 product에 대해 짧은 기간 안에 리뷰가 집중되는 패턴을 반영하기 위한 것이다. 조직적 리뷰 어뷰징은 특정 대상에 대해 비슷한 시점에 리뷰가 몰리는 형태로 나타날 수 있기 때문에, 시간적으로 가까운 product-level review 관계를 edge로 반영하였다.

### 5.3 Undirected Graph 처리

본 프로젝트의 graph는 개념적으로 undirected graph이다. 다만 PyTorch Geometric에서는 message passing을 위해 undirected edge를 양방향 directed edge로 저장한다.

예를 들어 개념적으로 다음과 같은 관계가 있다면,

```text
review 10 -- review 25
```

PyTorch Geometric의 `edge_index`에는 다음과 같이 저장된다.

```text
10 → 25
25 → 10
```

또한 R-U-R edge와 Product-Time edge가 같은 node pair를 동시에 생성할 가능성이 있으므로, undirected pair 기준으로 중복 edge를 제거한 뒤 PyG용 양방향 `edge_index`를 생성하였다.

---

## 6. Models

본 프로젝트에서는 두 가지 baseline model을 비교하였다.

### 6.1 MLP Baseline

MLP baseline은 graph structure를 사용하지 않고, 각 review node의 feature만을 사용하여 `label`을 예측한다.

사용한 feature는 다음과 같다.

- `rating`
- `text_length`
- `word_count`
- `user_review_count`
- `prod_review_count`
- `days_from_start`

MLP baseline은 GraphSAGE와 비교하기 위한 non-graph baseline 역할을 한다.

### 6.2 GraphSAGE Baseline

GraphSAGE baseline은 MLP와 동일한 node feature를 사용하되, 추가로 R-U-R edge와 Product-Time edge를 활용한다.

GraphSAGE는 연결된 이웃 node의 feature를 aggregate하여 각 node의 representation을 업데이트한다. 따라서 단순히 review 자체의 정보만 보는 MLP와 달리, GraphSAGE는 주변 review와의 관계 정보를 함께 학습한다.

---

## 7. Evaluation Metrics

본 프로젝트에서는 다음 세 가지 metric을 사용하였다.

| Metric | 설명 |
|---|---|
| `Accuracy` | 전체 예측 중 정답 비율 |
| `Macro F1` | class별 F1-score를 동일한 비중으로 평균낸 값 |
| `PR-AUC` | positive class 탐지 성능을 평가하는 precision-recall 기반 지표 |

이 데이터는 정상 리뷰가 사기 리뷰보다 많은 imbalanced classification 문제이므로, `Accuracy`만으로 모델을 평가하기 어렵다. 따라서 `Macro F1`과 `PR-AUC`를 주요 성능 지표로 함께 확인하였다.

---

## 8. Results

| Model | Accuracy | Macro F1 | PR-AUC |
|---|---:|---:|---:|
| MLP | 0.8527 | 0.6091 | 0.2514 |
| GraphSAGE | 0.8447 | 0.6136 | 0.2632 |

GraphSAGE는 MLP보다 `Accuracy`는 소폭 낮았지만, `Macro F1`과 `PR-AUC`는 개선되었다. 특히 fraud class의 recall이 MLP보다 상승하여, graph structure가 사기 리뷰 탐지에 일부 도움을 준 것으로 해석할 수 있다.

다만 성능 향상 폭이 크지는 않으며, fraud class의 precision과 recall이 여전히 낮기 때문에 최종 모델로 사용하기에는 한계가 있다. 본 결과는 GraphSAGE baseline이 MLP baseline보다 관계 정보를 일부 활용했다는 초기 실험 결과로 해석하는 것이 적절하다.





