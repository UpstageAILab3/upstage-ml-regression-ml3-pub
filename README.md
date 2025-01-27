[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/D1pZhJxu)
# '디지털 보물찾기(Digital Treasure Quest)' 팀의 '아파트 실거래가 예측' 경연 도전기

## Team

| ![박석](https://avatars.githubusercontent.com/u/5678836?v=4) | ![백경탁](https://avatars.githubusercontent.com/u/62689715?v=4) | ![한아름](https://avatars.githubusercontent.com/u/121337152?v=4) | ![이승현](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이한국](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박석](https://github.com/parksurk)             |            [백경탁](https://github.com/UpstageAILab)             |            [한아름](https://github.com/UpstageAILab)             |            [이승현](https://github.com/UpstageAILab)             |            [이한국](https://github.com/UpstageAILab)             |
|                            Lead, R&D                             |                            R&D                             |                            R&D                             |                            R&D                             |                            R&D                             |

![박석](./images/team-member-ps.png)
![백경탁](./images/team-member-pkt.png)
![한아름](./images/team-member-har.png)
![이승현](./images/team-member-lsh.png)
![이한국](./images/team-member-lhk.png)

## Table of Contents
- ['디지털 보물찾기(Digital Treasure Quest)' 팀의 '아파트 실거래가 예측' 경연 도전기](#디지털-보물찾기digital-treasure-quest-팀의-아파트-실거래가-예측-경연-도전기)
  - [Team](#team)
  - [Table of Contents](#table-of-contents)
  - [1. Competiton Info](#1-competiton-info)
    - [1.1. Overview](#11-overview)
      - [대회 개요](#대회-개요)
        - [목표](#목표)
        - [소개](#소개)
        - [제공 데이터셋](#제공-데이터셋)
        - [사용 가능한 알고리즘](#사용-가능한-알고리즘)
        - [모델링 목표](#모델링-목표)
        - [제출 형식](#제출-형식)
    - [1.2. Timeline](#12-timeline)
      - [프로젝트 전체 기간](#프로젝트-전체-기간)
      - [주요 일정](#주요-일정)
      - [상세 일정](#상세-일정)
    - [1.3. Evaluation](#13-evaluation)
      - [평가방법](#평가방법)
        - [RMSE란?](#rmse란)
        - [평가 기준](#평가-기준)
        - [계산 방법](#계산-방법)
        - [맥락](#맥락)
  - [2. Winning Strategy](#2-winning-strategy)
    - [2.1. DTQ Team's Pros and Cons](#21-dtq-teams-pros-and-cons)
      - [Pros](#pros)
      - [Cons](#cons)
    - [2.2. DTQ Team's strategic approach](#22-dtq-teams-strategic-approach)
    - [2.3. DTQ Team's culture \& spirit](#23-dtq-teams-culture--spirit)
  - [3. Components](#3-components)
    - [3.1. Directory](#31-directory)
  - [4. Data descrption](#4-data-descrption)
    - [4.1. Dataset overview](#41-dataset-overview)
      - [학습데이터](#학습데이터)
      - [예측데이터](#예측데이터)
    - [4.2. EDA](#42-eda)
      - [Feature Description](#feature-description)
      - [Excess zeros](#excess-zeros)
      - [Outliers](#outliers)
      - [Disguised missing values](#disguised-missing-values)
      - [Inliers](#inliers)
      - [Target leakage](#target-leakage)
    - [4.3. Feature engineering](#43-feature-engineering)
      - [One-Hot Encoding](#one-hot-encoding)
      - [Missing Values Imputed](#missing-values-imputed)
      - [Smooth Ridit Transform](#smooth-ridit-transform)
      - [Binning of numerical variables](#binning-of-numerical-variables)
      - [Matrix of char-grams occurrences using tfidf](#matrix-of-char-grams-occurrences-using-tfidf)
      - [Feature Selection](#feature-selection)
        - [Input Features](#input-features)
        - [Target Feature](#target-feature)
  - [5. Modeling](#5-modeling)
    - [5.1. Model Selection](#51-model-selection)
      - [모델 검증 안정성](#모델-검증-안정성)
      - [데이터 분할 방법론](#데이터-분할-방법론)
      - [Selected Models](#selected-models)
    - [5.2. eXtreme Gradient Boosted Trees Regressor(DataRobot)](#52-extreme-gradient-boosted-trees-regressordatarobot)
      - [Modeling Descriptions](#modeling-descriptions)
      - [Modeling Process](#modeling-process)
        - [Hiperparameters](#hiperparameters)
        - [Feature Impact](#feature-impact)
        - [Word Cloud](#word-cloud)
    - [5.3. Keras Slim Residual Network Regressor(DataRobot)](#53-keras-slim-residual-network-regressordatarobot)
      - [Modeling Descriptions](#modeling-descriptions-1)
      - [Modeling Process](#modeling-process-1)
        - [Neural Network](#neural-network)
        - [Hiperparameters](#hiperparameters-1)
        - [Training](#training)
        - [Feature Impact](#feature-impact-1)
        - [Word Cloud](#word-cloud-1)
    - [5.4. Light Gradient Boosted Trees Regressor(DataRobot)](#54-light-gradient-boosted-trees-regressordatarobot)
      - [Modeling Descriptions](#modeling-descriptions-2)
      - [Modeling Process](#modeling-process-2)
        - [Hiperparameters](#hiperparameters-2)
        - [Feature Impact](#feature-impact-2)
        - [Word Cloud](#word-cloud-2)
    - [5.5. PyTorch Residual Network Regressor(박석)](#55-pytorch-residual-network-regressor박석)
      - [Modeling Descriptions](#modeling-descriptions-3)
        - [PyTorch를 활용한 Residual Network 모델링 설명](#pytorch를-활용한-residual-network-모델링-설명)
          - [Step 1: 라이브러리 임포트](#step-1-라이브러리-임포트)
          - [Step 2: 데이터 전처리 함수](#step-2-데이터-전처리-함수)
          - [Step 3: 커스텀 데이터셋 클래스](#step-3-커스텀-데이터셋-클래스)
          - [Step 4: Residual Network 모델 정의](#step-4-residual-network-모델-정의)
          - [Step 5: 학습 및 추론 함수](#step-5-학습-및-추론-함수)
        - [모델링 시 주목할 만한 사항](#모델링-시-주목할-만한-사항)
      - [Modeling Process](#modeling-process-3)
        - [Hiperparameters](#hiperparameters-3)
        - [Training](#training-1)
        - [Trial history](#trial-history)
        - [Addiional Trial shared UpStage AI Stages Community](#addiional-trial-shared-upstage-ai-stages-community)
    - [5.6. RandomForestRegressor(백경탁)](#56-randomforestregressor백경탁)
      - [Baseline Code 수정하여 모델별 성능 비교 :](#baseline-code-수정하여-모델별-성능-비교-)
      - [Daily Log](#daily-log)
        - [2024.07.16](#20240716)
        - [2024-07-17](#2024-07-17)
        - [2024-07-19](#2024-07-19)
      - [Trial history](#trial-history-1)
        - [1회 제출](#1회-제출)
        - [2회 제출](#2회-제출)
        - [3회 제출](#3회-제출)
        - [4회 제출](#4회-제출)
        - [5회 제출](#5회-제출)
        - [6회 제출](#6회-제출)
        - [7/8회 제출](#78회-제출)
        - [9회 제출](#9회-제출)
        - [10회 제출](#10회-제출)
        - [11/12회 제출](#1112회-제출)
        - [13/14회 제출](#1314회-제출)
      - [Mentoring list](#mentoring-list)
        - [2024-07-18 오전 9:30](#2024-07-18-오전-930)
    - [5.6. Light GBM(한아름)](#56-light-gbm한아름)
      - [Trail history](#trail-history)
        - [Baseline Code 로 모델별 성능 비교 :](#baseline-code-로-모델별-성능-비교-)
        - [Light GBM 모델로 Time Series K-Fold 데이터 분할 후 성과 측정 :](#light-gbm-모델로-time-series-k-fold-데이터-분할-후-성과-측정-)
      - [Metoring list](#metoring-list)
        - [2024-07-18 오전 9:30](#2024-07-18-오전-930-1)
        - [1대1 첨삭 지도](#1대1-첨삭-지도)
    - [5.7. Baseline Code Enhancement(이승현)](#57-baseline-code-enhancement이승현)
  - [6. Result](#6-result)
    - [Leader Board](#leader-board)
      - [Final - Rank 3](#final---rank-3)
      - [Submit history](#submit-history)
    - [Presentation](#presentation)
  - [etc](#etc)
    - [Meeting Log](#meeting-log)
    - [Reference](#reference)

## 1. Competiton Info

### 1.1. Overview

#### 대회 개요

##### 목표
서울시 아파트 실거래가 데이터를 바탕으로 아파트 가격을 예측하는 모델을 개발합니다.

##### 소개
House Price Prediction 경진대회는 주어진 데이터를 활용해 서울의 아파트 실거래가를 예측하는 모델을 개발하는 대회입니다. 아파트 가격은 주변 요소(강, 공원, 백화점 등)와 같은 다양한 요인에 영향을 받습니다. 예측 모델은 이러한 요인들을 고려해 정확한 시세를 예측하고, 부동산 거래를 도울 수 있습니다.

##### 제공 데이터셋
1. **아파트 실거래가 데이터**: 위치, 크기, 건축 연도, 주변 시설 및 교통 편의성 등 포함
2. **지하철역 정보**: 서울시에서 제공
3. **버스정류장 정보**: 서울시에서 제공
4. **평가 데이터**: 모델 성능 검증용

##### 사용 가능한 알고리즘
선형 회귀, 결정 트리, 랜덤 포레스트, 딥 러닝 등 다양한 회귀 알고리즘을 사용할 수 있습니다.

##### 모델링 목표
정확하고 일반화된 모델을 개발해 아파트 시장의 동향을 예측하고, 부동산 관련 의사 결정을 돕는 것입니다. 참가자들은 모델 성능을 평가하고 다양한 특성 간의 상관 관계를 이해함으로써 실전 경험을 쌓을 수 있습니다.

##### 제출 형식
CSV 파일로 결과물을 제출합니다.

- **Input**: 9,272개의 아파트 특징 및 거래 정보
- **Output**: 9,272개의 예상 아파트 거래 금액

자세한 내용은 [대회 페이지](https://stages.ai/en/competitions/312/overview/description)에서 확인하세요.

### 1.2. Timeline

#### 프로젝트 전체 기간
- **7월 9일 (화) 10:00 ~ 7월 19일 (금) 19:00**

#### 주요 일정
- **대회 시작**: 7월 9일 (화) 10:00
- **팀 병합 마감**: 7월 10일 (수) 10:00
- **개발 및 테스트 기간**: 7월 9일 (화) 10:00 ~ 7월 18일 (목) 19:00
- **최종 모델 제출**: 7월 19일 (금) 19:00

#### 상세 일정
1. **7월 9일 (화)**: 데이터셋 배포 및 대회 시작
   - 데이터 탐색 및 전처리 시작
   - 팀 구성 및 역할 분담
2. **7월 10일 (수)**: 팀 병합 마감
   - 초기 모델 개발 시작
   - 데이터 전처리 완료
3. **7월 11일 (목)**: 모델 성능 검증
   - 모델 학습 및 초기 결과 분석
   - 모델 성능 향상 방안 논의
4. **7월 12일 (금)**: 모델 개선
   - 다양한 알고리즘 시도
   - 파라미터 튜닝
5. **7월 13일 (토)**: 모델 테스트
   - 검증 데이터로 모델 성능 평가
   - 에러 분석 및 수정
6. **7월 14일 (일)**: 피드백 반영
   - 피드백 기반 모델 수정 및 개선
   - 추가 데이터 수집 및 통합
7. **7월 15일 (월)**: 최적화
   - 모델 최적화 및 성능 극대화
   - 추가 피처 엔지니어링
8. **7월 16일 (화)**: 결과 검토
   - 최종 모델 검토 및 테스트
   - 결과 분석 및 문서화 시작
9. **7월 17일 (수)**: 문서화
   - 모델 개발 과정 및 결과 문서화
   - 최종 검토 및 수정
10. **7월 18일 (목)**: 최종 점검
    - 최종 모델 점검 및 제출 준비
    - 최종 테스트 및 결과 확인
11. **7월 19일 (금)**: 최종 모델 제출 및 대회 종료
    - 최종 모델 제출
    - 결과 발표 준비

### 1.3. Evaluation

#### 평가방법

이번 대회는 주어진 시점의 아파트 매매 실거래가를 예측하는 **회귀 대회**입니다. 참가자들이 개발한 모델은 **RMSE** 를 평가지표로 사용하여 평가됩니다.

##### RMSE란?
RMSE는 예측된 값과 실제 값 간의 평균편차를 측정하는 지표입니다. 이 값은 예측 오차의 제곱 평균을 구한 후, 이를 다시 제곱근으로 변환하여 계산됩니다.

##### 평가 기준
- **예측 정확도**: 모델이 실제 거래 가격과 예측 가격 간의 차이를 얼마나 잘 잡아내는지를 평가합니다.
- **낮은 RMSE 값**: RMSE 값이 낮을수록 모델의 예측 성능이 우수함을 의미합니다.

##### 계산 방법
![RMSE](./images/rmse-desc.png)

##### 맥락
아파트 매매의 맥락에서 RMSE는 회귀 모델이 실제 거래 가격과 얼마나 일치하는지를 정량적으로 나타내며, 예측 모델의 성능을 평가하는 데 중요한 역할을 합니다.

## 2. Winning Strategy

### 2.1. DTQ Team's Pros and Cons
#### Pros
- 다양한 경력과 경험을 가진 팀원들
- 평균 나이가 높음
- AI Assistant에 대한 수용력이 높음

#### Cons 
- Git을 활용한 팀단위의 R&D 경험 수준 낮음
- Python기반 R&D 경험 수준 낮음
- 머신러닝/딥러닝 R&D 경험 수준 낮음
- 경연 주제와 관련된 도메인 지식이 낮음
- Career Path에 대한 개인적인 목표가 모두 다름 

### 2.2. DTQ Team's strategic approach
- 첫째, DataRobot과 같은 AutoML 도구를 적극 활용하여 Feature Engineering 과 Model Selection 의 방향성을 잡는다.
- 둘째, 팀원별 서로 다른 머신러닝 모델링을 각 팀원별 수준에 맞게 진행한다.

### 2.3. DTQ Team's culture & spirit
- 경연 참가의 목적은 개인별 학습을 통해 머신러닝 R&D에 필요한 지식과 경험을 얻는 것에 있다.
- 팀원 각각이 처한 상황을 서로 이해하고 인정하고 Respect 한다.
- AI Assistant를 적극적으로 활용하여 개인별 생산성을 극대화 한다.
- 팀 전체 목표을 위해 팀원 개개인의 스케쥴이나 리소스를 희생해서는 안된다.
- 팀원별로 최소한 한번의 제출은 해 본다.

## 3. Components

### 3.1. Directory

- code : 팀원별 실험 소스 코드 및 관련 문서
  - tm1 : 팀원(박석) 실험 소스 코드 및 관련 문서
  - tm2 : 팀원(백경탁) 실험 소스 코드 및 관련 문서
  - tm3 : 팀원(한아름) 실험 소스 코드 및 관련 문서
  - tm4 : 팀원(이승현) 실험 소스 코드 및 관련 문서
  - tm5 : 팀원(이한국) 실험 소스 코드 및 관련 문서
- docs : 팀 문서(발표자료, 참고자료 등)
  - presentation : 발표자료
  - reference : 참고자료
- images : 첨부 이미지
- README.md : 디지털 보물찾기(Digital Treasure Quest)' 팀의 '아파트 실거래가 예측' 경연 도전기 Readme.md
  
## 4. Data descrption

### 4.1. Dataset overview

#### 학습데이터
| **Desc**   | **Details**  |
|----------------|--------------|
| File name      | train.csv    |
| Rows           | 1,118,822    |
| Features       | 52           |
| Numeric        | 22           |
| Text           | 5            |
| Categorical    | 18           |
| Date           | 7            |
| Size           | 244 MB       |

#### 예측데이터
| **Desc**   | **Details**  |
|----------------|--------------|
| File name      | test.csv     |
| Rows           | 9,272        |
| Features       | 51           |
| Numeric        | 22           |
| Text           | 5            |
| Categorical    | 17           |
| Date           | 7            |
| Size           | 2.46 MB      |



### 4.2. EDA

#### Feature Description
| Feature Name | Index | Importance | Var Type   | Unique  | Missing | Mean                  | Std Dev     | Median                | Min                   | Max                 |
|--------------|-------|------------|------------|---------|---------|-----------------------|-------------|-----------------------|-----------------------|---------------------|
| target       | 52    |   Target    | Numeric    | 12,875  | 0       | 57,963                | 46,348      | 44,750                | 500                   | 1,450,000           |
| 도로명         | 11    | 1          | Text       | 9,195   | 973     |                       |             |                       |                       |                     |
| 아파트명       | 5     | 2          | Text       | 6,522   | 1,712   |                       |             |                       |                       |                     |
| 시군구         | 1     | 3          | Text       | 338     | 0       |                       |             |                       |                       |                     |
| 전용면적(㎡)   | 6     | 4          | Numeric    | 14,218  | 0       | 77.16                 | 29.38       | 81.86                 | 10.02                 | 424                 |
| 번지          | 2     | 5          | Categorical| 6,555   | 184     |                       |             |                       |                       |                     |
| 계약년월       | 7     | 6          | Numeric    | 198     | 0       | 201,476               | 419         | 201,507               | 200,701               | 202,306             |
| k_시행사      | 27    | 7          | Text       | 553     | 697,757 |                       |             |                       |                       |                     |
| 건축년도       | 10    | 8          | Numeric    | 60      | 0       | 1,999                 | 9.33        | 2,000                 | 1,961                 | 2,023               |
| k_전화번호    | 17    | 9          | Categorical| 1,078   | 696,280 |                       |             |                       |                       |                     |
| k_팩스번호    | 18    | 10         | Categorical| 1,356   | 698,258 |                       |             |                       |                       |                     |
| 고용보험관리번호 | 39    | 11         | Categorical| 526     | 730,705 |                       |             |                       |                       |                     |
| k_건설사(시공사) | 26    | 12         | Text       | 344     | 696,911 |                       |             |                       |                       |                     |
| k_홈페이지    | 36    | 13         | Categorical| 220     | 804,418 |                       |             |                       |                       |                     |
| 좌표Y         | 50    | 14         | Numeric    | 740     | 695,785 | 37.55                 | 0.05        | 37.54                 | 37.45                 | 37.69               |
| k_연면적      | 29    | 15         | Numeric    | 733     | 695,698 | 161,461               | 183,831     | 101,633               | 0                     | 9,591,851           |
| k_사용검사일_사용승인일 | 28 | 16 | Date | 672 | 695,814 | 2002-05-20T18:39:16.828813 | 3231.17 days | 2003-10-04T00:00:00 | 1976-07-09T00:00:00 | 2023-01-27T00:00:00 |
| k_주거전용면적 | 30    | 17         | Numeric    | 738     | 695,737 | 94,170                | 101,750     | 60,280                | 2,338                 | 734,781             |
| 주차대수       | 44    | 18         | Numeric    | 525     | 695,817 | 1,064                 | 1,235       | 683                   | 0                     | 12,096              |
| k_복도유형    | 22    | 19         | Categorical| 5       | 695,962 |                       |             |                       |                       |                     |
| k_관리비부과면적 | 31    | 20         | Numeric    | 734     | 695,698 | 120,667               | 128,828     | 78,125                | 0                     | 969,877             |
| 세대전기계약방법 | 41    | 21         | Categorical| 2       | 703,084 |                       |             |                       |                       |                     |
| k_85㎡~135㎡이하 | 34    | 22         | Numeric    | 244     | 695,737 | 167                   | 248         | 63                    | 0                     | 1,500               |
| 좌표X         | 49    | 23         | Numeric    | 740     | 695,785 | 127                   | 0.09        | 127                   | 127                   | 127                 |
| k_전체동수    | 24    | 24         | Numeric    | 41      | 696,577 | 14.79                 | 17.69       | 10                    | 1                     | 124                 |
| 층            | 9     | 25         | Numeric    | 73      | 0       | 8.87                  | 5.98        | 8                     | -4                    | 69                  |
| k_수정일자    | 38    | 26         | Date       | 742     | 695,737 | 2023-09-03T09:00:50.819451 | 163.909 days | 2023-09-25T07:04:29 | 2020-02-17T04:28:42 | 2023-09-26T12:46:39 |
| 관리비 업로드  | 48    | 27         | Categorical| 2       | 695,698 |                       |             |                       |                       |                     |
| k_세대타입(분양형태) | 20 | 28      | Categorical| 3       | 695,698 |                       |             |                       |                       |                     |
| k_난방방식    | 23    | 29         | Categorical| 4       | 695,698 |                       |             |                       |                       |                     |
| 건축면적       | 43    | 30         | Numeric    | 454     | 695,817 | 190,762               | 1,736,524   | 1,624                 | 0                     | 3.16e+7             |
| k_전체세대수  | 25    | 31         | Numeric    | 520     | 695,698 | 1,184                 | 1,190       | 768                   | 59                    | 9,510               |
| k_전용면적별세대현황(60㎡~85㎡이하) | 33 | 32 | Numeric | 386 | 695,737 | 477 | 727 | 256 | 0 | 5,132 |
| 중개사소재지   | 15    | 33         | Categorical| 575     | 0       |                       |             |                       |                       |                     |
| k_전용면적별세대현황(60㎡이하) | 32 | 34 | Numeric | 347 | 695,737 | 479 | 760 | 226 | 0 | 4,975 |
| 거래유형       | 14    | 35         | Categorical| 3       | 0       |                       |             |                       |                       |                     |
| 본번          | 3     | 36         | Numeric    | 1,522   | 62      | 565                   | 516         | 470                   | 0                     | 4,969               |
| 단지신청일     | 51    | 37         | Date       | 258     | 695,746 | 2013-08-02T02:03:17.947906 | 495.16 days | 2013-03-07T09:46:39 | 2013-03-07T09:46:12 | 2023-08-04T15:31:27 |
| 기타/의무/임대/임의=1/2/3/4 | 45 | 38 | Categorical | 4 | 695,698 | | | | | |
| k_관리방식    | 21    | 39         | Categorical| 3       | 695,698 |                       |             |                       |                       |                     |
| 사용허가여부    | 47    | 40         | Categorical| 1       | 695,698 |                       |             |                       |                       |                     |
| 청소비관리형태  | 42    | 41         | Categorical| 4       | 696,996 |                       |             |                       |                       |                     |
| k_단지분류(아파트,주상복합등등) | 16 | 42 | Categorical | 5 | 696,619 | | | | | |
| 단지승인일     | 46    | 43         | Date       | 734     | 696,286 | 2015-05-17T09:25:31.234696 | 1330.404 days | 2015-10-16T10:47:13 | 1982-09-18T00:00:00 | 2023-08-04T15:39:34 |
| 경비비관리형태  | 40    | 44         | Categorical| 4       | 696,847 |                       |             |                       |                       |                     |
| 등기신청일자    | 13    | 45         | Date       | 181     | 883,084 | 2023-06-16            | 57.27 days  | 2023-06-22            | 2023-01-02            | 2023-09-22          |
| k_등록일자    | 37    | 46         | Date       | 125     | 886,218 | 2018-09-28T18:58:16.806901 | 570.066 days | 2018-06-25T21:14:05 | 2017-02-01T10:49:21 | 2023-05-12T10:09:44 |
| 단지소개기존clob | 19 | 47 | Numeric | 94 | 840,380 | 542 | 753 | 144 | 1 | 2,888 |
| 해제사유발생일  | 12    | 48         | Date       | 986     | 890,216 | 2021-07-04            | 368.27 days | 2021-03-08            | 2020-02-21            | 2023-09-26          |
| 계약일         | 8     | 49         | Numeric    | 31      | 0       | 15.80                 | 8.72        | 16                    | 1                     | 31                  |
| k_135㎡초과    | 35    | 50         | Numeric    | 2       | 894,794 | 70                    | 0           | 70                    | 70                    | 70                  |
| 부번          | 4     | 51         | Numeric    | 328     | 62      | 5.96                  | 46.86       | 0                     | 0                     | 2,837               |

#### Excess zeros
Excess zeros 현상이 5개의 변수에서 나타나서 Drop 처리되었습니다.  

| Feature Name                    | Index | Var Type | Unique | Missing | Mean    | Std Dev  | Median | Min | Max      |
|---------------------------------|-------|----------|--------|---------|---------|----------|--------|-----|----------|
| k_85㎡~135㎡이하                | 34    | Numeric  | 244    | 695,737 | 167     | 248      | 63     | 0   | 1,500    |
| 건축면적                         | 43    | Numeric  | 454    | 695,817 | 190,762 | 1,736,524| 1,624  | 0   | 31,600,000 |
| k_전용면적별세대현황(60㎡~85㎡이하) | 33    | Numeric  | 386    | 695,737 | 477     | 727      | 256    | 0   | 5,132    |
| k_전용면적별세대현황(60㎡이하)     | 32    | Numeric  | 347    | 695,737 | 479     | 760      | 226    | 0   | 4,975    |
| 부번                             | 4     | Numeric  | 328    | 62      | 5.96    | 46.86    | 0      | 0   | 2,837    |


#### Outliers
Outliers 현상이 15개의 변수에서 나타나서. Drop 처리 되었습니다.  

| Feature Name                    | Index | Var Type | Unique | Missing | Mean    | Std Dev  | Median | Min  | Max        |
|---------------------------------|-------|----------|--------|---------|---------|----------|--------|------|------------|
| target                          | 52    | Target   | 12,875 | 0       | 57,963  | 46,348   | 44,750 | 500  | 1,450,000  |
| 전용면적(㎡)                    | 6     | Numeric  | 14,218 | 0       | 77.16   | 29.38    | 81.86  | 10.02| 424        |
| k_연면적                        | 29    | Numeric  | 733    | 695,698 | 161,461 | 183,831  | 101,633| 0    | 9,591,851  |
| k_주거전용면적                  | 30    | Numeric  | 738    | 695,737 | 94,170  | 101,750  | 60,280 | 2,338| 734,781    |
| 주차대수                        | 44    | Numeric  | 525    | 695,817 | 1,064   | 1,235    | 683    | 0    | 12,096     |
| k_관리비부과면적                | 31    | Numeric  | 734    | 695,698 | 120,667 | 128,828  | 78,125 | 0    | 969,877    |
| k_85㎡~135㎡이하                | 34    | Numeric  | 244    | 695,737 | 167     | 248      | 63     | 0    | 1,500      |
| k_전체동수                      | 24    | Numeric  | 41     | 696,577 | 14.79   | 17.69    | 10     | 1    | 124        |
| 층                              | 9     | Numeric  | 73     | 0       | 8.87    | 5.98     | 8      | -4   | 69         |
| 건축면적                         | 43    | Numeric  | 454    | 695,817 | 190,762 | 1,736,524| 1,624  | 0    | 31,600,000 |
| k_전체세대수                    | 25    | Numeric  | 520    | 695,698 | 1,184   | 1,190    | 768    | 59   | 9,510      |
| k_전용면적별세대현황(60㎡~85㎡이하) | 33    | Numeric  | 386    | 695,737 | 477     | 727      | 256    | 0    | 5,132      |
| k_전용면적별세대현황(60㎡이하)     | 32    | Numeric  | 347    | 695,737 | 479     | 760      | 226    | 0    | 4,975      |
| 본번                             | 3     | Numeric  | 1,522  | 62      | 565     | 516      | 470    | 0    | 4,969      |
| 부번                             | 4     | Numeric  | 328    | 62      | 5.96    | 46.86    | 0      | 0    | 2,837      |


#### Disguised missing values
Disguised missing values 현상이 나타나지 않았습니다.  

#### Inliers
Inliers 발생 여부를 판단하기에 해당 도메인 지식이 없어 처리하지 않았습니다.

#### Target leakage
첫번째 EDA과정에서는 Target leakage 현상이 나타나지 않는 것으로 판단했습니다. 


### 4.3. Feature engineering
다음의 5가지의 방법을 고려했습니다.
- One-Hot Encoding
- Missing Values Imputed
- Smooth Ridit Transform
- Binning of numerical variables
- Matrix of char-grams occurrences using tfidf

#### One-Hot Encoding
이 변환기는 binary one-hot (aka one-of-K) 인코딩을 수행합니다. 피쳐가 취할 수 있는 가능한 문자열 값 각각에 대해 한개 부울 값 피쳐가 구성됩니다. 
이 인코딩은 많은 추정기, 특히 선형 모델 및 SVM에 범주형 데이터를 공급하는 데 필요합니다.

#### Missing Values Imputed
숫자형 피쳐의 경우 결측치를 중앙값(V2)으로 대치합니다.
숫자형 피쳐의 결측치를 중앙값으로 대치하고 indicator 변수를 생성하여 대치된 레코드를 식별합니다. np.partition 기반의 빠른 중앙값 알고리즘이 구현되어 중앙값 피쳐 값을 계산합니다.

#### Smooth Ridit Transform
숫자형 피쳐의 경우 백분위수 순위를 기반으로 한 Ridit 점수로 변환합니다. 백분위수 점수는 -1과 1 사이의 간격으로 추가로 조정됩니다. 바이너리 피쳐 및 날짜/시간 파생 피쳐 건너뛰도록 변환기를 구성할 수 있습니다. 희소성이 sparsity_threshold보다 높으면 데이터가 중앙값에 집중되고 출력은 희소 행렬이 됩니다.
Ridit 변환은 Bross(1958)의 RIDIT 채점 방법을 확장한 것으로, injury categories와 같이 순서가 지정되어 있지만 간격 척도가 아닌 데이터에 대해 Ridit 분석을 사용할 것을 제안합니다. Bross(1958) RIDIT의 절차는 다음과 같습니다. 동일한 범주(예: injury)를 가진 참조 모집단에서 각 범주에 대한 "ridit" 또는 점수를 결정합니다. 이 범주 점수는 참조 모집단에 있는 항목의 백분위수 순위이며 모든 하위 범주에 있는 항목 수에 주제 범주에 있는 항목 수의 1/2을 더한 값을 모두 모집단 크기로 나눈 값과 같습니다. 정의에 따르면 기준 모집단에 대해 계산된 Bross Ridit의 평균은 항상 0.5입니다.

#### Binning of numerical variables
비닝 작업은 숫자형 피쳐를 균일하지 않은 형태로 변환하는 전처리 방법입니다.
이는 우수한 예측 정확도와 인간이 이해할 수 있는 통찰력을 모두 제공하는 보다 강력한 선형 모델을 구축하는 데 유용한 도구입니다. 선형 모델을 사용하여 포착하기 어려운 비선형 관계가 포함된 데이터에 대한 비닝 전략을 수동으로 작성하는 프로세스를 자동화하는 것을 목표로 합니다.
빈의 경계는 각 입력 특성에 대해 개별적으로 훈련된 의사결정 트리에 의해 결정됩니다. 따라서 타겟 변수에 대한 정보는 비닝 과정에서 활용됩니다. 특히, 트리 노드에서 사용되는 임계값은 빈 경계로 사용됩니다. 기본적으로 xgboost 모델은 의사결정 트리에서 사용할 타겟 변수를 근사화하는 데 사용됩니다.

#### Matrix of char-grams occurrences using tfidf
문서 용어 행렬(Document-Term Matrix)은 텍스트 피쳐에서 숫자형 피쳐(Numeric Features)을 생성하는 방법입니다. 이러한 숫자형 피쳐은 원본 텍스트 열에 있는 용어(즉, 토큰)를 기반으로 생성됩니다.

숫자형 피쳐을 생성하는 방법을 결정하는 방법은 다양합니다.

여러 매개변수가 텍스트 특징에서 생성된 용어에 영향을 미칩니다. 예를 들어, 텍스트는 문자(Character, 예: 'cat'은 'c', 'a', 't)의 세 용어가 됨) 또는 단어(Word, 예: 'cat dog'는 'cat'라는 두 용어가 됨)로 구분할 수 있습니다. n개의 연속된 단어 또는 문자 그룹을 기반으로 용어를 생성하는 단어(Word) 또는 문자(Character) n-그램(n-gram)을 사용할 수도 있습니다(예를 들어 'cat' 문자를 기반으로 하는 2-gram은 'ca' 및 'at'이라는 용어를 반환합니다).

의미가 거의 없는 일반적인 용어를 제외할 수 있습니다. 'the', 'a' 및 'is'와 같은 단어는 빈도가 높기 때문에 문서 용어 행렬에서 큰 가중치를 부여받을 수 있지만 실제로 예측 열을 생성하는 데는 도움이 되지 않습니다. 이러한 용어는 일반적으로 'stop_words'라고 하며 stop_words를 True로 설정하여 행렬을 생성할 때 고려 대상에서 제거할 수 있습니다. 빈도에 따라 단어를 제거할 수도 있습니다. 단어가 너무 자주 나타나는 경우(max_df 이상) 또는 자주 나타나지 않는 경우(min_df 미만).

빈도가 높고 가치가 낮은 용어의 영향을 완화하는 또 다른 방법은 용어에 가중치를 부여하여 용어의 영향을 줄이는 것입니다. 가중치 부여 방법에는 여러 가지가 있습니다. 매우 일반적인 방법은 tf-idf 변환(Term Frequency-Inverse Document Frequency)을 사용하는 것입니다. 이 방법은 용어가 모든 행에 걸쳐 나타나는 빈도에 대해 총 행 수 로그의 역수로 텍스트 열의 용어 빈도에 가중치를 둡니다. 예를 들어, 10개 행 데이터세트에서 해당 용어가 특정 행에 3번 나타나고 총 10개 행 중 2개 행에 있는 경우 해당 특정 행에 대해 3 * log(10 / 2) 값이 지정됩니다.

#### Feature Selection
변수 선택시에는 합리성을 기준으로 예를 들어, 대부분의 값이 전부 1로 구성된 피쳐, 대부분의 값이 전부 0로 구성된 피쳐, 중복된 피쳐 또는 결측치가 많은 피쳐 같이 정보가 낮다고 판단하는 피쳐둘을 제외하는 것이 좋습니다. 

Feature Association Metrix는 그래프상에 보이는 색상의 불투명도(즉, num/cat, num/num, cat/cat)로 시각적으로 표시되는 숫자 및 범주형 피쳐 쌍 간의 연관 강도에 대한 정보를 제공합니다. 여기서 밝은 음영은 약한 연관을 나타내고 반대의 경우도 마찬가지입니다. 반대) 및 피쳐 클러스터. 매트릭스에서 색상으로 표시되는 피쳐군인 클러스터는 연관 구조에 따라 그룹으로 분할된 피쳐입니다.
Feature Association Metrix의 주목할만한 이점 중 일부는 다음과 같습니다.
- 데이터 내 연관성의 강도와 성격을 이해합니다.
- 피쳐쌍별로 연관 클러스터의 패밀리를 감지합니다.
- 모델을 구축하기 전에 연관성이 높은 특징의 클러스터를 식별합니다.

![Feature Association Metrix](./images/Feature-Association-Metrix.png)

또한 타겟 피쳐와의 상관 관계도 고려하여야 합니다. 이 때 사용되는 것이 Feature Importance 입니다.

![Feature Importance](./images/Feature-Importance.png)

##### Input Features
이러한 과정을 통해서 최종 선택된 입력 피쳐는 다음과 같습니다.

| Feature Name   | Var Type    | Unique | Missing | Mean     | Std Dev  | Median   | Min     | Max       | Target Leakage |
|----------------|-------------|--------|---------|----------|----------|----------|---------|-----------|----------------|
| 시군구         | Text        | 338    | 0       | N/A      | N/A      | N/A      | N/A     | N/A       | N/A            |
| 번지           | Categorical | 6,555  | 184     | N/A      | N/A      | N/A      | N/A     | N/A       | Low            |
| 아파트명       | Text        | 6,522  | 1,712   | N/A      | N/A      | N/A      | N/A     | N/A       | N/A            |
| 전용면적(㎡)   | Numeric     | 14,218 | 0       | 77.16    | 29.38    | 81.86    | 10.02   | 424.32    | Low            |
| 계약년월       | Numeric     | 198    | 0       | 201,475.9| 418.81   | 201,507.0| 200,701.0| 202,306.0 | Low            |
| 건축년도       | Numeric     | 60     | 0       | 1998.76  | 9.33     | 2000.0   | 1961.0  | 2023.0    | Low            |
| target         | Numeric     | 12,875 | 0       | 57,962.59| 46,347.64| 44,750.0 | 500.0   | 1,450,000.0| N/A            |

각 피쳐에 적용된 전처리는 다음과 같습니다.
| Feature Name   | Var Type    | Missing Count | Missing Percentage | Imputation Name       | Imputation Description                   |
|----------------|-------------|---------------|--------------------|-----------------------|------------------------------------------|
| 시군구         | Text    | 338           | 0                  | Document-Term Matrix Encoding      |      |
| 번지           | Categorical | 147           | 0                  | One-Hot Encoding      | Missing indicator treated as feature     |
| 아파트명         | Text    | 6522           | 0                  | Document-Term Matrix Encoding      |      |
| 전용면적(㎡)   | Numeric     | 0             | 0                  | Missing Values Imputed| Imputed value: 81.89                     |
| 계약년월       | Numeric     | 0             | 0                  | Missing Values Imputed| Imputed value: 201507                    |
| 건축년도       | Numeric     | 0             | 0                  | Missing Values Imputed| Imputed value: 2000                      |

##### Target Feature
EDA 과정에서 타겟 피쳐는 'target'이고 아래 그래프를 보면 편향되어 있는 것을 볼 수 있습니다. 피쳐 엔지니어링 과정에서 타겟 값을 표준화하거나 정규화하는 방법을 고려했습니다.

![Feature Target Histogram](./images/Feature-target-Histogram.png)


## 5. Modeling

### 5.1. Model Selection

#### 모델 검증 안정성
예측할 수 있는 데이터세트에서 패턴을 찾으려면 알고리즘은 먼저 과거 사례(일반적으로 예측하려는 출력 변수가 포함된 과거 데이터세트)에서 학습해야 합니다. 그러나 모델이 훈련 데이터에 대해 너무 밀접하게 훈련되면 과적합될 수 있습니다. 과대적합은 모델이 훈련 데이터에 너무 잘 맞아서 샘플 외부 데이터(모델을 훈련하는 데 사용되지 않은 데이터)에서 제대로 수행되지 않을 때 발생하는 모델링 오류입니다. 과적합은 일반적으로 모델이 포착하려는 기본 추세보다는 훈련 데이터의 특이성과 무작위 노이즈를 설명하는 지나치게 복잡한 모델을 생성합니다. 과적합을 방지하기 위한 가장 좋은 방법은 샘플 외부 데이터에서 모델 성능을 평가하는 것입니다. 모델이 표본 내 데이터(훈련 데이터)에서는 매우 잘 수행되지만 표본 외부 데이터에서는 성능이 좋지 않은 경우 이는 모델이 과적합되었음을 나타낼 수 있습니다.

이를 위해 표준 모델링 기술을 사용하여 모델 성능을 검증하고 과적합이 발생하지 않도록 합니다. 모델 성능의 샘플 외 안정성을 테스트하기 위해 강력한 모델 k-겹 교차 검증 프레임워크를 사용했습니다. 교차 검증 분할 외에도 홀드아웃 샘플을 사용하여 샘플 외 모델 성능을 추가로 테스트하고 모델이 과적합되지 않았는지 확인합니다.
과적합이 발생하지 않도록 개발 중에 다음 절차가 사용되었습니다.
학습 데이터의 20%를 홀드아웃 데이터 세트로 따로 보관했습니다. 이 데이터세트는 학습 과정 전반에 걸쳐 다루지 않은 데이터에 대해 최종 모델이 잘 작동하는지 확인하는 데 사용됩니다.
추가 모델 검증을 위해 나머지 데이터는 5개의 교차 검증 파티션으로 나뉩니다. 대규모 데이터세트로 작업할 때 발생하는 오버헤드를 보상하기 위해 먼저 데이터의 작은 부분에 대해 모델을 학습하고 하나의 교차 검증 접기만 사용하여 모델 성능을 평가합니다.
다음 그림은 CV 프로세스를 요약한 것입니다. 여기서 파란색은 훈련에 사용할 수 있는 데이터의 80%를 나타내며, 교차 검증을 위해 5겹으로 나뉘고 빨간색은 홀드아웃 샘플을 나타냅니다.

![Data Partitioning](./images/data-partitioning.png)

#### 데이터 분할 방법론
- Main : 데이터 파티션은 무작위 샘플링을 통해 선택되었습니다.
- Additional : UpStage AI Lab 멘토링을 통해 얻은 인사이트로 target 값의 시계열적 경향도 학습하기 위해 무작위 샘플링을 하지 않는 실험도 추가적으로 진행했습니다. 

#### Selected Models
DataRobot과 활용하여 Model Selection 을 수행한 결과를 바탕으로 Leaderboard의 상위권에 속한 다음과 같은 3개의 머신러닝 모델을 선택했습니다.

1. eXtreme Gradient Boosted Trees Regressor
2. Keras Slim Residual Network Regressor
3. Light Gradient Boosted Trees Regressor

![DataRobot Learderboard result](./images/DataRobot-Learderboard-result.png)

### 5.2. eXtreme Gradient Boosted Trees Regressor(DataRobot)

#### Modeling Descriptions 
Gradient Boosting Machines(또는 Generalized Boosted Models, 약어 'GBM'에 대한 설명을 요청한 사람에 따라 다름)는 매우 정확한 예측 모델을 피팅하기 위한 고급 알고리즘입니다. GBM은 최근 여러 예측 모델링 대회에서 우승했으며 많은 데이터 과학자들에 의해 가장 다재다능하고 유용한 예측 모델링 알고리즘으로 간주됩니다. GBM은 전처리가 거의 필요하지 않고 누락된 데이터를 우아하게 처리하며 편향과 분산 사이의 균형을 잘 유지하고 일반적으로 복잡한 상호 작용 용어를 찾을 수 있으므로 예측 모델의 유용한 "스위스 군용 칼"이 됩니다.
GBM은 임의의 손실 함수를 처리하는 Freund와 Schapire의 adaboost 알고리즘(1995)을 일반화한 것입니다. 이는 개별 의사결정 트리를 입력 데이터의 무작위 재샘플링에 적합하다는 점에서 개념상 랜덤 포레스트와 매우 유사합니다. 여기서 각 트리는 데이터 세트 행의 부트스트랩 샘플과 N이 임의로 선택된 열(여기서 N은 구성 가능)의 부트스트랩 샘플을 확인합니다. 모델의 매개변수. GBM은 단일 주요 측면에서 랜덤 포레스트와 다릅니다. 즉, GBM은 트리를 독립적으로 맞추는 대신 각 연속 트리를 결합된 모든 이전 트리의 잔차 오류에 맞춥니다. 이는 모델이 예측하기 가장 어려운(따라서 수정하는 데 가장 유용한) 샘플에 각 반복을 집중하기 때문에 유리합니다.
반복적인 특성으로 인해 GBM은 충분한 반복이 주어지면 학습 데이터에 과적합되는 것이 거의 보장됩니다. 따라서 알고리즘의 2가지 중요한 매개변수는 학습률(또는 모델이 데이터에 맞는 속도)과 모델이 적합하도록 허용되는 트리 수입니다. 이 2가지 매개변수 중 하나를 조정하는 것이 중요하며 올바르게 완료되면 GBM은 훈련 데이터에서 과적합이 시작되는 정확한 지점을 찾고 해당 지점 이전에 한 번의 반복을 중지할 수 있습니다. 이러한 방식으로 GBM은 일반적으로 훈련 세트에서 정보의 마지막 비트까지 모두 짜내고 과적합 없이 가능한 최고 정확도의 모델을 생성할 수 있습니다.
XGBoost(Extreme Gradient Boosting)는 수많은 Kaggle 대회에서 우승한 GBM의 매우 효율적인 병렬 버전입니다. 기본 알고리즘은 R 또는 Python의 GBM과 매우 유사하지만 더 빠른 런타임과 더 높은 예측 정확도를 위해 크게 최적화되고 조정되었습니다.
- 손실 함수: XGBoost 회귀 분석기는 기본적으로 최소 제곱 손실을 사용하지만 0이 팽창된 양수 분포에 대한 트위디 손실, 카운트 문제에 대한 포아송 손실, 오른쪽으로 편향된 양수 분포에 대한 감마 손실을 사용할 수도 있습니다.
- 그리드 검색 지원:
이 그리드 검색이 지원됩니다. 학습 중에 그리드 검색을 실행하여 최상의 성능을 제공하는 최적의 XGBoost 매개변수 값을 추정합니다(구성된 손실 함수로 평가). 그리드 검색은 학습 데이터 내에서 70/30 학습/테스트 분할로 실행됩니다. 예상 점수는 학습 데이터 분할의 30%를 사용합니다. 그리드 검색이 완료되고 최상의 튜닝 매개변수가 발견되면 최종 모델은 100% 훈련 데이터로 재훈련됩니다. 최종 모델의 검증 점수는 그리드 검색의 검증 점수와 다릅니다.
- Early Stopping 지원:
Early Stopping Extreme Gradient Boosting 모델은 또한 조기 중지를 사용하여 최적의 트리 수를 결정합니다. 조기 중지는 Early Stopping은 XGB 모델에 사용할 트리 수를 결정하는 방법입니다. 학습 데이터는 학습 세트와 테스트 세트로 분할되며, 각 반복마다 테스트 세트에서 모델의 점수가 매겨집니다. 200회 반복 동안 테스트 세트 성능이 저하되면 학습 절차가 중지되고 모델은 지금까지 본 최고의 트리에서 피팅을 반환합니다. 이 접근 방식은 모델이 과적합되고 추가 트리가 더 높은 정확도를 가져오지 않는다는 것이 분명한 지점을 지나 계속 진행하지 않음으로써 시간을 절약합니다.

![DataRobot XGB](./images/DataRobot-XGB.png)

#### Modeling Process

##### Hiperparameters
| Name                    | Description                                                                                                                                                                                                                       | Best Searched |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| base_margin_initialize  | True일 경우, 인터셉트가 타겟의 로그 오즈로 초기화됩니다.                                                                                                                                                                           | True          |
| colsample_bylevel       | 각 트리의 분할 전에 특징의 하위 샘플입니다.                                                                                                                                                                                       | 1.0           |
| colsample_bytree        | 각 트리를 구성할 때 열의 하위 샘플 비율입니다. 기본적으로 XGBoost 클래스의 colsample_bytree 값은 1.0입니다. 그러나 학습 데이터에 따라 DataRobot은 이 매개변수에 대해 다른 초기 값을 선택할 수 있습니다.                                                            | 0.3           |
| interval                | 조기 중지를 위한 간격입니다. 모델이 "smooth_interval" 반복에 도달하면, 조기 중지 논리는 연속적으로 "interval" 반복에서 오류가 증가하는지 확인합니다. 예를 들어, smooth_interval=200 및 interval=10인 경우 XGBoost는 최소 200회 반복을 실행하고, 마지막 200회 반복의 이동 평균 손실이 연속적으로 10회 증가하면 조기 중지를 실행합니다. 값이 클수록 XGBoost가 더 오래 실행됩니다. | 10            |
| learning_rate           | 각 트리의 기여도를 learning_rate로 축소합니다. learning_rate (lr)와 n_estimators (n) 간에는 절충이 있습니다.                                                                                                                          | 0.05          |
| loss                    | 최적화할 손실 함수입니다. 'ls'는 최소 제곱 회귀를 의미합니다.                                                                                                                                                                     | gamma         |
| max_bin                 | tree_method가 'hist'로 설정된 경우 사용됩니다. 연속 특징을 버킷으로 분할하기 위한 최대 이산화 빈 수입니다. 이 수를 늘리면 분할의 최적성이 향상되지만 계산 시간이 더 길어집니다.                                                                                           | 256           |
| max_delta_step          | 각 트리의 가중치 추정에 허용되는 최대 델타 단계입니다. 값이 0으로 설정되면 제약이 없습니다. 양의 값으로 설정하면 업데이트 단계가 더 보수적이 됩니다. 이 매개변수는 일반적으로 필요하지 않지만, 클래스가 매우 불균형한 로지스틱 회귀에서 도움이 될 수 있습니다. 1-10의 값으로 설정하면 델타 단계 업데이트를 제어할 수 있습니다.                        | 0.0           |
| max_depth               | 개별 회귀 추정기의 최대 깊이입니다. 최대 깊이는 트리의 노드 수를 제한합니다. 최적의 성능을 위해 이 매개변수를 조정하십시오. 최적의 값은 입력 변수의 상호 작용에 따라 다릅니다. 트리가 깊을수록 모델이 더 많은 변수 상호 작용을 캡처할 수 있습니다. 부모 모델보다 더 큰 샘플 크기를 가진 고정 모델의 경우 유사한 정확성을 유지하기 위해 max_depth 값이 증가합니다.                   | 7             |
| min_child_weight        | 자식 노드에 필요한 인스턴스 가중치(hessian)의 최소 합입니다. 트리 분할 단계가 인스턴스 가중치 합이 min_child_weight보다 작은 리프 노드를 생성하면 추가 분할을 포기합니다. 선형 회귀 모드에서는 단순히 각 노드에 필요한 최소 인스턴스 수에 해당합니다. 값이 클수록 알고리즘이 더 보수적이 됩니다.                                   | 1.0           |
| min_split_loss          | 리프 노드의 추가 분할을 위해 필요한 최소 손실 감소량입니다. 값이 클수록 알고리즘이 더 보수적이 됩니다.                                                                                                                                                                     | 0.01          |
| missing_value           | 누락된 값으로 처리해야 하는 float 값입니다. mono_up 또는 mono_down이 설정된 경우 누락된 값은 -9999.0으로 설정됩니다.                                                                                                                            | 0.0           |
| mono_down               | 타겟에 대해 단조 감소 관계를 갖는 특징 집합을 정의하는 featurelist의 ID입니다.                                                                                                                                                         | no            |
| mono_up                 | 타겟에 대해 단조 증가 관계를 갖는 특징 집합을 정의하는 featurelist의 ID입니다.                                                                                                                                                         | no            |
| n_estimators            | 수행할 부스팅 단계 수입니다. 그라디언트 부스팅은 과적합에 상당히 강하므로 일반적으로 더 많은 수가 더 나은 성능을 제공합니다.                                                                                                                           | 3500          |
| num_parallel_tree       | 각 부스팅 단계에서 생성되는 병렬 트리 수입니다. 이 값이 1보다 크면 모델은 (num_parallel_tree * n_estimators) 트리를 가진 그라디언트 부스팅 랜덤 포레스트가 됩니다.                                                                                                       | 1             |
| random_state            | 난수 생성기에 사용되는 시드 값입니다.                                                                                                                                                                                                  | 1234          |
| reg_alpha               | 가중치에 대한 L1 정규화 항목입니다. 이 값을 늘리면 모델이 더 보수적이 됩니다.                                                                                                                                                                | 0.0           |
| reg_lambda              | 가중치에 대한 L2 정규화 항목입니다. 이 값을 늘리면 모델이 더 보수적이 됩니다.                                                                                                                                                                 | 1.0           |
| scale_pos_weight        | 양수 클래스의 예제에 대한 스케일링 인자입니다.                                                                                                                                                                                            | 1.0           |
| smooth_interval         | 조기 중지를 위한 이동 평균 간격입니다. 조기 중지를 결정하기 위해 손실의 마지막 n 간격이 평균화됩니다. 예를 들어, smooth_interval=200인 경우 XGBoost는 최소 200회 반복을 실행하고, 200회 반복 후 마지막 200회 반복의 이동 평균 손실을 사용하여 조기 중지를 결정합니다. 이는 손실 함수의 노이즈를 제거하는 데 도움이 됩니다. 값이 클수록 XGBoost가 더 오래 실행됩니다. | 200           |
| subsample               | 학습 인스턴스의 하위 샘플 비율입니다. 0.5로 설정하면 XGBoost가 트리를 성장시키기 위해 데이터 인스턴스의 절반을 무작위로 수집하여 과적합을 방지합니다.                                                                                                                                       | 1.0           |
| tree_method             | 트리 구성 알고리즘을 선택합니다. 'auto': 더 빠른 알고리즘을 선택하기 위한 휴리스틱입니다. 중소 데이터셋(<4M 행)에는 정확한 그리디 알고리즘을 사용합니다. 대형 데이터셋(>=4M 행)에는 근사 알고리즘을 사용합니다. 'exact': 정확한 그리디 알고리즘. 'approx': 스케치 및 히스토그램을 사용한 근사 그리디 알고리즘. 'hist': 히스토그램 최적화된 근사 그리디 알고리즘. 성능 개선(예: 빈 캐싱)을 사용합니다. | hist          |
| tweedie_p               | Tweedie 분포의 전력 매개변수입니다. Tweedie 손실이 사용되는 경우에만 적용됩니다.                                                                                                                                                                  | 1.5           |

##### Feature Impact

![DataRobot XGB - FeatureImpact](./images/DataRobot-XGB-Feature-Impact.png)

##### Word Cloud

![DataRobot XGB - Word Cloud](./images/DataRobot-XGB-WordCloud.png)


### 5.3. Keras Slim Residual Network Regressor(DataRobot)

#### Modeling Descriptions 
신경망은 생물학적 신경망(동물의 중추신경계, 특히 뇌)에서 영감을 받은 모델군으로, 일반적으로 알려지지 않은 수많은 입력에 의존할 수 있는 피쳐를 추정하거나 근사화하는 데 사용됩니다. 신경망은 일반적으로 서로 메시지를 교환하는 상호 연결된 "뉴런"의 시스템으로 표시됩니다. 연결에는 경험에 따라 조정될 수 있는 숫자 가중치가 있어 신경망이 입력에 적응하고 학습할 수 있게 됩니다.
Hidden Layer가 없는 신경망은 사용된 활성화 함수(예: 시그모이드 대 선형 활성화)에 따라 로지스틱 또는 선형 회귀 모델과 동일합니다. (입력 및 출력 계층 사이에) 활성화가 뒤따르는 일련의 뉴런인 "Hidden Layer"을 신경망에 추가하면 비선형성이 발생합니다. 이를 통해 모델은 특성 간의 비선형 관계를 학습할 수 있으며, 이는 단순한 선형 모델보다 훨씬 더 강력한 모델로 이어질 수 있습니다.
신경망은 옵티마이저와 역전파를 사용하여 학습합니다. 즉, 작은 배치의 데이터를 반복적으로 가져와 예측과 실제의 차이를 계산하고 레이어별로 가중치를 조금씩 조정하여 실제에 가까운 예측을 생성합니다.
이러한 형태의 모델링은 매우 유연하여 임의의 함수를 구성할 수 있지만 일반 회귀 모델보다 입력 데이터에 훨씬 더 민감하므로 Batch Normalization과 같은 특별한 방법이 필요합니다. 작업이 텍스트 데이터 내에서 상호 작용을 찾는 것과 관련된 경우 신경망 활용을 더욱 고려할 필요가 있습니다.
Keras는 딥 러닝 모델용 Tensorflow 프레임워크를 사용하여 신경망을 구축하기 위한 고급 라이브러리입니다. Keras는 최첨단 딥러닝 모델을 신속하게 통합할 수 있는 유연성을 제공합니다. Keras는 Sparse 데이터도 지원하는데, 이는 텍스트가 많은 데이터나 멀티 레벨의 범주형 데이터에 특히 중요할 수 있습니다.
Keras의 Python API 클래스는 여러 숨겨진 계층이 있는 표준 신경망 모델뿐만 아니라 Self-Normalizing Neural Networks(https://arxiv.org/abs/1706.02515에 설명됨) 및 Residual Connection(예: https://arxiv.org/abs/1712.09913에 설명되어 있음)도 지원합니다.
자체 정규화 신경망은 배치 정규화를 사용하지 않고도 기울기가 사라지거나 폭발하는 것을 방지하기 위해 매우 구체적인 입력 이니셜라이저와 "Scaled Exponential Linear Units"라는 특수 활성화 함수를 사용합니다.
Residual 네트워크에는 입력에서 출력으로의 직접 연결이 포함되어 있어 손실 함수를 원활하게 하고 네트워크를 더 효과적으로 최적화할 수 있습니다.

![DataRobot Keras Slim RestNet](./images/DataRobot-Keras-Slim-RestNet.png)

#### Modeling Process

##### Neural Network

![DataRobot Keras Slim RestNet - Neural Network](./images/DataRobot-Keras-Slim-RestNet-NeuralNetwork.png)

##### Hiperparameters

| Name                             | Description                                                                                                                                                                                                                                                         | Best Searched |
|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| batch_size                       | Keras 신경망은 SGD를 통해 미니 배치로 학습됩니다. 이 매개변수는 각 미니 배치에 고려할 행 수를 결정합니다. 값이 클수록 학습 속도가 빨라지지만 RAM을 더 많이 사용합니다. 값이 너무 높으면 모델이 수렴하는 데 문제가 생길 수 있습니다. 'auto'로 설정된 경우 데이터셋의 행 수를 기준으로 배치 크기를 계산하는 휴리스틱을 사용합니다. 기본적으로 auto 휴리스틱은 데이터셋의 64행마다 배치 크기를 1씩 증가시키며, 가장 가까운 2의 거듭제곱으로 반올림됩니다. 그러나 이는 데이터셋마다 다르며, DataRobot은 데이터셋별로 최적의 배치 크기 휴리스틱을 결정합니다. 가능한 값: {'intgrid': [1, 131072], 'select': ['auto']} | 16384         |
| double_batch_size                | 1일 경우 배치 크기가 매 에포크마다 두 배로 늘어납니다(최대 값은 max_batch_size). 가능한 값: [0, 1]                                                                                                                                                                                             | 0             |
| dropout_type                     | 일반 드롭아웃 또는 알파 드롭아웃을 사용할지 여부를 지정합니다. 숨겨진 층과 출력 층 모두에 적용됩니다. 'normal' 또는 'alpha'를 사용합니다. 가능한 값: ['normal', 'alpha']                                                                                                                                                             | normal        |
| early_stopping                   | 개선이 없는 에포크 수 이후에 학습이 중지됩니다. early_stopping = 0이면 조기 중지가 없습니다. early_stopping > 0이면 그리드 검색 테스트 세트에서 검증 손실을 확인하고, 해당 손실이 연속으로 early_stopping 횟수만큼 증가하면 에포크에 도달하기 전에 종료됩니다. stochastic_weight_average_epochs와 early_stopping 중 하나만 0이 아닐 수 있습니다. 가능한 값: [0, 1000] | 3             |
| epochs                           | 데이터를 통과하는 횟수입니다. 1 에포크는 모델이 학습 데이터를 정확히 한 번씩 고려함을 의미합니다. 확률적 경사 하강법이 작동하는 방식 때문에, 손실이 계산되어 가중치를 업데이트하는 방법을 결정할 때, 추정된 값과 실제 타겟 값의 차이에 따라 가중치가 소량(학습률에 따라) 변경됩니다. 이 때문에 여러 번 또는 많은 데이터를 통과하는 것이 좋습니다. 가능한 값: [1, 1000] | 3             |
| hidden_activation                | 숨겨진 층에 사용할 활성화 함수입니다. "relu"와 "prelu"가 보통 좋은 선택입니다. units, hidden_dropout, hidden_batch_norm, hidden_l1 및 hidden_l2는 목록이며 층별로 변경될 수 있지만, hidden_activation은 모든 숨겨진 층에 동일하게 적용됩니다. 가능한 값: ['linear', 'sigmoid', 'hard_sigmoid', 'relu', 'elu', 'selu', 'tanh', 'softmax', 'softplus', 'softsign', 'exponential', 'swish', 'mish', 'thresholdedrelu', 'leakyrelu', 'prelu', 'cloglog', 'probit'] | prelu         |
| hidden_batch_norm                | 각 숨겨진 층을 배치 정규화할지 여부입니다. 이는 모델 수렴 속도를 높일 수 있습니다. hidden_batch_norm = 1과 hidden_dropout > 0을 동시에 설정할 때 주의하십시오. 1 = 배치 정규화 사용, 0 = 배치 정규화 사용 안 함. 모든 숨겨진 층에 적용됩니다. 가능한 값: [0, 1]                                                                                        | 0             |
| hidden_bias_initializer          | 숨겨진 층의 바이어스를 초기화하는 방법입니다. 모든 층에 사용됩니다. 가능한 값: ['zeros', 'ones', 'random_uniform', 'lecun_uniform', 'glorot_uniform', 'he_uniform', 'random_normal', 'lecun_normal', 'glorot_normal', 'he_normal', 'truncated_normal', 'VarianceScaling', 'orthogonal']                                                 | zeros         |
| hidden_dropout                   | 학습의 각 전방 패스에서 무작위로 활성화를 드롭할 비율을 나타냅니다. 이를 "드롭아웃"이라고 합니다. 이는 모델을 정규화하고 일반적으로 일반화를 향상시킵니다. 여기 제공된 부동 소수점 값은 각 층의 드롭아웃 수준을 결정하는 데 사용됩니다. (모든 숨겨진 층에 적용됨) 드롭아웃 없음은 0으로 설정합니다. 가능한 값: [0, 0.99]                                                                  | 0.0           |
| hidden_initializer               | 모델의 숨겨진 층을 초기화하는 방법입니다. 기본값으로 두는 것이 좋습니다. 가능한 값: ['zeros', 'ones', 'random_uniform', 'lecun_uniform', 'glorot_uniform', 'he_uniform', 'random_normal', 'lecun_normal', 'glorot_normal', 'he_normal', 'truncated_normal', 'VarianceScaling', 'orthogonal']                                    | he_uniform    |
| hidden_l1                        | 각 숨겨진 층에 사용할 L1 정규화입니다. 숨겨진 층으로 들어오는 변수를 선택하는 경향이 있습니다. L1 정규화를 사용하지 않으려면 0으로 설정합니다. 이는 손실 함수에서 l1(weights)에 적용되는 패널티 계수입니다. 모든 숨겨진 층에 적용됩니다. 가능한 값: [0, 1000000.0]                                                                                  | 0.0           |
| hidden_l2                        | 각 숨겨진 층에 사용할 L2 정규화입니다. 숨겨진 층으로 들어오는 계수를 축소하는 경향이 있습니다. L2 정규화를 사용하지 않으려면 0으로 설정합니다. 이는 손실 함수에서 l2(weights)에 적용되는 패널티 계수입니다. 모든 숨겨진 층에 적용됩니다. 가능한 값: [0, 1000000.0]                                                                                  | 0.0           |
| hidden_units                     | 네트워크의 숨겨진 층의 유닛 수입니다. 없는 경우 모델은 확률적 경사 하강법(SGD)을 통해 적합된 간단한 회귀 모델과 동등하며, 특징 간의 상호 작용을 찾지 않습니다. 여러 숨겨진 층을 위한 숨겨진 유닛 목록을 지정하십시오. 예: list(512, 256, 128)은 유닛 수가 감소하는 3개의 층을 나타냅니다. 숨겨진 층이 없는 모델을 적합시키려면 "list()"를 사용하십시오. 가능한 값: {'length': [0, 25], 'int': [1, 8192]} | [64]          |
| hidden_use_bias                  | 숨겨진 층에 바이어스 항을 사용할지 여부입니다. 모든 층에 적용됩니다. 1 = 바이어스 사용, 0 = 바이어스 사용 안 함. 모든 숨겨진 층에 적용됩니다. 가능한 값: [0, 1]                                                                                                                                                            | 1             |
| learning_rate                    | 최적화에 사용되는 학습률입니다. 낮은 학습률은 더 정확한 모델로 이어질 수 있지만 수렴하는 데 더 많은 에포크가 필요하며 국소 최소값에 더 민감합니다. 학습 스케줄을 사용하는 경우, learning_rate는 최대 학습률을 나타냅니다. 가능한 값: [1e-10, 1000]                                                                                                           | 0.015         |
| loss                             | 모델이 최적화하는 손실 함수입니다. 가능한 값: ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'logcosh', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'gamma', 'tweedie', 'quantile', 'cosine_proximity']   | gamma         |
| loss_quantile_level              | 분위 손실이 보정되어야 하는 분위 수준입니다. 기본값 0.5는 중앙값 최적화 모델과 동일합니다. 분위 손실을 사용하는 모델에만 적용됩니다. 가능한 값: [0.01, 0.99]                                                                                                                                                                         | 0.5           |
| max_batch_size                   | 배치 크기 증가로 인해 미니 배치가 너무 커지지 않도록 모델이 고려할 최대 배치 크기입니다. double_batch_size가 1로 설정된 경우에만 적용됩니다. 가능한 값: [1048, 131072]                                                                                                                                                           | 131072        |
| optimizer                        | 모델을 적합시키기 위해 사용할 SGD의 변형입니다. 'adam'을 사용하는 것이 좋습니다. 가능한 값: ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam', 'adabound']                                                                                                                               | adam          |
| output_activation                | 네트워크의 최종 출력 층에 대한 활성화입니다. 기본값으로 두는 것이 좋습니다. 가능한 값: ['linear', 'sigmoid', 'softsign', 'exponential', 'tanh', 'cloglog', 'softplus', 'probit', 'selu', 'elu']                                                                                                             | exponential   |
| output_batch_norm                | 출력 층을 배치 정규화할지 여부입니다. 이는 모델 수렴 속도를 높일 수 있습니다. 1 = 배치 정규화 사용, 0 = 배치 정규화 사용 안 함. 가능한 값: [0, 1]                                                                                                                                                                      | 0             |
| output_bias_initializer          | 출력 층의 바이어스를 초기화하는 방법입니다. 모든 층에 사용됩니다. "mean"으로 설정하면 타겟의 평균으로 바이어스를 초기화하여 수렴 속도를 높이는 데 도움이 될 수 있습니다. 가능한 값: ['zeros', 'ones', 'random_uniform', 'lecun_uniform', 'glorot_uniform', 'he_uniform', 'random_normal', 'lecun_normal', 'glorot_normal', 'he_normal', 'truncated_normal', 'VarianceScaling', 'orthogonal']  | mean          |
| output_initializer               | 네트워크의 최종 출력 층을 초기화하는 방법입니다. 기본값으로 두는 것이 좋습니다. 가능한 값: ['zeros', 'ones', 'random_uniform', 'lecun_uniform', 'glorot_uniform', 'he_uniform', 'random_normal', 'lecun_normal', 'glorot_normal', 'he_normal', 'truncated_normal', 'VarianceScaling', 'orthogonal']                                      | he_uniform    |
| output_l1                        | 출력 층에 사용할 L1 정규화입니다. 출력 층으로 들어오는 변수를 선택하는 경향이 있습니다. L1 정규화를 사용하지 않으려면 0으로 설정합니다. 이는 손실 함수에서 l1(weights)에 적용되는 패널티 계수입니다. 가능한 값: [0, 1000000.0]                                                                                          | 0.0           |
| output_l2                        | 출력 층에 사용할 L2 정규화입니다. 출력 층으로 들어오는 계수를 축소하는 경향이 있습니다. L2 정규화를 사용하지 않으려면 0으로 설정합니다. 이는 손실 함수에서 l2(weights)에 적용되는 패널티 계수입니다. 가능한 값: [0, 1000000.0]                                                                                          | 0.0           |
| pass_through_inputs              | pass_through_inputs가 1인 경우 입력을 출력 층에 직접 연결합니다. 이러한 추가 연결은 종종 스킵 연결이라고 합니다. 신경망이 스킵 연결을 활용하면 이를 "잔차 신경망"이라고 합니다. 가능한 값: [0, 1]                                                                                                                  | 1             |
| prediction_batch_size            | 예측을 위한 배치 크기입니다. 높은 설정은 예측 시 더 많은 RAM을 사용하지만 일반적으로 더 빠릅니다. 이 설정은 학습에는 전혀 영향을 미치지 않으며, 기본값으로 두는 것이 좋습니다. 전용 예측 서버에서 모델이 너무 많은 RAM을 사용하는 경우 이 매개변수를 낮은 값으로 조정해 볼 수 있습니다(예: 4096, 2048 또는 1024). 이는 배치 예측을 더 느리게 만들지만 RAM 사용량을 줄이는 데 도움이 됩니다. 모델의 정확성에는 영향을 미치지 않습니다. 또한 이 매개변수는 한 번에 1행씩 예측하는 경우에는 영향을 미치지 않습니다. 가능한 값: [1, 131072]                  | 8192          |
| random_seed                      | 네트워크를 구성할 때 시드 값을 사용하는 모든 작업(예: 드롭아웃, 초기 가중치, 초기 바이어스 등)에 사용할 무작위 시드입니다. 가능한 값: [0, 2147483646]                                                                                                                                                           | 42            |
| stochastic_weight_average_epochs | 네트워크 가중치를 평균화할 데이터 통과 횟수입니다. 0이면 SWA가 없습니다. 0보다 크면 SWA가 마지막 N=`stochastic_weight_average_epochs` 에포크에서 적용됩니다. 에포크 이하이어야 합니다. stochastic_weight_average_epochs와 early_stopping 중 하나만 0이 아닐 수 있습니다. 가능한 값: [0, 1000]                                    | 0             |
| training_schedule_curve          | 학습 스케줄의 지점 간 전환에 사용할 함수를 정의합니다. 일반적으로 학습 스케줄의 대부분을 낮은 학습률에서 수행해야 하는 경우 'exponential'이 잘 작동하고, 모든 학습률에서 동일한 시간을 소비해야 하는 경우 'linear'가 잘 작동합니다. 'cosine'은 높은 학습률과 낮은 학습률 모두에서 더 많은 시간을 할애해야 할 때 잘 작동합니다. 'cosine'은 기본적으로 워밍업 및 워밍다운 효과를 제공하면서 높은 학습률에서 더 많은 시간을 할애하므로 기본적으로 사용됩니다. 가능한 값: ['linear', 'exponential', 'cosine']             | cosine        |
| training_schedule_cycle_count    | 손실 함수를 최소화하기 위해 학습률을 더욱 낮추기 전에 학습 스케줄에서 수행할 사이클 수입니다. 가능한 값: [0, 1000]                                                                                                                                                                                      | 1             |
| training_schedule_cycle_scale    | 학습 스케줄에서 각 사이클의 규모를 정의합니다. 학습 스케줄에서 사용되는 최대 학습률은 learning_rate로 정의됩니다. 각 사이클에서 사용되는 최소 학습률은 training_schedule_cycle_scale * learning_rate로 정의됩니다. 각 사이클은 최소 학습률에서 시작하여 최대 학습률로 증가한 후 다시 최소 학습률로 감소합니다. 가능한 값: [0.0, 1.0]                        | 0.04          |
| training_schedule_cycle_warm_up_fraction | 사이클의 어느 부분이 최대 학습률로 도달하는 데 사용될지 정의합니다. 기본적으로 사이클은 최소 학습률에서 시작하여 사이클의 25% 지점에서 최대 학습률에 도달한 다음, 나머지 75% 동안 최소 학습률로 감소합니다. 가능한 값: [0.0, 1.0]                                                                                               | 0.25          |
| training_schedule_post_cycle_scale | "워밍 다운" 단계 동안 학습 스케줄의 규모를 정의합니다. 워밍 다운 단계에서 학습률을 더욱 낮춤으로써 모델이 손실 함수의 샤프한 최소값에 도달하도록 합니다. 기본적으로 이는 learning_rate의 0.2%로 설정됩니다. 가능한 값: [0.0, 1.0]                                                                                              | 0.002         |
| training_schedule_warm_down_fraction | 모든 사이클이 완료된 후 학습률을 더 낮추어 손실 지형의 현재 최소값에 최대한 도달하도록 세부 조정하기 위해 전체 반복 횟수의 얼마나 큰 비율을 할당할지 정의합니다. 기본적으로 이는 학습의 마지막 25%입니다. 즉, 모든 사이클은 첫 번째 75% 내에 완료됩니다. 가능한 값: [0.0, 1.0]                                                        | 0.25          |
| use_training_schedule            | 학습 스케줄을 사용할지 여부입니다. 이 기능이 활성화되면 학습률과, 최적화가 이를 지원하는 경우 모멘텀이 스케줄에 따라 조정됩니다. 학습률은 작게 시작하여 지정된 learning_rate로 증가하고, 나머지 학습 동안 더욱 낮아집니다. 이 과정은 training_schedule_cycle_count에 따라 반복되며, 나머지 반복의 warm_down_fraction에서만 학습률이 더욱 낮아집니다. 가능한 값: [0, 1] | 1             |


##### Training

![DataRobot Keras Slim RestNet - Training](./images/DataRobot-Keras-Slim-RestNet-Training.png)

##### Feature Impact

![DataRobot Keras Slim RestNet - FeatureImpact](./images/DataRobot-Keras-Slim-RestNet-Feature-Impact.png)

##### Word Cloud

![DataRobot Keras Slim RestNet - Word Cloud](./images/DataRobot-Keras-Slim-RestNet-WordCloud.png)

### 5.4. Light Gradient Boosted Trees Regressor(DataRobot)

#### Modeling Descriptions 

LightGBM은 그래디언트 부스팅 프레임워크입니다. 트리 기반 알고리즘을 사용하며 분산되고 효율적으로 설계되어 다음과 같은 이점을 제공합니다.
- 그라디언트 부스팅 머신
- 그리드 검색 지원
- Early Stopping 지원

![DataRobot LightGBM](./images/DataRobot-LightGBM.png)

#### Modeling Process

##### Hiperparameters

| Name                   | Description                                                                                                                                                                                                                                               | Best Searched |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| boost_from_average     | 레이블의 평균으로 초기 점수를 조정하여 더 빠른 수렴을 유도합니다.                                                                                                                                                                                           | True          |
| colsample_bytree       | 각 트리를 구성할 때 열의 하위 샘플 비율입니다. 기본적으로 LightGBM 클래스의 colsample_bytree 값은 1.0입니다. 그러나 학습 데이터에 따라 DataRobot은 이 매개변수에 대해 다른 초기 값을 선택할 수 있습니다.                                                                                            | 0.3           |
| early_stopping_rounds  | 검증 데이터의 메트릭 중 하나가 마지막 early_stopping_round 라운드에서 개선되지 않으면 학습을 중지합니다.                                                                                                                                                         | 200           |
| fair_c                 | Fair 손실을 위한 매개변수입니다.                                                                                                                                                                                                                             | 1.0           |
| huber_delta            | Huber 손실을 위한 매개변수입니다.                                                                                                                                                                                                                            | 1.0           |
| learning_rate          | 각 트리의 기여도를 learning_rate로 축소합니다. learning_rate (lr)와 n_estimators (n) 간에는 절충이 있습니다. dart에서는 정규화에도 영향을 미칩니다.                                                                                                                | 0.05          |
| max_bin                | 특징 값이 버킷에 들어갈 최대 bin 수입니다. 작은 bin은 학습 정확도를 줄일 수 있지만 일반적인 성능을 향상시킬 수 있습니다(과적합 처리). LightGBM은 max_bin에 따라 메모리를 자동으로 압축합니다. 예를 들어, max_bin=255이면 LightGBM은 특징 값에 uint8_t를 사용합니다.                                    | 255           |
| max_delta_step         | 최적화를 보호하기 위해 사용되는 매개변수입니다. 값이 클수록 증가가 더 보수적입니다.                                                                                                                                                                                 | 0.7           |
| max_depth              | 개별 회귀 추정기의 최대 깊이입니다. 최대 깊이는 트리의 노드 수를 제한합니다. 최적의 성능을 위해 이 매개변수를 조정하십시오. 최적의 값은 입력 변수의 상호 작용에 따라 다릅니다. 트리가 깊을수록 모델이 더 많은 변수 상호 작용을 캡처할 수 있습니다. 트리는 여전히 리프 단위로 성장합니다. <0은 제한이 없음을 의미합니다.                 | none          |
| min_child_samples      | 자식(리프)에 필요한 최소 데이터 수입니다.                                                                                                                                                                                     | 10            |
| min_child_weight       | 자식(리프)에 필요한 인스턴스 가중치(hessian)의 최소 합입니다.                                                                                                                                                                         | 5             |
| min_split_gain         | 트리의 리프 노드에서 추가 분할을 수행하기 위해 필요한 최소 손실 감소입니다.                                                                                                                                                                               | 0.0           |
| n_estimators           | 수행할 부스팅 단계 수입니다. 그라디언트 부스팅은 과적합에 상당히 강하므로 일반적으로 더 많은 수가 더 나은 성능을 제공합니다.                                                                                                                                             | 3500          |
| num_leaves             | 한 트리의 리프 수입니다.                                                                                                                                                                                                                                   | 64            |
| objective              | 최적화할 목적 함수입니다.                                                                                                                                                                                                                                   | gamma         |
| reg_alpha              | 가중치에 대한 L1 정규화 항목입니다.                                                                                                                                                                                                                         | 0.0           |
| reg_lambda             | 가중치에 대한 L2 정규화 항목입니다.                                                                                                                                                                                                                         | 0.0           |
| subsample              | 학습 인스턴스의 하위 샘플 비율입니다.                                                                                                                                                                                                                        | 1.0           |
| subsample_for_bin      | bin을 구성하기 위한 샘플 수입니다.                                                                                                                                                                                                                            | 50000         |
| subsample_freq         | 하위 샘플 빈도입니다. 'none'은 비활성화를 의미합니다.                                                                                                                                                                                                                   | 1             |
| tweedie_p              | Tweedie 손실을 위한 매개변수입니다.                                                                                                                                                                                                                          | 1.5           |


##### Feature Impact

![DataRobot Keras Slim RestNet - FeatureImpact](./images/DataRobot-LightGBM-Feature-Impact.png)

##### Word Cloud

![DataRobot Keras Slim RestNet - Word Cloud](./images/DataRobot-LightGBM-WordCloud.png)

### 5.5. PyTorch Residual Network Regressor(박석)

#### Modeling Descriptions 

##### PyTorch를 활용한 Residual Network 모델링 설명

PyTorch를 활용하여 Residual Network를 구현하고 학습하는 과정은 다음과 같습니다.

###### Step 1: 라이브러리 임포트

필요한 라이브러리들을 임포트하고, GPU를 사용할 수 있는지 확인합니다. `torch`, `torch.nn`, `torch.optim` 및 `torch.utils.data` 등을 포함합니다. GPU가 사용 가능한지 확인하여 모델을 GPU에 배치할지 결정합니다.

###### Step 2: 데이터 전처리 함수

데이터 전처리를 위한 함수들을 정의합니다. 이 단계는 모델 학습에 적합한 형식으로 데이터를 준비하는 데 필수적입니다. 주요 작업으로는 결측값 처리, 데이터 타입 최적화, 이상치 제거, 텍스트 데이터 벡터화 등이 있습니다.

###### Step 3: 커스텀 데이터셋 클래스

PyTorch의 `Dataset` 클래스를 상속받아 커스텀 데이터셋 클래스를 정의합니다. 이 클래스는 모델 학습에 필요한 데이터를 로드하고, 인덱스를 통해 샘플을 반환하는 역할을 합니다. 이를 통해 데이터 전처리 및 배치를 보다 효율적으로 관리할 수 있습니다.

###### Step 4: Residual Network 모델 정의

Residual Network는 입력을 모델의 출력과 더하여 학습을 돕는 잔차 연결을 포함한 신경망입니다. 이를 통해 더 깊은 네트워크를 효과적으로 학습할 수 있습니다.

###### Step 5: 학습 및 추론 함수

모델 학습과 예측을 위한 함수들을 정의합니다. 주요 함수는 다음과 같습니다:
- `train_model`: 모델을 학습시키는 함수입니다. 이 함수는 학습 데이터를 사용하여 모델을 학습시키고, 주기적으로 손실(loss)를 출력합니다.
- `predict`: 학습된 모델을 사용하여 새로운 데이터에 대한 예측을 수행합니다.

##### 모델링 시 주목할 만한 사항

1. **데이터 전처리**:
   - 데이터의 품질이 모델의 성능에 큰 영향을 미칩니다. 결측값 처리, 데이터 타입 최적화, 이상치 제거 등을 신경 써야 합니다.

2. **Residual Network 사용**:
   - 잔차 연결을 통해 더 깊은 네트워크를 효과적으로 학습할 수 있습니다. 이는 vanishing gradient 문제를 완화하는 데 도움이 됩니다.

3. **모델 복잡도**:
   - 너무 복잡한 모델은 과적합(overfitting)될 수 있습니다. 적절한 네트워크 구조와 정규화 기법을 사용해 과적합을 방지해야 합니다.

4. **하이퍼파라미터 튜닝**:
   - 학습률, 배치 크기, 에폭 수 등 하이퍼파라미터는 모델 성능에 큰 영향을 미칩니다. 실험을 통해 최적의 값을 찾아야 합니다.

5. **조기 종료(Early Stopping)**:
   - 학습 중 과적합을 방지하기 위해 조기 종료 기법을 사용할 수 있습니다.

6. **GPU 활용**:
   - 가능하면 GPU를 사용해 학습 속도를 높입니다. 데이터와 모델이 GPU에 적합하게 설정되어 있는지 확인해야 합니다.

7. **로그 및 체크포인트**:
   - 학습 과정 중 로그를 남기고, 모델 체크포인트를 저장해 두는 것이 중요합니다. 이를 통해 학습 중간에 중단되더라도 재시작할 수 있습니다.

이러한 사항들을 고려하며 Residual Network를 구축하면 더 나은 성능과 안정성을 얻을 수 있습니다.

![TM1-PyTorch-ResNet-Regressor](./images/TM1-PyTorch-ResNet-Regressor.png)

#### Modeling Process

##### Hiperparameters

- batch_size = 32
- learning_rate = 0.001  # 학습률을 더 낮게 설정
- num_epochs = 10
- hidden_units = [128, 128]  # 모델의 복잡성을 높이기 위해 유닛 수 증가
- dropout_type = 'normal'
- hidden_activation = 'prelu'
- early_stopping = 5  # Early stopping 설정
- max_batch_size = 131072
- optimizer_type = 'adam'
- loss_function = 'gamma'

##### Training 

![TM1-PyTorch-ResNet-Regressor - Training](./images/TM1-PyTorch-ResNet-Regressor-Training.png)

##### Trial history
1. 첫번째 시도 : Team3-DTQ_code.trial.first.py
- wandb 초기화 및 설정 추가
- 필요한 모듈(math, scipy.stats) import 추가
- preprocess_data 함수 추가: 주요 입력 피처 및 위치 정보를 전처리하고 결측치를 채움
- remove_outliers 함수 추가: 특정 시기와 지역의 이상치를 제거하도록 설정
- train_model 함수 수정: 배치 단위로 wandb에 로깅 추가
- plot_rmse 함수 수정: 에포크 및 배치 단위의 RMSE를 시각화
- train_data와 test_data에 전처리 및 이상치 제거 함수 적용
- RealEstateDataset 클래스를 사용하여 데이터셋 준비
- DataLoader를 사용하여 데이터 로더 생성
- SimpleResidualNetwork 모델 정의
- 모델 학습 후 plot_rmse 함수를 통해 RMSE 시각화
- 모델 예측 및 결과 저장

2. 두번째 시도 : Team3-DTQ_code.trial.second.py
- 필요한 모든 라이브러리를 임포트
- GPU 사용 여부를 확인하는 코드 추가
- 하이퍼파라미터를 변수로 관리하고 wandb를 통해 로그를 남기도록 설정
- 메모리 사용을 줄이기 위해 데이터 타입 최적화
- 결측값을 수치형 데이터는 중앙값으로, 범주형 데이터는 최빈값으로 대체
- smooth_ridit_transform을 지정된 열에 대해 구현
- 희소 행렬(sparse matrix)을 사용하여 one_hot_encode 함수를 수정
- haversine_distance 함수를 추가하여 두 지점 간의 거리 계산
- 계약 날짜를 전처리
- Z-점수 기준값을 사용하여 이상치 제거
- 중요한 열(번지, 아파트명)에 결측값이 있는 행 삭제
- RealEstateDataset 클래스를 정의하여 데이터 로딩 처리
- SimpleResidualNetwork 클래스를 정의하여 잔여 연결(residual connections)과 드롭아웃(dropout) 레이어 추가
- train_model 함수를 구현하여 모델 훈련, 로깅 및 체크포인트 처리
- RMSE 계산과 중간 출력값 로깅 추가
- plot_rmse 함수를 구현하여 에포크 및 배치별 RMSE 시각화
- predict 함수를 구현하여 훈련된 모델로 예측
- 100 배치마다 및 각 에포크 끝에 모델 체크포인트 저장
- 체크포인트에서 모델을 로드하도록 업데이트
- RMSE 계산과 로깅이 올바른지 확인
- 희소 행렬을 사용하여 전처리 및 훈련 파이프라인의 메모리 효율성 보장

3. 세번째 시도 : Team3-DTQ_code.trial.third.py
- SimpleResidualNetwork 모델에 Residual Connections 추가
- 요청한 하이퍼파라미터를 코드에 반영 (learning_rate, num_epochs, hidden_units, dropout_type 등)
- wandb를 사용하여 실험 초기화 및 학습 과정 로깅 설정
- 체크포인트 저장 및 불러오는 기능 추가
- process_contract_date 및 preprocess_data 함수로 데이터 전처리
- 결측치 채우기 및 특정 피쳐 제거
- 모델 훈련 및 예측 기능 추가
- 최적의 모델 저장 및 예측 결과를 CSV 파일로 저장

4. 네번째 시도 : Team3-DTQ_code.trial.4th.py
- '시군구'와 '아파트명' 텍스트 컬럼을 TfidfVectorizer를 사용하여 벡터화
- vectorize_text_columns 함수를 추가하여 텍스트 컬럼을 벡터화하고 데이터프레임에 병합
- process_contract_date 함수 수정: 계약 날짜를 연도와 월로 분리
- preprocess_data 함수 수정: 결측값 처리, 데이터 타입 최적화, 필요 없는 컬럼 삭제
- remove_outliers 함수 수정: 특정 기간 내에서 z-점수를 사용하여 이상치 제거
- SimpleResidualNetwork 클래스 정의: 잔차 연결이 있는 간단한 신경망 모델
- train_model 함수 수정: 모델 학습 및 RMSE 기록
- plot_rmse 함수 수정: 학습 과정 중의 에포크 및 배치 RMSE 시각화
- predict 함수 수정: 학습된 모델을 사용하여 예측 수행
- 모델 학습 후 scaler, vectorizer, 모델을 파일로 저장
- 저장된 모델, 스케일러, 벡터라이저를 로드하여 새로운 테스트 데이터에 대해 예측 수행
- 예측 결과를 역정규화하여 원래 스케일로 복원한 후 CSV 파일로 저장

5. 다섯번째 시도 : Team3-DTQ_code.trial.5th.py
- optimize_dtypes, impute_missing_values, haversine_distance, process_contract_date, preprocess_data, vectorize_text_columns 함수를 정의하여 데이터 최적화 및 결측치 처리
- RealEstateDataset 클래스를 정의하여 데이터를 로드하고 인덱싱 가능하게 함
- SimpleResidualNetwork 클래스를 정의하여 잔차 네트워크 모델 구현
- predict 함수를 정의하여 모델 예측값 생성 기능 구현
- 학습 데이터의 열 이름을 저장하고 테스트 데이터에서 동일한 열 이름 사용하여 데이터 불일치 문제 해결
- 최종 저장된 모델 파일을 로드하고 테스트 데이터 예측 및 CSV 파일로 저장

6. 여섯번째 시도 : Team3-DTQ_code.trial.6th.py
- wandb 초기화 및 설정 추가
- 필요한 모듈(math, scipy.stats) import 추가
- preprocess_data 함수 수정: 주요 입력 피처 및 위치 정보를 전처리하고 결측치를 채움
- remove_outliers 함수 수정: 특정 시기와 지역의 이상치를 제거하도록 설정
- train_model 함수 수정: 배치 단위로 wandb에 로깅 추가
- plot_rmse 함수 수정: 에포크 및 배치 단위의 RMSE를 시각화
- train_data와 test_data에 전처리 및 이상치 제거 함수 적용
- RealEstateDataset 클래스를 사용하여 데이터셋 준비
- DataLoader를 사용하여 데이터 로더 생성
- SimpleResidualNetwork 모델 정의
- 모델 학습 후 plot_rmse 함수를 통해 RMSE 시각화
- 모델 예측 및 결과 저장

7. 일곱번째 시도 : Team3-DTQ_code.trial.7th.py, Team3-DTQ_code.prediction.7th.py
- wandb 초기화 및 설정 추가
- 필요한 모듈(math, scipy.stats) import 추가
- preprocess_data 함수 수정: 주요 입력 피처 및 위치 정보를 전처리하고 결측치를 채움
- remove_outliers 함수 수정: 특정 시기와 지역의 이상치를 제거하도록 설정
- train_model 함수 수정: 배치 단위로 wandb에 로깅 추가
- plot_rmse 함수 수정: 에포크 및 배치 단위의 RMSE를 시각화
- train_data와 test_data에 전처리 및 이상치 제거 함수 적용
- RealEstateDataset 클래스를 사용하여 데이터셋 준비
- DataLoader를 사용하여 데이터 로더 생성
- SimpleResidualNetwork 모델 정의
- 모델 학습 후 plot_rmse 함수를 통해 RMSE 시각화
- 모델 예측 및 결과 저장

##### Addiional Trial shared UpStage AI Stages Community
1. [(공유) 도메인 지식을 기반으로 Geo 정보를 활용하는 방법](https://stages.ai/en/competitions/312/board/community/post/2712)
2. [(공유) 아파트 거래에는 어떤 이상치가 있을까?](https://stages.ai/en/competitions/312/board/community/post/2713)

### 5.6. RandomForestRegressor(백경탁)

#### Baseline Code 수정하여 모델별 성능 비교 : 
* LGBMRegressor
* RandomForestRegressor
  
#### Daily Log 

##### 2024.07.16
1. 중요 피쳐를 선택하자.
2. 중요피쳐들을 줄여가자.

##### 2024-07-17
1. 결측치를 최대한 정확히 보정하자
- 아파트명,도로명,번지
2. Null 데이터를 인터넷에서 찾아 데이터를 직접 처리해보자.

##### 2024-07-19
1. target-log변환 해보자
2. 시계열 분할시 날짜별로 정렬후 해보자

#### Trial history
딥러닝은 아직 잘 모르겠어서 BaseLine Code를 이용하여 기초적인 ML로 접근 시도함.

##### 1회 제출
- Baseline code를 있는 그대로 실행해서 output 파일을 제출해 봄 (RMSE : 47133.7121)

##### 2회 제출
- 18개의 주요 피쳐를 선정하여 진행 (Feature importances 확인)
- 선택 피처: `('전용면적', '계약년', '계약월', '건축년도', '좌표X', '좌표Y', '강남여부', '신축여부', '구', '부번', '아파트명', '도로명', 'k-주거전용면적', 'k-전체동수', '주차대수', 'k-건설사(시공사)', 'k-시행사', '번지')`
- 피쳐를 줄임으로써 성능 향상이 두드러짐 (RMSE : 22391.4457)

##### 3회 제출
- 복잡도를 줄이기 위해 10개의 주요 피쳐를 선정하여 진행 (Feature importances 확인)
- 선택 피처: `('아파트명', '전용면적', '층', '건축년도', '도로명', '구', '동', '계약년', '계약월', '강남여부')`
- 피쳐를 줄임으로써 성능 향상됨 (RMSE : 19745.2556)

##### 4회 제출
- 복잡도를 줄이기 위해 8개의 주요 피쳐를 선정하여 진행 (Feature importances 확인)
- 선택 피처: `('시군구', '번지', '본번', '아파트명', '전용면적', '건축년도', '도로명', '계약년')`
- 피쳐를 줄임으로써 성능 향상됨 (RMSE : 19070.3806)

##### 5회 제출
- 복잡도를 줄이기 위해 6개의 주요 피쳐를 선정하여 진행 (Feature importances 확인)
- 선택 피처: `('시군구', '번지', '아파트명', '전용면적', '건축년도', '계약년')`
- 피쳐를 줄임으로써 성능 개선되지 않음 (RMSE : 19324.2673)

##### 6회 제출
- 아파트명과 번지 데이터를 Null값이 아닌 실제 값으로 찾아서 적용해 봄
- 9개의 주요 피쳐를 선정하여 진행 (Feature importances 확인)
- 선택 피처: `('번지', '본번', '아파트명', '전용면적', '건축년도', '도로명', '구', '동', '계약년')`
- 결측치를 보강했으나 성능 개선 거의 없음 (RMSE : 19237.2973)

##### 7/8회 제출
- 시계열 k-Fold(5)를 적용 - 성능 개선이 있었음 
- 아파트명과 번지 데이터를 Null값이 아닌 실제 값으로 찾아서 적용해 봄
- 11개의 주요 피쳐를 선정하여 진행 (Feature importances 확인)
- 선택 피처: `('번지', '본번', '아파트명', '전용면적', '층', '건축년도', '도로명', '구', '동', '계약년', '강남여부')`
- 두 가지 테스트 모델 중 RandomForestRegressor 성능이 더 좋음
  - LGBMRegressor (RMSE : 18873.1753)
  - RandomForestRegressor (RMSE : 18211.8005)

##### 9회 제출
- 시계열 k-Fold(5) + 일자별 정렬하여 적용해 봤으나 실패 - LGBMRegressor (RMSE : 85754.8909)
- 아파트명과 번지 데이터를 Null값이 아닌 실제 값으로 찾아서 적용해 봄
- 11개의 주요 피쳐를 선정하여 진행 (Feature importances 확인)
- 선택 피처: `('번지', '본번', '아파트명', '전용면적', '층', '건축년도', '도로명', '구', '동', '계약년', '강남여부')`

##### 10회 제출
- 시계열 k-Fold(5) + target 값 log 적용했으나 별다른 성능 개선이 없었음 
- 아파트명과 번지 데이터를 Null값이 아닌 실제 값으로 찾아서 적용해 봄
- 11개의 주요 피쳐를 선정하여 진행 (Feature importances 확인)
- 선택 피처: `('번지', '본번', '아파트명', '전용면적', '층', '건축년도', '도로명', '구', '동', '계약년', '강남여부')`
- LGBMRegressor (RMSE : 19930.6836)

##### 11/12회 제출
- '구'에 대해 one-hot encoding 적용
- 시계열 k-Fold(5)를 적용 
- 아파트명과 번지 데이터를 Null값이 아닌 실제 값으로 찾아서 적용해 봄
- 34개의 주요 피쳐(원핫인코딩 포함)를 선정하여 진행 (Feature importances 확인)
- 선택 피처: `('번지', '본번', '아파트명', '전용면적', '층', '건축년도', '도로명', '동', '계약년', '구'(25개))`
- 두 가지 테스트 모델 중 RandomForestRegressor 성능이 더 좋음
  - LGBMRegressor (RMSE : 19180.4314)
  - RandomForestRegressor (RMSE : 18419.8465)

##### 13/14회 제출
- 파생변수 만들지 않고 Feature 6개만 선택하여 실행해 봄 
- 선택 피처: `('시군구', '번지', '아파트명', '전용면적', '계약년월', '건축년도')`
- 아파트명과 번지 데이터를 Null값이 아닌 실제 값으로 찾아서 적용해 봄
- k-Fold(5) 적용, LGBMRegressor로 제출 시 (RMSE : 19087.0222)
- k-Fold(3) 적용, RandomForestRegressor로 제출 시 (RMSE : 17910.0918)
- 두 가지 테스트 모델 중 RandomForestRegressor 성능이 더 좋음

#### Mentoring list
##### 2024-07-18 오전 9:30 
- 테스트 셋 분리 : 랜덤 분할을 시계열 분할하면 좋을 듯.
- 모델 찾고
- 데이터 분할- 시계열 고려, k-Fold
- 피쳐 엔지니어링


### 5.6. Light GBM(한아름)

#### Trail history
##### Baseline Code 로 모델별 성능 비교 :
- Randomforest
- XGBoost
- LightGBM
- CatBoost
##### Light GBM 모델로 Time Series K-Fold 데이터 분할 후 성과 측정 :
- Baseline Code 5개 fold 평균
- Baseline Code Top3 fold 평균
- Feature Importance 9개, 5개 fold 평균
- Feature Importance 9개,  Top3 fold 평균

#### Metoring list
##### 2024-07-18 오전 9:30 
-  Test 데이터는 날짜 순으로 정렬이 안되어 있는데,, 그럼 시계열 요소를 고려하지 않고, 모델 훈련을 하는게 맞는걸까요? 아니면 시계열 요소를 고려하고 학습 후 Test 데이터에서 알아서 날짜 적용이 되는 건가요? Time Series K-fold 로 데이터 분할 후 성과가 더 떨어져서요

##### 1대1 첨삭 지도
- 캐글에서 1등 50번 등 경진대회 수상 이력만으로는 크게 의미없다. 그래서 뭐? 라는 반응이라고...
- 어떤 논리와 사고로 접근했는지가 중요하고, 그 과정에서의 사고의 깊이를 본다고 하더라구요.
- 블로그나 노션 등에 등에 접근방식과 사고의 깊이를 볼 수있게 잘 정리해 두면 좋다.
- 블로그는 단순 기술 블로그 보다는 위와 같은 내용을 담고 커뮤니케이션 수단으로 활용하면 좋고,
- 깃허브는 최종코드 게시 용도로 활용하면 좋다.

### 5.7. Baseline Code Enhancement(이승현)
- Baseline code를 이용하여 거래 기록이 적은 아파트들은 마지막 거래 가격으로 측정하는 피쳐만 추가해봄
  


## 6. Result

### Leader Board

#### Final - Rank 3

![UpStage-ML-Regression-ML3-LeaderBoard-Final](./images/UpStage-ML-Regression-ML3-LeaderBoard-Final.png)

#### Submit history
| Final Submission | Model Name              | Submitter | RMSE         | RMSE (Final)  | Created at           | Phase      |
|------------------|-------------------------|-----------|--------------|---------------|----------------------|------------|
|                  | Slim Residua...l.7th    | 박석      | 544909.9845  | 553395.0503   | 2024.07.19 18:35     | Complete   |
|                  | Slim Residua...l.7th    | 박석      | -            | -             | 2024.07.19 18:33     | Failed     |
|                  | Slim Residua...l.7th    | 박석      | -            | -             | 2024.07.19 18:25     | Failed     |
|                  | LightGBM_TSS...g.0-4    | 한아름    | 84279.3703   | 76605.7941    | 2024.07.19 17:47     | Complete   |
|                  | LightGBM_TSS...g.0-4    | 한아름    | -            | -             | 2024.07.19 17:44     | Failed     |
| ✅               | feat6_k_fold3_RFR       | 백경탁    | 17910.0918   | 14779.9300    | 2024.07.19 17:02     | Complete   |
|                  | LightGBM_Tim...eline    | 한아름    | 92986.5551   | 86036.4088    | 2024.07.19 16:58     | Complete   |
|                  | LightGBM_Tim...eline    | 한아름    | -            | -             | 2024.07.19 16:56     | Failed     |
|                  | features6_k_...LGBMR    | 백경탁    | 19087.0222   | 16114.5788    | 2024.07.19 16:56     | Complete   |
|                  | feat10_34_k_fold_RFR    | 백경탁    | 18419.8465   | 15327.6759    | 2024.07.19 16:14     | Complete   |
|                  | f10-34_k_fold_LGBM      | 백경탁    | 19180.4314   | 16632.0555    | 2024.07.19 16:01     | Complete   |
|                  | features11_k...r_log    | 백경탁    | 19930.6836   | 17291.7021    | 2024.07.19 13:55     | Complete   |
|                  | o_11features...LGBMR    | 백경탁    | 85754.8909   | 80829.6259    | 2024.07.19 11:25     | Complete   |
|                  | output_few11...ssor1    | 백경탁    | 18211.8005   | 15486.8446    | 2024.07.19 00:08     | Complete   |
|                  | output_few11...essor    | 백경탁    | 18873.1753   | 16126.8200    | 2024.07.18 23:58     | Complete   |
|                  | output_few11..._LGBM    | 백경탁    | 18873.1753   | 16126.8200    | 2024.07.18 19:01     | Complete   |
|                  | features_9_apt          | 백경탁    | 19237.2973   | 17739.6405    | 2024.07.18 00:14     | Complete   |
|                  | LightGBM_Bas...eCode    | 한아름    | 42953.6440   | 30844.8901    | 2024.07.17 23:00     | Complete   |
|                  | features_6              | 백경탁    | 19324.2673   | 17830.7967    | 2024.07.17 16:00     | Complete   |
|                  | features_8              | 백경탁    | 19070.3806   | 17634.2845    | 2024.07.17 14:53     | Complete   |
|                  | features_10             | 백경탁    | 19745.2556   | 17015.5028    | 2024.07.17 14:14     | Complete   |
|                  | features_18             | 백경탁    | 22391.4457   | 19965.5628    | 2024.07.16 18:59     | Complete   |
| ✅               | House_Price_..._Slim    | 박석      | 15901.0045   | 11943.8758    | 2024.07.16 18:53     | Complete   |
|                  | House_Price_..._Slim    | 박석      | -            | -             | 2024.07.16 18:50     | Failed     |
|                  | House_Price_..._Slim    | 박석      | -            | -             | 2024.07.16 18:50     | Failed     |
|                  | House_Price_..._Slim    | 박석      | -            | -             | 2024.07.16 18:44     | Failed     |
|                  | Keras Slim R...essor    | 박석      | -            | -             | 2024.07.16 18:36     | Failed     |
|                  | Keras Slim R...esson    | 박석      | -            | -             | 2024.07.16 18:25     | Failed     |
|                  | BaselineCode            | 백경탁    | 47133.7121   | 34223.5885    | 2024.07.16 16:19     | Complete   |


### Presentation
- [UpStageAILAB-1st-Competition-ML-REGRESSION-Presentation-TEAM3-20240722.pptx](https://docs.google.com/presentation/d/1iltGXim8d76wyGYrOvsqiViW75pLsv-1/edit?usp=sharing&ouid=102302788798357252978&rtpof=true&sd=true)

## etc

### Meeting Log

- [Team 3 Notion](https://sincere-nova-ec6.notion.site/3-9a10ac89075c40ea904bb187b30d178c) 참고

### Reference

  - [eXtreme_Gradient_Boosted_Trees_Regressor_with_Early_Stopping_(Gamma_Loss)_(Fast_Feature_Binning)_House_Price_Prediction_documentation.docx](https://docs.google.com/document/d/1ieoE_vvncqXRddT5xMxrzeIUCXxyMmXP/edit?usp=sharing&ouid=102302788798357252978&rtpof=true&sd=true)
  - [Keras_Slim_Residual_Neural_Network_Regressor_using_Adaptive_Training_Schedule_(1_Layer__64_Units)_House_Price_Prediction_documentation.docx](https://docs.google.com/document/d/1id1UiyfcAMOcOv4JGMpMAfNa-2bHD-4q/edit?usp=sharing&ouid=102302788798357252978&rtpof=true&sd=true)
  - [Light_Gradient_Boosted_Trees_Regressor_with_Early_Stopping_(Gamma_Loss)_House_Price_Prediction_documentation.docx](https://docs.google.com/document/d/1iaKlkWX0nRr29ese_se3u-FrKDwUyW9G/edit?usp=sharing&ouid=102302788798357252978&rtpof=true&sd=true)
