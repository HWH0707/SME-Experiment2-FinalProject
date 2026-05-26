# 모티베이션 & 인트로

이번 프로젝트의 입력은 18개 기지국에서 측정한 RTT 기반 거리값 `d_hat`이고, 출력은 각 사용자의 2차원 위치 `p_hat`이다. 중간 발표 전후 실험에서 반복적으로 확인한 점은, 거리 측정값에는 단순 잡음뿐 아니라 기지국별 bias, 일부 큰 outlier, 사용자 위치에 따른 기하학적 민감도 차이가 함께 섞인다는 것이다. 따라서 측정값을 바로 위치로 회귀하거나, 모든 센서 값을 같은 신뢰도로 평균내는 방식은 데이터가 조금만 바뀌어도 오차가 커질 수 있다.

본 프로젝트에서는 이 문제를 `Geometry-Aware Residual Correction`으로 접근했다. 먼저 RTT 값을 기지국별로 보정하여 실제 거리와 가까운 값으로 바꾸고, 그 거리들이 만드는 원들의 교차 구조를 이용해 robust multilateration 위치를 구한다. 이후 기하 기반 해가 남기는 잔차 패턴을 feature로 만들고, 작은 학습 모델이 그 잔차를 보정하도록 설계했다. 핵심은 ML 모델이 전체 위치를 처음부터 맞히는 것이 아니라, 물리적으로 설명 가능한 기하 해의 남은 오차만 학습하게 하는 것이다.

이 방향을 선택한 이유는 세 가지이다. 첫째, 최종 데이터는 `d_hat`과 `p_bs`가 명확히 주어지므로 anchor geometry를 직접 활용할 수 있다. 둘째, hidden test에서는 제공된 700명에 대한 과적합이 위험하므로, 순수한 black-box fitting보다 기하 제약을 가진 구조가 더 안정적이다. 셋째, 위치 추정 문제 자체의 특성인 원 교차, 잔차, 조건수, anchor 배치 정보를 알고리즘 중심에 두면 보고서와 코드 모두에서 독립적인 설계 의도가 분명해진다.

# 알고리즘 설명

기지국 좌표를 `b_i`, 사용자 위치 후보를 `x`, i번째 측정값을 `d_i`라고 두었다. 실제 거리는 `r_i(x) = ||x - b_i||`로 표현된다. 알고리즘은 보정, 기하학적 초기 위치 추정, robust refinement, 잔차 보정의 네 단계로 구성된다.

첫 번째 단계는 기지국별 affine 거리 보정이다. 제공된 training 위치 `p`로부터 기지국과 사용자 사이의 실제 거리 `r_i`를 계산하고, 각 기지국마다 `r_i ≈ a_i d_i + c_i`가 되도록 `a_i`와 `c_i`를 추정한다. 이때 전체 데이터를 그대로 쓰지 않고, 측정값의 1~99 percentile 범위만 먼저 사용한 뒤, residual의 median absolute deviation 기준으로 큰 outlier를 한 번 더 제거한다. 최종 보정 거리는 `d'_i = clip(a_i d_i + c_i)`로 사용한다. 또한 보정 후 residual의 robust scale을 `sigma_i`로 저장하여, 이후 기지국별 신뢰도 가중치로 사용한다.

두 번째 단계는 선형화된 multilateration 초기값이다. 유효한 거리 중 가장 가까운 기지국을 reference anchor로 잡고, 거리 제곱 차이를 이용해 위치에 대한 선형 방정식을 만든다. 가까운 거리일수록 상대적으로 신뢰도가 높다고 보고 `1 / d'_i` 계열의 가중치를 둔다. 이 선형해가 불안정하거나 유효 anchor가 부족하면, 거리의 역제곱 가중 centroid를 초기값으로 사용한다.

세 번째 단계는 robust nonlinear multilateration이다. 초기 위치 `x_0`에서 시작해 `sum rho((||x - b_i|| - d'_i) / sigma_i)`를 최소화한다. 여기서 `rho`는 soft-L1 손실이므로 일부 anchor의 거리값이 크게 튀어도 전체 위치가 한쪽으로 과도하게 끌려가지 않는다. 탐색 범위는 training 위치와 기지국 좌표를 포함하는 박스에 margin을 더해 설정했다. 이 단계의 출력은 `p_geo`이며, 이는 ML 없이도 설명 가능한 1차 위치 추정값이다.

네 번째 단계는 `p_geo`가 남긴 잔차를 학습하는 residual correction이다. 각 사용자에 대해 다음 정보를 feature로 만든다.

| feature 그룹 | 의미 |
|---|---|
| 기하 위치 | robust multilateration으로 얻은 `p_geo`의 x, y |
| 보정 거리 | 18개 anchor의 `d'_i`와 `p_geo`에서 각 anchor까지의 계산 거리 |
| 재계산 거리 | `p_geo`에서 각 anchor까지의 거리 `||p_geo - b_i||` |
| 잔차 | `||p_geo - b_i|| - d'_i`, 절댓값 잔차, `sigma_i`로 정규화한 잔차 |
| 분위수 | 거리, 잔차, 절댓값 잔차의 주요 percentile |
| 통계량 | 평균, 표준편차, 최솟값, 최댓값, 잔차 norm |
| 기하 조건수 | `p_geo`에서의 range Jacobian singular value ratio |

학습 target은 위치 자체가 아니라 `delta = p_true - p_geo`이다. 최종 보정 모델은 HistGradientBoostingRegressor와 ExtraTreesRegressor의 평균 앙상블이다. HGB는 residual feature의 연속적인 변화에 따른 보정량을 학습하는 데 도움이 될 수 있고, ExtraTrees는 anchor별 residual 조합에서 나타나는 국소적인 패턴을 보완할 수 있다고 보았다. 최종 출력은 `p_hat = p_geo + mean(delta_HGB, delta_ET)`이다. 추가로 scikit-learn pickle 버전 차이에 대비하기 위해 같은 feature를 사용하는 RBF kernel ridge fallback을 numpy 배열 형태로 저장했다. 기본 경로가 정상 동작하면 HGB+ExtraTrees 앙상블을 사용하고, 모델 로드 문제가 생길 때만 fallback을 사용한다.

기하 조건수는 `p_geo` 주변에서 각 anchor 방향 벡터로 구성한 선형화 행렬의 condition number를 사용했다. 이 값은 anchor 배치가 작은 거리 오차를 위치 오차로 얼마나 증폭시킬 수 있는지를 나타내는 보조 feature로 사용했다.

# Agent AI(e.g., ChatGPT, Claude Code, Gemini 등) 활용 방안

ChatGPT 기반 AI 도구를 사용했다. 활용 범위는 프로젝트 규칙 정리, 주차별 실험 기록에서 반복적으로 나온 문제점 요약, 알고리즘 후보 비교, `main.py`와 `train.py`의 실행 규격 점검, 보고서 초안 구조화였다. 특히 `main.py`가 hidden test에서 GT `p`를 사용하지 않는지, 사용자 수를 700으로 고정하지 않는지, 반환 shape이 `(2, num_user)`인지 확인하는 데 사용했다.

최종 알고리즘 방향은 여러 후보를 비교하면서 정했다. 처음에는 거리값을 바로 좌표로 회귀하는 방식도 생각했지만, 이 경우 기지국 배치나 거리 오차의 구조가 잘 드러나지 않는다고 보았다. 그래서 먼저 기하 기반 위치를 구하고, 그 결과에서 남은 오차만 residual correction으로 보정하는 방향으로 정리했다. AI가 제안한 내용은 참고용으로만 사용했고, 실제로는 README의 제출 조건과 `main.py`, `train.py` 실행 결과를 확인하면서 맞지 않는 부분을 수정했다. 최종 제출 코드에서는 표준 환경에 포함된 numpy, scipy, scikit-learn만 사용했다.

# 결과 도출 & 디스커션

평가는 제공된 700명 데이터를 80% training, 20% validation으로 나누어 수행했다. 아래 단일 split 결과는 random seed 42 기준이며, validation sample 수는 140개이다. 각 split에서 affine calibration parameter, robust scale, bounds, max_range, residual correction model은 training subset에서만 학습하거나 계산했고, validation subset에는 학습된 calibration과 모델만 적용했다. baseline은 affine 보정과 robust multilateration까지만 적용한 `p_geo`이고, proposed는 여기에 residual correction 앙상블을 더한 결과이다. 오차 값은 좌표 단위 기준이다.

| 방법 | MAE | RMSE | Median | P90 | Max |
|---|---:|---:|---:|---:|---:|
| Robust geometry only | 8.4120 | 10.5733 | 6.8788 | 15.8171 | 42.8975 |
| Geometry-aware residual correction | 5.5136 | 7.0662 | 4.6553 | 9.5425 | 32.2676 |

최종 보정 모델은 후보 모델을 같은 validation protocol에서 비교한 뒤 선택했다. 아래 표는 seed 3, 11, 21, 42, 77의 다섯 split 평균 ± 표준편차이다.

| 후보 모델 | MAE | RMSE | P90 | 선택 |
|---|---:|---:|---:|---|
| HGB residual correction | 6.0471 ± 0.3016 | 7.6972 ± 0.4433 | 10.8158 ± 0.5216 | 후보 |
| ExtraTrees residual correction | 6.1883 ± 0.1779 | 7.7947 ± 0.4242 | 10.9089 ± 0.9449 | 후보 |
| HGB + ExtraTrees average | 5.9669 ± 0.2332 | 7.5674 ± 0.4194 | 10.6909 ± 0.5795 | 최종 |

표의 평균과 표준편차는 seed 3, 11, 21, 42, 77의 다섯 개 validation split 결과를 기준으로 계산했다.

단일 split이 우연히 유리하게 나온 것인지 확인하기 위해 seed 3, 11, 21, 42, 77의 다섯 번 random split도 확인했다. 표의 값은 평균 ± 표준편차이다.

| 방법 | MAE | RMSE | Median | P90 | Max |
|---|---:|---:|---:|---:|---:|
| Robust geometry only | 8.6131 ± 0.3352 | 10.7760 ± 0.5660 | 6.9571 ± 0.3898 | 15.8971 ± 0.5299 | 43.4126 ± 10.7061 |
| Geometry-aware residual correction | 5.9669 ± 0.2332 | 7.5674 ± 0.4194 | 5.0441 ± 0.2868 | 10.6909 ± 0.5795 | 35.8433 ± 9.6622 |

12주차의 1차 결과는 seed 42 단일 split 기준이고, 여기의 결과는 다섯 개 random split 평균이므로 두 수치를 직접적으로 같은 조건의 성능으로 비교하지는 않았다.

전체 700명으로 최종 모델을 다시 학습한 뒤, 같은 visible data에서 저장 모델 inference 경로를 점검했다. 아래 수치는 hidden 성능을 의미하지 않고, `model.pkl`, fallback 모델, `main.py`의 inference 경로가 정상적으로 연결되는지 확인하기 위한 fitted check이다.

| 항목 | 값 |
|---|---:|
| `train.py` 실행 시간 | 4.48 s |
| `main.py` 출력 shape | (2, 700) |
| Primary fitted MAE | 0.8756 |
| Primary fitted RMSE | 1.4137 |
| Primary fitted Median | 0.6409 |
| Primary fitted P90 | 1.6118 |
| Primary fitted Max | 20.0911 |
| Fallback fitted MAE | 2.8742 |
| Fallback fitted RMSE | 3.4364 |
| Fallback fitted Median | 2.5274 |
| Fallback fitted P90 | 5.1034 |
| Fallback fitted Max | 14.7214 |

이 비교는 공정한 baseline이라고 판단한다. proposed 방법과 baseline 모두 같은 affine 거리 보정과 같은 robust multilateration을 공유하며, 차이는 마지막 residual correction을 추가했는지 여부뿐이다. 따라서 성능 향상은 단순히 전혀 다른 문제를 푼 결과가 아니라, 기하 기반 추정 이후 남은 residual feature가 validation split에서 추가적인 오차 보정에 도움이 된 결과로 해석할 수 있다.

제안한 방법의 장점은 hidden test에서 사용자 수가 달라져도 동작하고, 입력 anchor 개수와 geometry를 직접 반영한다는 점이다. 또한 `main.py`는 `p`를 읽지 않고 `d_hat`과 `p_bs`만 사용하므로 평가 규칙에 맞다. ML 모델도 전체 좌표를 직접 예측하는 역할이 아니라, 기하 기반 초기 위치가 남긴 보정량만 학습하도록 제한했기 때문에 위치 추정 문제의 물리적 구조를 어느 정도 유지할 수 있다.

# 한계 및 보완 방향

첫 번째 한계는 training data의 크기이다. 현재 모델은 제공된 700개 sample에서 기지국별 거리 보정값과 residual correction 패턴을 학습했다. 따라서 hidden data의 측정 환경이나 bias 분포가 training data와 크게 달라지면, validation에서 확인한 만큼의 개선이 그대로 유지되지 않을 수 있다.

두 번째 한계는 큰 outlier에 대한 처리이다. soft-L1 기반 robust multilateration과 residual correction을 사용했지만, 일부 anchor의 거리 측정값이 매우 크게 어긋나는 경우에는 Max error가 여전히 남아 있다. 실제 validation에서도 평균 오차와 P90은 줄었지만, Max error는 완전히 제거되지 않았다. 이는 NLOS와 같은 극단적인 측정 오류를 별도로 탐지하는 단계가 추가되면 더 개선될 수 있는 부분이다.

세 번째 한계는 모델 파일 호환성이다. 기본 제출 모델은 `model.pkl`에 저장된 HGB+ExtraTrees 앙상블이지만, scikit-learn 버전 차이로 pickle 로드 문제가 생길 가능성을 고려해 `model_fallback.npz`도 함께 준비했다. fallback 모델은 실행 안정성을 높이기 위한 장치이지만, primary 앙상블보다 성능이 낮을 수 있다. 따라서 최종 성능 측면에서는 `model.pkl`이 정상적으로 로드되는 환경이 가장 바람직하다.

향후 개선 방향으로는 anchor별 NLOS 가능성을 별도 feature로 추정하거나, validation에서 큰 Max error가 나온 sample을 분석해 outlier rejection 단계를 추가할 수 있다. 또한 기지국 배치나 측정 환경이 바뀌는 경우에는 affine calibration과 residual correction 모델을 다시 학습하는 방식으로 일반화 성능을 확인할 필요가 있다.
