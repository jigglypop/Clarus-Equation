# G9-S: differential growth와 국소 꼬임 항

## 수학적 최소모형

주기적 피질 띠의 높이를 \(y(s,t)\)라 하고 에너지를

\[
E[y]=\int\left[\frac{\kappa}{2}(y'')^2+\frac{\mu}{2}y^2-\frac{\gamma(s)}2(y')^2+\frac{\beta}{4}(y')^4\right]ds
\]

로 둔다. gradient flow는

\[
\partial_t y=-\kappa y''''-\mu y-\partial_s(\gamma y')+\beta\partial_s((y')^3)
\]

다. 균일 \(\gamma\)의 선형 mode \(e^{iks}\) 성장률은 \(\lambda(k)=\gamma k^2-\kappa k^4-\mu\)이므로 \(\gamma>2\sqrt{\kappa\mu}\)일 때 불안정 mode가 존재하고 \(k_*^2=\gamma/(2\kappa)\)가 가장 빨리 자란다. 이는 꼬임 가설 없이도 주름이 생긴다는 기계적 영가설이다.

대안은 \(\gamma(s)=\gamma_0+a b(s)+c q(s)\)로 두며 \(b\)는 관측 성장장, \(q\)는 국소 결합/꼬임 대용장이다. 합성 null 세계는 \(c=0\), 대안 세계는 \(c=0.8\)이다. 숨은 작은 성장 교란을 양쪽에 넣어 정확한 자기복제를 막는다.

동일한 후보 grid를 null과 대안 train seed에 각각 적합한다. null에서 0이 아닌 \(c\)를 선택하면 거짓 양성이고, 대안 holdout에서 mechanics-only보다 profile RMSE와 주름 peak 위치가 좋아지지 않으면 꼬임 항은 식별되지 않은 것이다. 이 gate는 수치 식별성 검사일 뿐 뇌 주름의 생물학적 증거가 아니다.

V1 validation은 대안 계수 0 선택, 개선 0, 실행 18.29초로 실패했다. 감사 결과 `hidden=False` 분기가 난수장 생성을 건너뛰어 truth와 candidate 초기조건이 달라지는 paired-world 결함이 있었다. V2는 숨은 장 사용 여부와 무관하게 같은 난수 스트림을 소비하고 동일 truth trajectory를 캐시한다. 물리계수와 판정 문턱은 유지한다.

V2는 null 0, 대안 0.8을 정확히 선택하고 RMSE를 76.9% 줄였지만 peak gain이 5.6%p라 실패했다. 기준선 peak alignment가 이미 92.7%인 것은 2 grid 이내를 모두 같은 성공으로 센 지표 포화였다. V3는 각 실제 peak에서 가장 가까운 예측 peak까지의 원형 grid 거리 평균을 사용하고 이를 최소 20% 줄이도록 요구한다.

## V3 최종 결과

locked test는 `PASS`였다. null 세계의 선택 계수는 0, 대안 세계는 실제값 0.8을 회복했다. profile RMSE는 0.04640에서 0.01001로 78.4% 감소했고 실제 peak에서 최근접 예측 peak까지의 평균 원형 거리는 1.067에서 0.343 grid로 67.9% 감소했다. 외부 다운로드는 0, 실행시간은 11.81초였다. 결과는 `artifacts/agi/folding_twist_test_v3.json`에 있다.

seed 66003에서는 후보 peak 거리가 기준선보다 나빠졌다. 따라서 개별 발달세계마다 개선된다는 주장은 하지 않는다. 더 중요한 한계는 대안 세계 자체를 이 수식으로 생성했다는 점이다. 이 결과는 수치적 식별 가능성과 null 거짓양성 방지만 보이며 실제 뇌에서 \(q(s)\)가 존재한다는 증거가 아니다.
