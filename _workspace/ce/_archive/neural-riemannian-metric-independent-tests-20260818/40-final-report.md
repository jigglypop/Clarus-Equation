# 신경 리만 계량 가설의 독립 검정 경로

Status: COMPLETE

Audit Gate: PASS

## 초록

이 연구는 연결 변화가 신경 상태공간의 계량을 바꾸고 그 변화가 이후
궤적을 제약한다는 가설을 서로 독립적인 방법으로 검정하는 경로를
정리했다. E17은 활동에서 적합한 $J,Q$의 함수들을 비교했으므로 물리적
피질 접힘, 학습 전 기준 계량, 구조 연결 변화를 분리하지 못했다. 본
연구는 물리적 표면 계량 $h$, 기존 상태공간 계량 $g_0$, 개입 후 계량
$g_t$를 서로 다른 대상으로 고정했다. 문헌과 수학 검토 결과, 현재 공개
자료 가운데 $\Delta W^s\to\Delta g\to\Delta x$ 전체를 같은 실험 단위에서
검정한 자료는 없었다. 가장 빠른 부분검정은 홀로그래픽 미세자극으로
국소 반응 계량을 만들고 독립 미래 궤적을 예측하는 방법이며, 가장 작은
제안형 full-chain 설계는 같은 세포·시냅스의 구조, 반응장, 이후 궤적을
종단 측정하는 방법이다. 따라서 현재 지위는 수학적으로 검정 가능한
강한 가설이지만 생물학적으로 확인되지 않은 예측이다.

## 기존 주름의 위치

[정의] 물리적 피질 계량 $h$는 이랑과 고랑, 표면 지오데식, 층 깊이,
세포형, 축삭 배선 길이를 담는 해부학적 대상이다. 신경 활동 좌표 $z$의
계량과 정의역부터 다르므로 둘을 직접 더하거나 같은 곡률로 부르지
않는다.

[정의] $g_0(z)$는 학습이나 개입 이전에 발달, 기존 연결, 상태 점유와
동역학이 만든 기준 상태공간 계량이다. $g_t(z,c)$는 시점 $t$와 문맥
$c$에서 추정한 계량이다. 검정 대상은 $g_t$ 자체보다 상대 변형이다.

$$
A_t(z,c)=g_0(z)^{-1}g_t(z,c),
\qquad
L_t(z,c)=\log A_t(z,c).
\tag{1}
$$

[정리] 같은 좌표변환을 $g_0$와 $g_t$에 적용하면 $A_t$는 닮음변환되므로
그 고유값과 다음 상대 SPD 거리는 변하지 않는다.

$$
d_{\mathrm{SPD}}(g_0,g_t)
=\left\|\log\!\left(g_0^{-1/2}g_tg_0^{-1/2}\right)\right\|_F.
\tag{2}
$$

증명은 $A'_t=P A_tP^{-1}$에서 고유값이 보존된다는 사실로 끝난다.
자세한 반례와 조건은 [11-math.md](11-math.md)에 있다.

[산출] E17에는 표면 mesh, 이랑·고랑, 층, 세포형, 축삭 길이, 직접
$W^s$, 비상수 $g_0(z)$가 없었다. 기존 해부학의 효과가 활동과 $J,Q$에
암묵적으로 섞였을 수는 있지만, 어느 부분이 기존 주름이고 어느 부분이
학습 변형인지 식별할 수 없다. 따라서 E17은 기존 주름을 통제한 검정이
아니다.

## 독립 검정 포트폴리오

다음 경로들은 새 정보원을 추가하는 경우만 독립 경로로 센다. 같은
$J,Q$를 다른 식으로 다시 쓴 후보는 별도 증거로 세지 않는다.

| 순위 | 경로 | 독립 정보 | 검정 범위 |
|---|---|---|---|
| A1 | baseline-first 같은 세포·시냅스 종단 측정 | 개입 전후 $W^s$, 독립 반응장, 이후 궤적 | 조건부 $H_W,H_G,H_C$ |
| A2 | 표적 특이적 LTP/LTD·시냅스·회로 개입 | 무작위 표적, 음성·비표적·동일 발화 대조 | 조건부 $H_W,H_G,H_C$ |
| A3 | 홀로그래픽 perturbational system identification | 측정한 입력 $B$와 국소 반응 타원체 | $H_G$만 |
| A4 | 폐루프 최적제어 에너지 | 측정한 $B,R_u$, 실제 입력 에너지와 성공률 | $H_G$만 |
| A5 | 수면·replay 용량 무작위화 | pre/post 구조, sham replay, 다음날 궤적 | 조건부 $H_W,H_G,H_C$ |
| B1 | 자발활동에서 유발반응 예측과 역방향 예측 | 서로 분리한 활동 블록 | $H_G$ |
| B2 | committor·transition path·first-passage | 독립 생존분포와 경로 확률 | 명시한 bridge 검정 |
| B3 | 곡률·holonomy·geodesic deviation | 독립 loop와 $C^2$ 계량장 | 기하 성질만 |
| B4 | topology와 metric shortcut 분리 | edge 생성과 weight 변화의 분리 개입 | 필요 설명변수 판별 |
| B5 | BCI·심리물리·행동 일반화 | 새로운 decoder 방향과 학습곡선 | 행동 타당성 |
| B6 | cross-task·cross-animal·cross-scale transfer | 다른 과제, 동물, 측정 modality | 외적 재현성 |
| B7 | 자연실험·도구변수·용량반응 | 독립 개입원과 민감도 분석 | 가정 조건부 인과 |
| C1 | synthetic ground truth | 알려진 $W,g,v,Q$와 관측 왜곡 | 추정기 검증 |
| C2 | 공개자료 삼각측량 | 구조, 활동, 개입 자료의 역할 분담 | 구성요소 가능성 |
| C3 | 물리 표면 eigenmode·connectome harmonic | $h$와 해부학 기반 경쟁모형 | 기존 주름 대조 |

각 경로의 입력, 점수, null, nuisance, kill criterion은
[12-routes.md](12-routes.md)에 있고, 자료·기술별 가능 범위는
[10-sources.md](10-sources.md)와
[capability-matrix.csv](artifacts/capability-matrix.csv)에 있다.

## 기준 계량 선택

[공리: 모델 선택] 개입 전 자료에서만 네 생산자 후보를 고정한다.
`P-h`는 표면·층 해부학, `P-W`는 기존 구조 연결, `P-D`는 학습 전
동역학, `P-C`는 사전 선언한 결합모형에서 $g_0$를 만든다. 각 후보는
별도 복잡도 예산을 받고 개입 후 궤적으로 선택하지 않는다.

[예측] 실제 학습 전 구간 안에 가짜 개입 시점을 넣으면 상대 변형은
0과 양립해야 한다. 모든 생산자는 같은 자유도의 flat-pullback null과
경쟁한다. 이 placebo에서 이미 큰 $\Delta g$가 나오거나 비선형 좌표
warp가 같은 결과를 내면 학습 변형 주장을 기각한다.

## 궤적 연결 법칙

[공리: 물리 모형] 계량은 벡터장을 스스로 정하지 않는다. 인과적
$g\to\gamma$를 말하려면 다음과 같은 법칙 하나를 결과 전에 고정해야
한다.

$$
dz_t=-g(z_t,c)^{-1}\nabla V(z_t,c)\,dt
     +B(z_t,c)\,dB_t,
\qquad Q=BB^\top.
\tag{3}
$$

이 모형은 같은 수 이상의 매개변수를 가진 자유로운 $v,Q$ 모형과
경쟁한다. $\Phi(W^s,c)$, 식 (3), 매개변수 예산, 직접
$W^s\to\text{trajectory}$ 항을 모두 결과 전에 고정하지 않으면 $H_C$로
승격하지 않는다.

## 실행 순서

1. 가장 먼저 A3를 수행한다. 작은 무작위 광자극 방향에서 얻은 국소
   반응 타원체가 별도 무자극 궤적을 직접 $v,Q$보다 잘 예측하는지 본다.
2. 다음으로 작은 확인 회로나 광학적으로 접근 가능한 소동물에서 A1을
   구성한다. 전체 connectome은 필요하지 않으며, 재식별 가능한 작은
   회로의 표적 연결과 미래 궤적이면 된다.
3. A1이 통과한 뒤 A2로 특정 연결을 강화·약화해 방향과 용량반응을
   확인한다. 표적 특이성과 exclusion이 실패하면 mediation을 주장하지
   않는다.
4. 마지막으로 A5를 붙여 학습 직후, 수면 후, matched wake 후의
   $W^s,g_t$, 다음날 경로를 비교한다. Ricci flow는 별도 evolution law가
   단순 smoothing보다 나을 때만 후보가 된다.

## 관측과 비교

[경험식] E17에서는 S2가 일부 horizon에서 근소한 평균 우세를 보였지만
동물별 방향이 일치하지 않았고 population winner는 금지됐다. 이는
활동 기반 metric-like summary의 retrospective feasibility이지 독립
기하 관측이 아니다. [선행 최종 보고서](../neural-riemannian-metric-validation-20260818/40-final-report.md)가
그 경계를 기록한다.

[산출] 문헌은 필요한 구성요소가 이미 따로 존재함을 보인다. MICrONS는
같은 조직의 기능과 endpoint EM 연결을 공동등록했고, 종단 spine
연구는 학습 중 구조·기능 변화를 측정했으며, 홀로그래픽 광유전학은
세포 수준 입력과 반응을 측정할 수 있다. 그러나 이들을 같은 실험
단위에서 pre/post 구조, 독립 계량, 이후 궤적으로 묶은 공개자료는 찾지
못했다. 이것은 현재의 자료 공백이지 가설의 확인이 아니다.

## 기각 조건과 현재 지위

[예측] 사전고정 A1/A2에서 다음 중 하나가 독립 동물에서 반복되면 명시한
connectivity-mediated bridge는 신뢰를 잃는다.

1. $\Delta W^s$가 어떤 고정 생산자의 $L_t$도 안정적으로 예측하지 못한다.
2. $g$가 자유로운 $v,Q$, gain/noise, 물리 표면, flat-pullback null보다
   미래 경로 proper score를 높이지 못한다.
3. 세포 또는 연결 identity를 순열해도 같은 점수가 나온다.
4. 표적, 비표적, 동일 발화 대조에서 효과가 구별되지 않는다.
5. 효과 방향이 동물·과제·modality 사이에서 양립하지 않는다.

[미완성] $H_W$, $H_G$, $H_C$는 모두 생물학적 확인 전이다. A3만 통과하면
반응 기반 기하가 유용하다는 뜻이지 연결이 그 기하를 만들었다는 뜻은
아니다. A1/A2의 두 연결과 인과 식별 조건까지 통과해야 원래 핵심식에
경험적 지위를 줄 수 있다.

## 재현

```powershell
python _workspace/ce/neural-riemannian-metric-independent-tests-20260818/artifacts/check_counterexamples.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .codex/hooks/run.ps1 check _workspace/ce/neural-riemannian-metric-independent-tests-20260818 final
```

수학 spot check, capability matrix parse, 단계별 gate 결과는
[31-validation.md](31-validation.md)에 기록했다. 출처는 2026-08-18에
확인했으며 원 논문과 공식 저장소 링크를 [10-sources.md](10-sources.md)에
모았다.
