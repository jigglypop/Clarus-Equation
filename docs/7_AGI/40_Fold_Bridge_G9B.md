# G9-B: Fold-Bridge 희소 지름길 가설

## 가설 전환

기능 atlas 경계 실험은 고랑이 Yeo 대규모 기능망을 직접 분할한다는 단순 가설을
지지하지 않았다. 반대 가능성으로 접힘이 표면상 먼 영역을 3차원과 superficial
white matter에서 가깝게 만드는지 시험한다.

직접 회색질 접촉을 주장하지 않는다. 고랑 사이에는 연막과 CSF가 있으므로 실제
연결 후보는 U-fiber 같은 백질 경로다. 현재 TemplateFlow에는 tractography가
없으므로 B1은 기하학적 기회만 검출한다.

## 검출 조건

두 pial 꼭짓점 `i,j`가 다음을 만족해야 한다.

- pial Euclidean 거리 6 mm 이하
- mesh topology에서 최소 4-hop 이상
- 법선 대향 cosine 0.5 이상
- 두 법선이 상대 꼭짓점을 향하는 cosine 0.35 이상
- pial-white 대응 깊이 0.5 mm 이상

법선 방향은 `white → pial`로 정렬한다. 표면 거리는 pial edge-weighted Dijkstra로
80 mm까지 계산한다. 실제 U-fiber가 없으므로 낙관적인 white-route 하한

\[
d_W^{proxy}
=
\|p_i-w_i\|+\|w_i-w_j\|+\|w_j-p_j\|
\]

를 사용한다. shortcut ratio는

\[
R_{ij}=\frac{d_M(i,j)}{d_W^{proxy}(i,j)}
\]

다. 이 chord가 실제 white matter 내부에 머문다는 보장은 없으므로 `R>1`도 연결
증거가 아니라 추적 우선순위다.

## 합성 대조와 구현 수정

공간 hash bucket으로 3차원 근접쌍만 만들고, unweighted BFS로 local neighbor를
제외한 뒤 제한 Dijkstra를 적용한다. 첫 합성 U-strip은 법선 orientation 오류로
양성 0을 냈다. 임계값을 바꾸지 않고 pial-white 방향으로 법선을 정렬해 수정했다.

- 평면 strip 후보: 0
- U-fold 전체 후보: 126
- U-fold `R≥1.5` 강한 후보: 59
- 강체 회전·평행이동 결과: 불변

## V1: 보편적 bridge 가설 — 실패

좌반구 결과:

- 대향 후보: 5,932
- median `R`: 0.812
- `R>1` 비율: 32.43%
- 최대 `R`: 5.291

후보 대부분에서 white-route proxy가 표면 경로보다 짧지 않았다. 따라서 `고랑의
대부분이 지름길이다`라는 V1은 `FAIL`이다.

## V2: 희소 bridge tail — 양반구 통과

V1 분포를 본 뒤 선택했다는 사실을 공개하고 `R≥1.5`만 강한 bridge 후보로
재정의했다. 우반구를 보기 전에 strong count 250 이상, fraction 5% 이상,
90백분위 `R≥1.5`를 잠갔다.

| 지표 | 좌반구 | 우반구 |
|---|---:|---:|
| 전체 대향 후보 | 5,932 | 5,894 |
| `R≥1.5` 강한 후보 | 844 | 754 |
| 강한 후보 비율 | 14.23% | 12.79% |
| `R` 90백분위 | 1.715 | 1.679 |
| 최대 `R` | 5.291 | 4.991 |

좌/우 strong-count 비는 1.119로 사전등록 범위 0.5–2.0 안에 들었다. 양반구 모두
`PASS`다.

## 현재 해석

접힘은 모든 bank 사이를 연결하는 것이 아니라, 일부 위치에서만 표면 경로보다
훨씬 짧을 수 있는 희소 shortcut 후보를 만든다. 이는 small-world graph의 sparse
long edge와 비슷하다. 그러나 현재 `white chord`는 가장 낙관적인 하한이므로 실제
축삭 경로가 존재하는지 알 수 없다.

다음 반증 단계는 strong 후보와 동일한 거리·곡률을 가진 matched control을 만들고,
독립적인 superficial-white-matter tractography에서 U-fiber density 또는 streamline
existence를 비교하는 것이다. strong 후보가 matched control보다 높지 않다면
Fold-Bridge의 연결성 해석은 폐기해야 한다.
