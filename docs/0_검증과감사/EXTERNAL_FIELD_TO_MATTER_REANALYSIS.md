# 외부 원자료 기반 장→광자쌍→물질상 재분석

작성일: 2026-08-05  
데이터 고정본: `benchmarks/external_field_to_matter_v1.json`  
재분석 코드: `reality_stone/python/reality_stone/clarus/external_field_to_matter.py`

## 1. 결론부터

이번 루프는 합성 데이터나 새 검출식을 만들지 않았다. 서로 독립적인 세 공개
실험 자료를 내려받아 해시를 고정하고, 논문의 핵심 수치를 다시 계산했다.

| 질문 | 실제 재분석 결과 | 판정 |
|---|---|---|
| 구동된 비선형 광섬유에서 특정 파장의 비고전적 광자쌍이 나오는가 | 예측 파장 (954.313/1173.060) nm, 공개 CAR 최대 셀 (953.938/1172.889) nm, 에너지 잔차 \(0.0282\%\) | `REPRODUCED` |
| 전자기장 에너지가 질량 있는 \(e^+e^-\) 쌍으로 변환되는가 | CMS HEPData 7개 bin 적분 \(263.393\ \mu\mathrm b\), 논문값 \(263.5\ \mu\mathrm b\) | `REPRODUCED` |
| 빛으로 기존 고체의 준안정 전자·구조 상을 만들 수 있는가 | 공개 NeXus의 배열·fluence·시간축·체크섬은 확인, 정확한 (2.2) THz refit과 수명 refit은 미재현 | `PARTIAL` |
| 이 셋이 한 Clarus 장의 연속 작동을 증명하는가 | 같은 장치·동일 run·공통 coupling·에너지 전달 기록이 전혀 없음 | `NO` |
| 새 원자·새 입자·새 질량 pole 또는 무한 에너지가 생성됐는가 | 어느 자료에도 그런 관측량이 없음 | `NO EVIDENCE` |

따라서 실제로 닫힌 명제는 다음이다.

\[
\boxed{
\text{pump-powered nonclassical photon pairs}
\quad\text{and}\quad
\gamma\gamma\to e^+e^-
\quad\text{exist experimentally}
}
\]

그러나 다음 사슬은 닫히지 않았다.

\[
\boxed{
\text{Clarus field}
\to\text{optical pair}
\to\text{massive pair}
\to\text{persistent new matter}
}
\]

이는 Clarus가 반증됐다는 뜻이 아니라, 현재 외부 자료가 Clarus 고유 변수를 한 번도
측정하지 않았다는 뜻이다. 따라서 이 자료에서 얻는 Clarus 고유 존재 증거는
현재 `0`이며, 표준 QED·비선형광학·응집물질 설명의 재현도는 높다.

## 2. 데이터 provenance

### 2.1 광학 analogue DCE

- 1차 논문: [Vezzoli et al., Communications Physics 2, 84 (2019)](https://www.nature.com/articles/s42005-019-0183-z)
- 공식 데이터: [Heriot-Watt Research Portal](https://researchportal.hw.ac.uk/en/datasets/optical-analogue-of-the-dynamical-casimir-eect-in-a-dispersion-os/)
- 공개 ZIP SHA-256:
  `5aad353ad779e0ced16e35c2f809a33ed4a71e024f0f77a071e60d0c7d807af1`

공개본은 `Figure 2.txt`, `Figure 3.txt`, `Figure 4.txt`로 이루어진 그림
source data다. TDC event timestamp나 정수 원시 count는 아니므로, 게재된 오차를
사용한 수치는 재계산할 수 있지만 event-level likelihood는 다시 만들 수 없다.

### 2.2 CMS quasi-real photon fusion

- 1차 논문: [CMS, JHEP 08 (2025) 006](https://arxiv.org/abs/2412.15413)
- 공식 표: [HEPData record 155674](https://www.hepdata.net/record/ins2861858)
- 사용 API payload SHA-256:
  `f2a6b865131884f8a3d12a9784585b01e2fa4d9e1e4b5b599516210958212595`

HEPData의 \(p_{T,ee}=0\)--\(1\) GeV 전 구간 7개 unfolded bin과 세 예측
열을 사용했다. 이것도 검출기 event별 four-vector가 아니라 보정된 공개 표다.

### 2.3 1T-TaS2 준안정 hidden phase

- 1차 논문: [Maklar et al., Science Advances 9, eadi4661 (2023)](https://doi.org/10.1126/sciadv.adi4661)
- 공개 원자료: [Zenodo 8238531](https://zenodo.org/records/8238531)
- `summary.xlsx` MD5:
  `961e70d41f5e56551c4fed50debdd00c`
- Figure 3 `10.nxs` MD5:
  `0e1e418419ad94fd2ad68e4773bfdf68`
- 정적 C상 `1.nxs` MD5:
  `b10c4b8fcd670fed62bd15d6751cddb1`

이 자료는 실제 acquisition-level NeXus/HDF5다. 다만 공개 분석 코드와 정확한
fit bound는 포함되지 않았다.

## 3. 광학 광자쌍: 식이 아니라 공개 표에서 되찾은 값

논문의 실험 상수는

\[
\lambda_p=1052.44\ \mathrm{nm},\quad
\beta_2=0.45\ \mathrm{ps^2/km},\quad
\beta_4=-1.2\times10^{-55}\ \mathrm{s^4/m},
\]

\[
K=\frac{2\pi}{5\ \mathrm m},\qquad m=3
\]

이다. 공개 논문의 sideband 식

\[
\beta_2\Delta\omega^2+
\frac{\beta_4}{12}\Delta\omega^4=mK
\]

을 직접 풀어 물리적인 큰-detuning 해를 선택하면

\[
\lambda_s=954.3126699\ \mathrm{nm},\qquad
\lambda_i=1173.0600603\ \mathrm{nm}
\]

이다. Figure 3 공개 CAR 표의 전역 최대는

\[
\mathrm{CAR}_{\max}=5.1
\]

이며 위치는

\[
(\lambda_s^{\rm obs},\lambda_i^{\rm obs})
=(953.9375,1172.888889)\ \mathrm{nm}
\]

이다. 예측에서 각각 (-0.3752) nm, (-0.1712) nm 떨어져 있어 공개 표의
가장 가까운 측정 cell과 일치한다.

광자쌍 에너지 보존 잔차를

\[
\epsilon_E=
\frac{\lambda_s^{-1}+\lambda_i^{-1}}
{2\lambda_p^{-1}}-1
\]

로 계산하면

\[
\epsilon_E=2.8233\times10^{-4}=0.02823\%.
\]

관측 cell의 광자쌍 에너지는

\[
E_{s+i}=hc\left(\lambda_s^{-1}+\lambda_i^{-1}\right)
=2.3567938\ \mathrm{eV}
\]

다. Figure 4에서는 가장 좋은 heralded point가

\[
\mathrm{CAR}=4.745,\qquad
g^{(2)}(0)=0.380952381\pm0.06
\]

이고 CAR이 1보다 큰 네 점 모두 \(g^{(2)}(0)+1\sigma<1\)이다. CAR=0인
Raman control은 \(g^{(2)}(0)=1.00\pm0.04\)다. 공개 오차만 표준편차처럼 나눈

\[
\frac{1-g^{(2)}(0)}{0.06}=10.32
\]

는 강한 분리를 보여 주지만, 원시 count와 공분산이 없으므로 이를 독립 분석의
정식 \(10.32\sigma\) 발견 유의도로 부르지는 않는다.

이 결과는 구동된 \(\chi^{(3)}\) 매질에서 vacuum-seeded 비고전적 광자쌍이
선택된 파장에 나온다는 증거다. 그러나 논문 자체가 lab frame의 동일 식이
quasi-phase-matched spontaneous four-wave mixing과 일치한다고 밝힌다. 따라서
움직이는 거울의 literal DCE, Clarus 고유장 또는 자유에너지의 증명은 아니다.
에너지는 펌프에서 공급된다.

## 4. 질량 있는 입자쌍: CMS 공개 bin의 직접 적분

CMS 공개표의 각 bin에 대해

\[
\sigma_{\rm fid}
=\sum_i \Delta p_{T,i}
\left(\frac{d\sigma}{dp_T}\right)_i
\]

를 계산했다. 결과는

\[
\sigma_{\rm recalc}=263.3930128\ \mu\mathrm b,
\]

이고 논문값은

\[
\sigma_{\rm pub}=263.5
\pm1.8_{\rm stat}
\pm17.8_{\rm syst}\ \mu\mathrm b.
\]

중심값 차이는 \(0.107\ \mu\mathrm b=0.041\%\)다. 통계오차를 bin 독립으로
적분하면

\[
\delta\sigma_{\rm stat}
=\sqrt{\sum_i(\Delta p_{T,i}\delta_i)^2}
=1.7238\ \mu\mathrm b,
\]

즉 게재값 \(1.8\ \mu\mathrm b\)를 반올림 오차 안에서 되찾는다. systematic
공분산이 공개되지 않았으므로 거짓 독립 가정을 하지 않고 두 극한만 계산했다.

\[
16.4958\ \mu\mathrm b
\le \delta\sigma_{\rm syst}
\le19.3189\ \mu\mathrm b.
\]

게재값 \(17.8\ \mu\mathrm b\)는 이 범위 안에 있다.

같은 bin에서 모델을 적분하면

| 모델 | 적분 단면적 | 측정 대비 |
|---|---:|---:|
| gamma-UPC | \(265.6159\ \mu\mathrm b\) | \(+0.844\%\) |
| SuperChic | \(260.9444\ \mu\mathrm b\) | \(-0.930\%\) |
| STARlight | \(225.0995\ \mu\mathrm b\) | \(-14.54\%\) |

이다. 19,689개의 선택된 \(e^+e^-\) 후보와 단면적은 quasi-real photon
fusion을 포함한 표준 QED 예측과 맞는다. 다만 ultraperipheral heavy-ion의
광자는 이상적인 자유 on-shell 두 광자가 아니라 이온 Coulomb field의 작은
virtuality를 가진 equivalent photon이다. 정확한 주장은 “전자기장 에너지가
보통의 질량 있는 렙톤쌍으로 전환됨”이지 “자유광자 두 개의 완전 고립 충돌”이
아니다.

## 5. 왜 광학 결과와 CMS 결과가 Clarus 사슬로 이어지지 않는가

전자·양전자 정지질량 문턱은

\[
2m_ec^2=1.0219979\times10^6\ \mathrm{eV}.
\]

광학 실험의 관측 광자쌍 \(2.3567938\ \mathrm{eV}\)와 비교하면

\[
\frac{2m_ec^2}{E_{s+i}}
=433{,}639.1.
\]

즉 그 광자쌍 하나는 \(e^+e^-\) 정지질량 문턱보다 약 43만 배 낮다. CMS
selection의 \(m_{ee}>5\ \mathrm{GeV}\)와 비교하면 차이는

\[
\frac{5\ \mathrm{GeV}}{2.3567938\ \mathrm{eV}}
=2.1215\times10^9
\]

배다. 다광자 누적이나 별도 가속기를 가정하려면 그 결합률·에너지 전달·배경
모형을 새로 유도하고 같은 장치에서 측정해야 한다. 현재 자료에는 그 중간단계가
없다.

논리적으로도

\[
A\to B,\qquad C\to D
\]

를 서로 다른 실험에서 확인했다고 해서

\[
B\to C
\]

가 증명되지는 않는다. 현재 실험은 각각

\[
\text{optical pump}\to\text{optical photon pairs}
\]

및

\[
\text{relativistic-ion EM field}\to e^+e^-
\]

를 확인한다. 두 번째 입력이 첫 번째 출력이라는 event-level 기록은 없다.

## 6. 실제 물질의 준안정상: 확인된 것과 막힌 것

`10.nxs`에서 확인한 intensity tensor는

\[
I(\theta,E,t_{12})\in\mathbb R^{256\times344\times45}
\]

이다. 두 펌프의 fluence는 각각 \(0.30\ \mathrm{mJ/cm^2}\)이므로

\[
F_{\rm total}=0.60\ \mathrm{mJ/cm^2},\qquad
\frac{F_{\rm total}}{F_{\rm crit}}=
\frac{0.60}{0.50}=1.20.
\]

논문 주파수 \(f=2.2\pm0.1\ \mathrm{THz}\)를 주기로 바꾸면

\[
T=\frac1f=454.5\pm20.7\ \mathrm{fs}.
\]

원자료 초기 시간 간격은 25 fs이므로 한 주기당 약 18.2점이다. 따라서 시간축
sampling은 이 진동을 분해하기에 충분하다. 10 kHz 반복률은 정확히
\(100\ \mu\mathrm s\)이고, 논문의 160 K 수명 상한 \({<}100\ \mu\mathrm s\)와
stroboscopic reset 설계도 수치상 맞는다.

그러나 exact raw-to-result refit은 실패했다. 이유는 단순히 신호가 약해서가
아니라 공개 provenance가 불충분하고 서로 충돌하기 때문이다.

| 항목 | 출처 A | 출처 B | 판정 |
|---|---:|---:|---|
| 고정 pump–probe delay | 본문 35 ps | supplement/`10.nxs` 25 ps | `CONFLICT` |
| 온도 | `summary.xlsx` 160 K | `10.nxs` 내부 20 K | `CONFLICT` |
| 정확한 Gaussian fit 초기값·경계·코드 | 미공개 | 미공개 | `MISSING` |
| acquisition-level lifetime curve | `10.nxs`에 없음 | 논문 진술만 존재 | `MISSING` |

따라서 raw 파일·fluence·시간축 확인은 `PASS`, 정확한 2.2 THz 재적합과 장기
수명 재적합은 `CONDITIONAL/NOT REPRODUCED`다.

더 중요한 의미 경계는 이것이 새 물질 “종”의 생성이 아니라는 점이다. 관측된
hidden state는 기존 Ta/S 원자와 전자 안에서 CDW domain wall, 적층 및 Mott
스펙트럼 weight가 재배열된 전자·구조 상이다. 응집물질 문맥의 “새로운
물질상”은 맞지만 새 원소·새 기본입자·새 질량 pole 또는 진공 물질생성을 뜻하지
않는다.

## 7. 루프 엔지니어링 판정

아래 값은 존재 확률이 아니라 공개자료 완전성·재현성에 대한 감사 점수다.

| Gate | 점수 | 상태 |
|---|---:|---|
| 광학 sideband/CAR/g2 공개표 재계산 | 0.90 | `PASS`, event raw 부재 |
| CMS 단면적 공개 bin 재적분 | 0.95 | `PASS`, event raw 부재 |
| 1T-TaS2 체크섬·fluence·시간축 감사 | 0.90 | `PASS` |
| 1T-TaS2 2.2 THz exact raw refit | 0.55 | `CONDITIONAL` |
| 1T-TaS2 장기수명 raw refit | 0.35 | `NOT REPRODUCED` |
| 같은 장치의 광자쌍→질량쌍 전환 | 0.00 | `NOT MEASURED` |
| Clarus 고유 coupling 측정 | 0.00 | `NOT DEFINED/MEASURED` |
| 새 입자·원자·질량 pole | 0.00 | `NO EVIDENCE` |
| 자유에너지·진공에너지 순추출 | 0.00 | `NO EVIDENCE` |

현재 최강의 정직한 결론은 다음과 같다.

> 공명·위상 선택적인 pump가 비고전적 광자쌍과 펌프 후 준안정 물질상이라는
> 서로 다른 실제 현상을 만든다. 또한 충분히 높은 전자기장 에너지는 표준 QED에
> 따라 질량 있는 입자쌍으로 변환된다. 하지만 세 현상을 하나의 Clarus 장이
> 매개했다는 공통 원인, 같은 장치의 연속 사슬, 새 물질 종은 관측되지 않았다.

## 8. 다음에 해야 할 실제 증명

검출식을 더 추가하는 것이 아니라 Clarus가 표준모형과 다르게 예측하는 수치 하나를
먼저 고정해야 한다. 최소 증명 경로는 다음이다.

1. CE/Clarus 작용에서 coupling과 line shape를 유도해, QED/SFWM/CDW 모형에
   없는 사전등록 잔차
   \(\Delta_i=y_i-y_i^{\rm standard}\)의 부호·크기·주파수를 고정한다.
2. 하나의 장치에서 pump-off, phase-scramble, thermal-matched, material-swap을
   무작위 blind run으로 수집한다.
3. 입력 pump energy, 광자 출력, 열, 운동량, 생성물 정지질량을 같은 run ID로
   묶어 signed energy ledger를 닫는다.
4. 질량분석·원소분석·산란으로 새 pole/조성 변화를 확인하고, 기존 상 재배열과
   detector/cavity memory를 배제한다.
5. 분석코드·event-level raw·공분산을 공개하고 held-out run에서 동일 효과를
   재현한다.

이 중 1번의 Clarus 고유 정량예측이 나오지 않으면 어떤 양성 신호도 기존 물리의
재명명과 구별할 수 없다. 현재 가장 먼저 풀 문제는 검출기 설계가 아니라
`표준 QED/비선형광학 대비 Clarus의 단 하나의 수치적 초과예측`이다.

## 9. 재현 명령

```powershell
uv --cache-dir .uv-cache run python `
  examples/physics/external_field_to_matter_reanalysis.py

uv --cache-dir .uv-cache run --extra dev python -m pytest `
  tests/test_external_field_to_matter.py -q
```

테스트는 snapshot hash 변조, CAR peak 이동, 고전적 control 변조, HEPData bin
누락, 모델열 변조, NaN/Inf/bool, NeXus shape와 digest 변조를 fail-closed로
거부한다.
