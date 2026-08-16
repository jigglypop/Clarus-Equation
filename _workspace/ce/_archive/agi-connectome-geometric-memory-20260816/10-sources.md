# Connectome–Geometric Memory 외부 근거 감사

Status: COMPLETE

검증일: 2026-08-16 (Asia/Seoul)

범위: 사용자 노트가 인용한 6개 논문과 공식 데이터 경로, 그리고 이 연구계약의 식별가능성·제어비용·neural manifold·attractor/accessibility·reconsolidation에 직접 닿는 최소 1차 근거만 확인했다. 아래에서 `개입`은 단순 감각자극/과제조건과 **기록된 신경 노드 또는 회로에 대한 조작**을 구분한다. 논문이 공개되어 있다는 사실과 원시 데이터가 같은 라이선스로 공개되어 있다는 사실도 구분한다.

## 1. 핵심 판정

1. **인용한 논문 6개의 서지정보는 모두 실재하며, human cortex 논문의 누락 DOI는 `10.1126/science.adk4858`이다.** 나머지 다섯 DOI는 사용자 노트와 일치한다.
2. **MICrONS는 blind anatomical structure/function benchmark로는 조건부 사용 가능하지만, blind causal-connectome recovery의 ground truth는 아니다.** 동일한 한 마우스에서 시각 자극에 대한 흥분성 뉴런 칼슘활동과 사후 EM 구조를 정합했다. 그러나 전체 세포를 동시에 기록하지 않았고, 기록 활동은 흥분성 세포에 한정되며, 신경 노드 자극/lesion rollout이 없고, 관측 부피 밖 입력과 억제성·잠재상태가 남는다. EM synapse edge는 효과적 인과 edge와 동일하지 않다.
3. **MICrONS에는 기억 학습–부분단서–지연회상의 longitudinal trajectory가 없다.** P75–P81의 14개 순차 시각피질 scan과 P87 사후 EM은 학습 전/후 기억 실험이 아니다. 같은 마우스의 반복 scan을 여러 독립 동물로 셀 수 없다.
4. **Science 2025 engram EM은 기억과 연관된 말단·시냅스·소기관 구조 변화를 지지하지만, 동일 개체 pre/post EM 또는 부분단서 회상 동역학을 제공하지 않는다.** SBEM은 파괴적 종말점 측정이다. 조건화·태깅은 실험 처치이지만, 보고된 데이터는 신경 노드 개입 후 population rollout을 기록한 자료가 아니다.
5. **따라서 인용된 어느 한 데이터셋도 `anatomy + 동시/longitudinal dynamics + learning + neural intervention + partial-cue recall` 전체 체인을 한 번에 검증하지 못한다 (`P1`).** 구조–기능 benchmark와 geometric-memory benchmark를 분리해야 한다.
6. Hu et al.은 학습 후 neural-manifold separation과 행동의 연결을 직접 보고하지만, 2026-08-16 현재도 **preprint**이고 명백한 attractor signature는 찾지 못했다. 따라서 이 결과는 geometry 관련 가설(H8 후보)의 직접 근거이지 attractor 재생(H9 후보)의 근거가 아니다.

## 2. 사용자 인용 6건 검증

필드 표기: `A` anatomy/connectivity, `X` neural activity, `S` stimulus/task input, `B` behavior, `N-I` recorded neural intervention, `M` learning/recall protocol.

| ID | 1차 출처·DOI | 확인된 규모와 독립 통계단위 | 실제 제공 필드·개입 | 접근·라이선스 | 이 계약에 대한 판정 |
|---|---|---|---|---|---|
| `SRC-MIC-RESOURCE` | MICrONS Consortium, *Functional connectomics spanning multiple areas of mouse visual cortex*, Nature 640, 435–447 (2025), [`10.1038/s41586-025-08790-w`](https://doi.org/10.1038/s41586-025-08790-w) | **한 마우스**. 14개 2P scan, 115,372 functional units, 추정 75,909 unique excitatory neurons; EM에는 >200,000 cells와 약 0.5 billion synapses. Explorer v1300은 manual matches 19,181개, unique EM neurons 15,439개를 명시한다. 생물학적 독립단위는 mouse=1이며 frame·cell·edge·scan은 동물 복제가 아니다. | `A,X,S,B`; 자연/렌더/parametric video, treadmill, eye/pupil. `N-I` 없음. X는 GCaMP6s 흥분성 뉴런이고 14 scan 전체가 동시에 관측된 population은 아니다. `M` 없음. | [공식 cortical-mm3](https://www.microns-explorer.org/cortical-mm3), BossDB/CAVE/공개 cloud bucket; functional MySQL dump 97 GB. Explorer 자료는 [CC BY 4.0](https://www.microns-explorer.org/terms-and-conditions); [citation policy](https://www.microns-explorer.org/citation-policy). CAVE dynamic query는 Google login/ToS가 필요할 수 있다. | 구조–기능 상관 및 **숨긴 EM anatomical edge 예측**에는 조건부 적합. 정확 causal graph, held-out neural-intervention NLL, memory geometry에는 부적합. `P0` if EM=`G_causal`; `P1` single mouse/partial observation/no intervention/no memory. |
| `SRC-MIC-DING` | Ding et al., *Functional connectomics reveals general wiring rule in mouse visual cortex*, Nature 640, 459–469 (2025), [`10.1038/s41586-025-08840-3`](https://doi.org/10.1038/s41586-025-08840-3) | 위와 같은 한 마우스. joint subvolume 약 560×1,100×500 μm; 82,247 nuclei, 그중 intersection의 excitatory 43,679; manually matched excitatory 13,952. 최종 고정밀 graph는 functionally characterized presynaptic 148개와 postsynaptic partners 4,811개. 논문도 neuron-pair 분석이 한 마우스임을 명시한다. 별도 3 mice digital-twin validation은 connectome biological replicate가 아니다. | `A,X,S,B`; 실제 neural perturbation 없음. digital twin의 in-silico stimulus exploration은 empirical node intervention이 아니다. | [Nature 원문](https://www.nature.com/articles/s41586-025-08840-3), [BossDB DOI](https://doi.org/10.60533/BOSS-2021-T0SY), [analysis code tag 1.0.0](https://github.com/cajal/microns-funconn-2025/tree/1.0.0). 논문/Explorer CC BY 4.0. | like-to-like wiring과 structure/function association의 1차 근거. causal identification이나 기억 근거로 승격 불가. 선택적으로 proofread한 148 presynaptic cells는 full-volume causal truth가 아니다. |
| `SRC-ENGRAM` | Uytiepo et al., *Synaptic architecture of a memory engram in the mouse hippocampus*, Science 387, eado8316 (2025), [`10.1126/science.ado8316`](https://doi.org/10.1126/science.ado8316) | CA3→CA1 SBEM, fear-conditioned `n=3 mice`, neutral conditioned-stimulus comparison `n=2 mice`. 분석단위는 synapse가 아니라 mouse여야 한다. | `A,M`; FosDD-Cre/APEX2-mGFP activity-history tagging, AAV/TMP, contextual fear conditioning/foot shock. 이는 처치이지만 표적 neural-node perturb-and-record rollout은 아니다. dendrite/spine, Schaffer collateral axon/terminal, PSD, multi-synaptic bouton, mitochondria, smooth ER, astrocyte contacts를 계량한다. 동일 개체 pre/post X, partial-cue recall trajectory 없음. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/40112060/), [공개 author manuscript/associated Data S1–S5](https://pmc.ncbi.nlm.nih.gov/articles/PMC12233322/). `UNVERIFIED`: 원 논문/PMC의 data statement에서 전체 raw SBEM volume repository와 그 명시적 재사용 라이선스를 식별하지 못했다. 공개 보조표는 약 0.19–8.3 MB, methods PDF 약 33.3 MB이나 raw volume 총량은 확인되지 않았다. | 기억 관련 구조적 종말점/동기 근거. “engram끼리 edge가 단순 증가”보다는 MSB와 input-specific synaptic/organelle remodeling이라는 노트의 절제된 설명은 지지. longitudinal geometry/causal recall benchmark로 쓰면 `P0`; raw 접근·license는 `P1`. |
| `SRC-COGITATE` | Cogitate Consortium et al., *Adversarial testing of global neuronal workspace and integrated information theories of consciousness*, Nature 642, 133–142 (2025), [`10.1038/s41586-025-08888-1`](https://doi.org/10.1038/s41586-025-08888-1) | 총 256 participants: fMRI 120, MEG 102, iEEG 34; modality별 여러 실험실. 독립단위 participant. | `X,S,B`; suprathreshold stimulus의 category/identity/orientation/duration 및 target-detection task. 구조 connectome truth와 `N-I` 없음. | [Nature 원문](https://www.nature.com/articles/s41586-025-08888-1), [preregistration DOI](https://doi.org/10.17605/OSF.IO/92TBG), [raw/BIDS bundles](https://www.arc-cogitate.com/data-bundles), [XNAT](https://cogitate-data.ae.mpg.de), [documentation](https://cogitate-consortium.github.io/cogitate-data/). 데이터 CC BY 4.0, task/analysis code MIT. `UNVERIFIED`: bundle별 정확 download size는 공식 Data availability 본문에 없다. | IIT와 GNWT의 핵심 예측 양쪽을 일부 도전한 경계근거. SCC 또는 recurrence가 의식의 충분조건이라는 주장을 지지하지 않는다. CGM 구조·기억 검증용 데이터가 아니다. |
| `SRC-H01` | Shapson-Coe et al., *A petavoxel fragment of human cerebral cortex reconstructed at nanoscale resolution*, Science 384, eadk4858 (2024), **[`10.1126/science.adk4858`](https://doi.org/10.1126/science.adk4858)** | 한 epilepsy-surgery donor의 human temporal cortex 약 1 mm³; 약 57,000 cells, 230 mm vasculature, 약 150 million synapses, 1.4 PB. 독립단위 donor/specimen=1. | `A` only; 4×4×33 nm EM, masks, segmentations, cells, layer/vascular/synapse annotations. X/S/B/N-I/M 없음. | [공식 H01 release](https://h01-release.storage.googleapis.com/data.html), Google Cloud/Neuroglancer/TensorStore; released datasets CC BY 4.0. [공개 논문 원고](https://pmc.ncbi.nlm.nih.gov/articles/PMC11718559/). | 인간 구조 stress-test/방법 근거. human dynamics, memory, causal identification 또는 population generalization 근거 아님. |
| `SRC-LICONN` | Tavakoli et al., *Light-microscopy-based connectomic reconstruction of mammalian brain tissue*, Nature 642, 398–410 (2025), [`10.1038/s41586-025-08985-1`](https://doi.org/10.1038/s41586-025-08985-1) | cortical overview 약 396×109×22 μm = 0.95×10^6 μm³ native tissue, 약 16× expansion. CA1 fully proofread box 85×69×14 μm = 83,825 μm³, 68.6 gigavoxels. 여러 기술 replicate가 있으나 comprehensive proofread graph는 1 specimen이며 주요 connectivity analysis는 한 mouse의 imaging volumes이다. | `A` 및 molecular labels (예: bassoon, SHANK2); X/S/B/N-I/M 없음. 고정조직 acquisition method이다. | [Nature 원문](https://www.nature.com/articles/s41586-025-08985-1), Neuroglancer, [ISTA data DOI](https://doi.org/10.15479/AT:ISTA:18697), [code](https://github.com/danzllab/LICONN). **논문은 CC BY 4.0이나 ISTA data files는 CC BY-NC-SA 4.0**이다. repository가 열거한 단일 파일은 최대 2.24 GB; 전체 Neuroglancer corpus 총량은 `UNVERIFIED`(공식 페이지에 합계 없음). | molecularly informed light-microscopy connectomics의 방법 근거. 현 데이터는 동역학/기억/인과 benchmark가 아니다. 상업적 재사용 가능성을 논문 license만 보고 판단하면 안 된다 (`P1`). |

## 3. MICrONS blind benchmark 적합성

| 질문 | 판정 | 근거/필수 제한 |
|---|---|---|
| activity·stimulus·behavior를 보고 EM anatomy를 숨긴 뒤 anatomical edge를 맞힐 수 있는가? | **조건부 가능** | v1300 materialization, matching table, proofread subset, volume boundary, cell inclusion을 먼저 freeze한다. directed synapse existence/AUPRC, calibration, degree/distance-matched baseline처럼 **anatomical association** endpoint로 명명한다. 누설 방지를 위해 EM-derived morphology·cell type·distance를 입력에서 명시적으로 제외/분리한다. |
| 이것이 exact causal connectome recovery인가? | **아니다 (`P0`)** | 구조 synapse와 effective causal edge가 같지 않다. 전체 population 동시기록이 아니고 흥분성 활동만 기록되며, 억제/외부부피/neuromodulation/공통 자극/latent state가 남는다. 단일 observational animal로 일반 nonlinear recurrent graph를 유일하게 식별할 수 있다는 정리는 없다. |
| held-out neural intervention prediction을 empirical primary endpoint로 둘 수 있는가? | **아니다 (`P0`)** | 자연·합성 시각자극은 관측된 exogenous input이지 특정 뉴런/모듈 자극 또는 lesion이 아니다. digital-twin in-silico intervention은 실제 intervention ground truth가 아니다. MICrONS에서는 held-out **stimulus-response** prediction과 anatomical edge prediction을 별도 endpoint로 둔다. |
| biologically independent train/validation/test split이 가능한가? | **아니다 (`P1`)** | mouse=1. scan/stimulus/neuron/edge split은 같은 동물 내부의 기술적/조건부 일반화만 측정한다. biological generalization은 향후 별도 animals/connectomes가 필요하다. |
| learning/memory pre/post 및 partial-cue recall을 검증할 수 있는가? | **아니다 (`P1`)** | memory learning, pre/post same-circuit activity, interference, delayed recall, partial-cue trajectory가 없다. repeated visual scans를 memory longitudinal experiment로 재해석하면 안 된다. |

실행 시 전체 EM을 내려받지 말고 CAVE/BossDB의 고정 snapshot에서 필요한 matched/proofread subgraph만 query한다. functional DB만 공식 배포본이 97 GB이며, full EM corpus의 단일 download 총량은 공식 inspected page에서 `UNVERIFIED`이다. v1300, `coregistration_manual_v4`, code tag/commit, query timestamp, root IDs, proofread status를 manifest에 고정해야 한다 (`P2`).

## 4. 한 데이터셋으로 전체 체인을 검증할 수 있는가

**없다.** 최소한 다음 세 단계로 분리한다.

| 단계 | 적합 자료 | 검증 가능한 주장 | 검증하지 못하는 주장 |
|---|---|---|---|
| A. causal identifiability | ground-truth graph·latent state·single-node/module intervention을 생성하는 synthetic recurrent systems | 조건별 `G/F/z/g` 식별가능성, intervention rollout, counterexample | 생물학적 기억/AGI 자체 |
| B. structure/function bridge | MICrONS frozen matched/proofread subset | observational function에서 anatomical synapse/graph feature 예측, 구조–기능 상관 | exact causal graph, biological replication, learning/recall |
| C. geometric-memory | 별도의 longitudinal learning/partial-cue/interference/delayed-recall population dataset 또는 새 실험 | pre/post geometry, cue→recall path, basin/access cost, behavior prediction | synapse-level ground truth unless 별도 correlative anatomy가 추가됨 |

Science engram EM은 B와 C 사이의 **구조적 동기/종말점 근거**로 쓸 수 있지만 C의 동역학 dataset을 대체하지 못한다. MICrONS와 engram 결과를 서로 다른 동물·뇌영역·측정법에서 이어 붙여 동일 개체의 causal chain인 것처럼 통계 검정해서는 안 된다 (`P1`).

## 5. 식별가능성의 최소 1차 근거

| 1차 출처 | 실제로 보이는 것 | 이 계약의 경계 |
|---|---|---|
| Hyvärinen, Sasaki & Turner, *Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning*, AISTATS 2019, [PMLR 89](https://proceedings.mlr.press/v89/hyvarinen19a.html) | time/history 등 관측 auxiliary variable과 특정 생성모형 조건 아래 nonlinear ICA identifiability | 수동 calcium activity만으로 arbitrary latent dynamics/graph가 자동 식별된다는 결과가 아니다. |
| Khemakhem et al., *Variational Autoencoders and Nonlinear ICA: A Unifying Framework*, AISTATS 2020, [PMLR 108](https://proceedings.mlr.press/v108/khemakhem20a.html) | observed auxiliary variable에 조건화된 factorized latent prior 등 명시적 조건에서 단순 변환까지 식별 | MICrONS가 그 조건을 만족한다는 근거가 아니다. |
| Ahuja et al., *Interventional Causal Representation Learning*, ICML 2023, [PMLR 202](https://proceedings.mlr.press/v202/ahuja23a.html) | 논문이 정의한 intervention setting에서 latent causal factors의 identifiability | Phase A의 intervention 설계를 정당화하지만, intervention 없는 MICrONS를 승격하지 않는다. |
| Yang, Katcoff & Uhler, *Characterizing and Learning Equivalence Classes of Causal DAGs under Interventions*, ICML 2018, [PMLR 80](https://proceedings.mlr.press/v80/yang18a.html) | intervention target 집합에 따른 interventional Markov equivalence class | 제한된 intervention도 일반적으로 exact DAG 보장을 자동 제공하지 않는다. |

업데이트 권고: `CGM-N1`의 반례·동치류 경계를 유지한다. Phase A 성공 기준은 “정확 graph” 하나가 아니라 intervention 조건별 identifiable object(정확 edge, equivalence class, predictive effective graph)를 사전 선언하는 것이다.

## 6. geometric memory를 위한 직접 근거와 안전한 operationalization

### 6.1 제어 Gramian과 접근 비용

선형 기준모형 `dx/dt = Ax + Bu`에서 finite-horizon controllability Gramian과 minimum input energy를

`W_T = integral_0^T exp(A t) B B^T exp(A^T t) dt`,

`E*(x0→xT) = (xT-exp(A T)x0)^T W_T^{-1}(xT-exp(A T)x0)`

로 둘 수 있다(가제어·비특이 `W_T` 조건; 그렇지 않으면 regularization/pseudoinverse와 reachability 판정을 분리). 이는 `A,B,T`가 고정된 선형계에서의 **task-dependent quadratic accessibility baseline**이지, 기억이 보편적으로 Riemannian metric이라는 증명이 아니다.

| 1차 출처 | 직접 근거 | 제한 |
|---|---|---|
| Gu et al., *Controllability of structural brain networks*, Nature Communications 6, 8414 (2015), [`10.1038/ncomms9414`](https://doi.org/10.1038/ncomms9414) | 8 healthy adults, 3 DSI scans, 234-region structural networks에 linear network control/Gramian 적용 | smallest Gramian eigenvalue가 매우 작고 scan 간 재현되지 않았음을 논문이 직접 보고한다. mathematical energy를 생물학적 ATP/effort로 동일시하면 안 된다. |
| Betzel et al., *Optimally controlling the human connectome: the role of network topology*, Scientific Reports 6, 30770 (2016), [`10.1038/srep30770`](https://doi.org/10.1038/srep30770) | 8 cognitive-system states 사이 56 directed transitions에 minimum-energy input을 계산 | 상태가 관측된 기억 engram이 아니라 사전 정의된 systems이며 선형모형/정규화/시간지평에 의존한다. |
| Cornblath et al., *Temporal sequences of brain activity at rest are constrained by white matter structure and modulated by cognitive demands*, Communications Biology 3 (2020), [`10.1038/s42003-020-0961-x`](https://doi.org/10.1038/s42003-020-0961-x) | 실제 fMRI에서 추출한 rest/n-back brain states 사이의 minimum control energy와 관측 transition probability를 연결 | memory recall/partial cue 실험은 아니다. empirical state accessibility baseline으로만 사용한다. |

계약의 geometry endpoint에는 `E*`, basin-entry time, hitting probability를 Euclidean/weight-only/activity-only baseline과 함께 넣고, `A,B,T`, 안정화 상수, 상태 정규화, Gramian condition number를 반드시 기록한다. `W_T^{-1}`을 learned memory metric으로 부를 때는 **linear-control-derived metric candidate**라고 한정한다 (`P1` if universal memory metric).

### 6.2 학습과 neural-manifold geometry: H8와 H9 분리

| 1차 출처 | 직접 근거 | 판정 |
|---|---|---|
| Sadtler et al., *Neural constraints on learning*, Nature 512, 423–426 (2014), [`10.1038/nature13665`](https://doi.org/10.1038/nature13665) | 2 rhesus macaques BCI에서 within-manifold perturbation이 outside-manifold perturbation보다 수 시간 내 훨씬 잘 학습됨 | neural population geometry가 learnability를 제약한다는 직접 인과적 조작. motor BCI/선형 저차원 manifold이지 장기기억 metric deformation은 아니다. |
| Nieh et al., *Geometry of abstract learned knowledge in the hippocampus*, Nature 595, 80–84 (2021), [`10.1038/s41586-021-03652-7`](https://doi.org/10.1038/s41586-021-03652-7) | learned task에서 hippocampal population representation의 geometry를 직접 계량 | learned representational geometry 근거이지 recall basin/causal metric 근거는 아니다. |
| Hu et al., *Representational learning by optimization of neural manifolds in an olfactory memory network*, bioRxiv (2024), [`10.1101/2024.11.17.623906`](https://doi.org/10.1101/2024.11.17.623906), [PubMed preprint record](https://pubmed.ncbi.nlm.nih.gov/39605658/) | zebrafish pDp 2P activity: trained fish N=25, naïve N=6. 훈련이 task-relevant odor manifolds의 capacity/separation을 높였고 fish-level discrimination behavior를 예측. radius/dimension/center alignment 등 여러 geometric feature를 계량 | **2026-08-16 현재 preprint**. fish가 독립단위이며 odor-pair/timepoint는 nested sample이다. same-fish pre/post, partial cue, delayed recall이 아니다. 논문은 “no obvious signatures of attractor dynamics”를 보고하므로 H8 geometry 후보를 지지하지만 H9 attractor regeneration은 지지하지 않는다. `P1` preprint. |

Hu et al.의 현재 출판상태는 [PubMed 공식 record](https://pubmed.ncbi.nlm.nih.gov/39605658/)가 `[Preprint]`로 표시하고 DOI가 bioRxiv 원고로 연결되는 것으로 재확인했다. Research Square의 동일 제목 record도 peer-reviewed journal article이 아니라 preprint다. 따라서 최종 문서에서 “동료심사 완료 논문”으로 인용하지 않는다.

### 6.3 attractor access, engram recall, reconsolidation

| 1차 출처 | 직접 근거 | 허용되는 해석 |
|---|---|---|
| Jezek et al., *Theta-paced flickering between place-cell maps in the hippocampus*, Nature 478, 246–249 (2011), [`10.1038/nature10439`](https://doi.org/10.1038/nature10439) | rat CA3에서 cue/environment switch 뒤 미리 형성된 두 place-cell map 사이 theta-paced flicker | competing attractor-like maps와 cue-driven switching의 직접 population evidence. 학습된 Riemannian metric이나 partial-cue memory regeneration의 완전한 증명은 아니다. |
| Liu et al., *Optogenetic stimulation of a hippocampal engram activates fear memory recall*, Nature 484, 381–385 (2012), [`10.1038/nature11028`](https://doi.org/10.1038/nature11028) | tagged dentate-gyrus engram의 optogenetic activation이 freezing recall을 유발 | 기억 접근에 대한 causal sufficiency 근거. 직접 광자극은 자연 partial cue와 다르며 geometry를 측정하지 않는다. |
| Ryan et al., *Engram cells retain memory under retrograde amnesia*, Science 348, 1007–1013 (2015), [`10.1126/science.aaa5542`](https://doi.org/10.1126/science.aaa5542), [공개 원고](https://pmc.ncbi.nlm.nih.gov/articles/PMC5583719/) | protein-synthesis inhibition 후 자연 회상이 손상되어도 tagged engram의 광자극으로 회상 가능 | 저장 소실과 자연적 접근 실패를 분리할 동기. metric/basin 설명은 후속 가설이다. |
| Nader, Schafe & LeDoux, *Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval*, Nature 406, 722–726 (2000), [`10.1038/35021052`](https://doi.org/10.1038/35021052) | retrieval 직후 amygdala anisomycin이 이후 기억을 손상시키고, 비재활성화/지연 투여 대조와 구분됨 | recall 후 상태 update/reconsolidation endpoint의 직접 동기. “기억=metric update”를 특정하지 않는다. |

따라서 `cue → recurrence → behavior`와 `recall 후 persistent update`는 독립적으로 검정할 수 있지만, 위 결과를 합쳐 보편적인 `memory = metric` 정리로 쓰지 않는다.

## 7. P0/P1/P2 영향 원장

| 우선순위 | 영향 | 조치 |
|---|---|---|
| `P0` | MICrONS EM edge를 causal edge ground truth로 부르는 경우 | endpoint를 `anatomical-edge prediction`과 `stimulus-response prediction`으로 개명·분리. causal intervention endpoint는 synthetic/새 개입자료로 이동. |
| `P0` | Science engram EM을 동일 개체 longitudinal geometry/partial-cue trajectory로 기술하는 경우 | cross-sectional destructive structural endpoint로 내림. dynamics/recall dataset을 별도 선정. |
| `P0` | Cogitate를 SCC/recurrence 충분조건 또는 의식 승리 이론의 근거로 사용하는 경우 | boundary evidence로만 유지; `CGM-X1` 제외 유지. |
| `P0` | neural manifold/control energy 결과를 “기억은 Riemannian metric이다”의 증명으로 사용하는 경우 | empirical association/operational baseline으로만 유지. geometry, attractor, accessibility를 분리 검정. |
| `P1` | 한 인용 dataset으로 전체 chain을 검증할 수 없음 | Phase A synthetic, Phase B MICrONS, Phase C longitudinal memory로 분리. cross-dataset chain은 개념적 triangulation이지 동일 개체 causal test가 아님을 명시. |
| `P1` | MICrONS/H01/LICONN graph의 biological unit가 각 1 mouse/donor/specimen | edge·cell·frame bootstrap으로 biological CI를 만들지 않음. 자료 내부 결과와 생물학적 일반화를 분리. |
| `P1` | engram raw SBEM repository/license, Cogitate bundle sizes, LICONN 전체 corpus 크기 미확인 | `UNVERIFIED` 유지. 실제 대용량 실행 전 repository owner/manifest로 재확인. |
| `P1` | Hu et al. peer-review 미완료 | preprint로 명시하고 preregistered replication/peer-reviewed 대안과 함께 취급. H8와 H9를 분리. |
| `P1` | LICONN data는 CC BY-NC-SA 4.0 | 논문 CC BY 4.0과 혼동 금지; downstream/commercial workflow에서 별도 법적 검토. |
| `P2` | mutable segmentation/matches·code 환경 | MICrONS v1300/table/root IDs/query date, BossDB snapshot, Ding tag 1.0.0, package/container digest와 checksums를 dataset manifest에 고정. |

## 8. 출처 접근기록과 업데이트 권고

위 모든 URL의 마지막 접근일은 **2026-08-16**이다. 1차 논문 페이지, PubMed/PMC 공개 원고, 논문이 지정한 공식 dataset/repository, 공식 라이선스 페이지만 판정 근거로 사용했다. 검색결과의 보도자료·review는 최종 판정 근거에서 제외했다.

정본 반영 권고:

- 사용자 노트의 “실제 구조 G와 동역학 X(t)을 같은 시스템에서 비교”는 **한 마우스의 부분관측 structure/function association**으로 제한하면 유지 가능하다.
- “MICrONS blind recovery가 가장 강한 실험”은 **anatomical-edge bridge benchmark**로 유지하되 `intervention prediction`과 `causal connectome` 문구는 Phase A/새 실험으로 이동한다.
- “세 축을 MICrONS 같은 데이터로 서로 연결”은 MICrONS 단독이 아니라 **분리된 datasets를 통한 단계적 triangulation**으로 고친다.
- geometric memory의 첫 empirical target은 보편적 curvature가 아니라 (i) trained-vs-naïve manifold separation, (ii) cue-to-state minimum control cost/basin entry, (iii) behavior/recall 예측의 incremental value다. attractor convergence와 reconsolidation update는 각각 별도 endpoint로 둔다.
