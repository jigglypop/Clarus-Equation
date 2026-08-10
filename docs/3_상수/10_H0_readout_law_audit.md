# \(H_0\) readout 재현성 노트 **[미완성]**

## 현재 범위

조건부 FLRW·de Sitter 항등식은
[우주론 수식의 의미와 형식 출처](9_우주론_수식_의미와_후보.md)에 둔다.
이 문서는 관측 readout의 재현 조건만 기록한다.

## 조건부 FLRW scaling

**[정의]**

\[
E(z):=\frac{H(z)}{H_0}.
\]

**[정리]** dimensionless expansion history \(E(z)\), curvature branch와
redshift를 고정하면 FLRW의 모든 배경 거리에는

\[
D(z_1,z_2)=\frac{c}{H_0}\,
\mathcal D(z_1,z_2;E,\Omega_K)
\]

형태의 공통 \(H_0^{-1}\) scale이 붙는다. 이는 null-geodesic 거리
적분에서 \(H(z)=H_0E(z)\)를 인수분해하면 바로 따른다.

따라서 고정된 shape \(E(z)\)에서 절대거리 자료는 \(H_0\)를 정할 수
있지만, \(E(z)\), calibration 또는 lens/source model을 함께 바꾸면
degeneracy가 생길 수 있다. 이 조건부 scaling은 원자료가 없어도 성립하는
이론적 부분이다.

현재 checkout에는 \(\texttt{examples/physics/h0_readout/}\) 디렉터리가 없다.
따라서 이전 실행 결과, covariance/Fisher 변환, source-role 선택 또는 기계
판정은 여기서 재현할 수 없다. 관측 결론은 **[미완성]**이다.

## 누락 의존성

- \(\texttt{h0_real_covariance_targets.json}\): source URI, 버전, checksum,
  변수 순서와 단위가 고정된 manifest
- \(\texttt{h0_real_source_scout_gate.py}\): manifest와 원자료의 연결 검사
- 공개 원자료의 mean vector 및 covariance/Fisher matrix
- local/global 역할을 자료를 보기 전에 정하는 role map

## 재개 조건

1. 원자료의 영구 식별자와 라이선스를 기록한다.
2. 행·열 라벨, 단위, fiducial cosmology와 변수 변환을 고정한다.
3. covariance의 대칭성·양의 정부호 또는 Fisher의 가역성을 검산한다.
4. 같은 likelihood로 기준모형과 CE readout을 비교한다.
5. 실행 명령, 환경과 산출물 checksum을 함께 보존한다.
