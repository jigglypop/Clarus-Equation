# 6.4 FLRW 우주

> **요지.** 크게 보면 균질하고 등방인 우주는 척도인자 $a(t)$ 하나로 기술된다. 팽창률 $H=\dot a/a$는 프리드만 방정식으로 성분의 밀도와 이어지고, 성분의 비율 $\Omega$가 우주의 역사와 나이를 정한다.

## 학습 목표

1. FLRW 계량과 척도인자, 적색편이의 관계를 안다.
2. 프리드만 방정식과 밀도 모수 $\Omega$를 쓸 수 있다.
3. 평탄 ΛCDM 우주의 나이를 계산할 수 있다.
4. 드 시터 극한과 진공 팽창률 $H_\Lambda$를 안다.
5. 상태방정식 $w$와 밀도의 시간 변화를 연결할 수 있다.

## 선수 지식

[6.1절](01_일반상대론_요약.md), 적분.

## 본문

### FLRW 계량

$$ds^2=-c^2dt^2+a(t)^2\Big[\frac{dr^2}{1-kr^2}+r^2d\Omega^2\Big] . \tag{6.4.1}$$

$k=0$이면 공간이 평탄하다. 오늘 $a(t_0)=1$로 둔다. 멀리서 온 빛은 파장이 늘어나 적색편이 $z$를 갖고, $1+z=1/a(t_{\rm 방출})$이다.

### 프리드만 방정식

아인슈타인 방정식을 (6.4.1)에 넣으면

$$H^2=\Big(\frac{\dot a}a\Big)^2=\frac{8\pi G}3\rho-\frac{kc^2}{a^2}+\frac{\Lambda c^2}3 . \tag{6.4.2}$$

**정의 6.4.1 (밀도 모수).** 임계 밀도 $\rho_c=3H_0^2/8\pi G$에 대한 비로 $\Omega_i=\rho_{i,0}/\rho_c$를 정한다. 오늘 $\Omega_m+\Omega_r+\Omega_\Lambda+\Omega_k=1$이고, 평탄하면 $\Omega_k=0$이다. (6.4.2)는

$$\frac{H(z)^2}{H_0^2}=E(z)^2=\Omega_m(1+z)^3+\Omega_r(1+z)^4+\Omega_k(1+z)^2+\Omega_\Lambda . \tag{6.4.3}$$

### 상태방정식

압력 $p=w\rho c^2$인 성분은 에너지 보존 $\dot\rho+3H(\rho+p/c^2)=0$에서 $\rho\propto a^{-3(1+w)}$이다. 먼지 $w=0$($a^{-3}$), 복사 $w=\tfrac13$($a^{-4}$), 진공 $w=-1$(상수)다. $w$가 시간에 따라 바뀌는 암흑에너지는 $w(z)$로 쓴다. $w<-1$이면 "팬텀"으로, 널 에너지 조건을 어긴다.

### 우주의 나이

**정리 6.4.1.** 평탄 우주에서 먼지와 진공만 있으면($\Omega_m+\Omega_\Lambda=1$)

$$t_0=\frac{2}{3H_0\sqrt{\Omega_\Lambda}}\,\mathrm{arsinh}\sqrt{\frac{\Omega_\Lambda}{\Omega_m}} . \tag{6.4.4}$$

*증명.* $\dot a=H_0\sqrt{\Omega_ma^{-1}+\Omega_\Lambda a^2}$를 $a$에 대해 적분한다. $x=a^{3/2}$로 바꾸면 $t=\frac2{3H_0}\int_0^{1}\frac{dx}{\sqrt{\Omega_m+\Omega_\Lambda x^2}}$이 되어 (6.4.4)를 얻는다. ∎

### 드 시터 극한

먼 미래에 먼지가 묽어지면 $H\to H_\Lambda=H_0\sqrt{\Omega_\Lambda}$로 일정해지고 $a\propto e^{H_\Lambda t}$다(드 시터). 관측자에게는 반지름 $c/H_\Lambda$의 사건 지평선이 생기고, 그 온도는 $\hbar H_\Lambda/2\pi k_B$다(6.2절).

$H_\Lambda t$는 무차원 "위상"이다. 오늘의 값은 (6.4.4)에서 $H_\Lambda t_0=\tfrac23\mathrm{arsinh}\sqrt{\Omega_\Lambda/\Omega_m}$로, $H_0$와 무관하고 $\Omega_m$만의 함수다. 렌더링 이론은 이 위상의 절반 $H_\Lambda t/2$를 시간축의 기울기로 읽는다(14장).

## 예제

**예제 6.4.1** $h=0.67772$, $\Omega_m=0.30796$(평탄, 먼지+진공)에서 우주의 나이를 구하라.

*풀이.* $H_0=67.772$ km s⁻¹ Mpc⁻¹, $1/H_0=14.43$ Gyr. $\Omega_\Lambda=0.69204$, $\sqrt{\Omega_\Lambda/\Omega_m}=1.4991$, $\mathrm{arsinh}=1.1942$. $t_0=\frac{2}{3\times0.83189}\times14.43\times1.1942=13.81$ Gyr.

**예제 6.4.2** 같은 값에서 $H_\Lambda$, $c/H_\Lambda$, $H_\Lambda t_0$를 구하라.

*풀이.* $H_\Lambda=67.772\times0.83189=56.38$ km s⁻¹ Mpc⁻¹. $c/H_\Lambda=299792/56.38=5317$ Mpc. $H_\Lambda t_0=\tfrac23\times1.1942=0.7962$. 그 절반은 $0.398$로 $\pi/8=0.393$과 1.4% 다르다.

## 연습문제

**연습 6.4.1** ★ $z=1$인 빛이 나올 때 우주의 크기는 오늘의 몇 배였는가?

**연습 6.4.2** ★ $w=-\tfrac23$인 성분의 밀도는 $a$의 몇 제곱에 비례하는가?

**연습 6.4.3** ★ Planck 2018($h=0.6736$, $\Omega_m=0.3153$)에서 우주의 나이를 (6.4.4)로 구하라.

**연습 6.4.4** ★★ (6.4.4)를 유도하라.

**연습 6.4.5** ★★ $H_\Lambda t_0=\pi/4$가 되려면 $\Omega_m$은 얼마여야 하는가?

**연습 6.4.6** ★★ 에너지 보존 $\dot\rho+3H(\rho+p)=0$에서 $\rho\propto a^{-3(1+w)}$를 유도하라($c=1$).

**연습 6.4.7** ★★★ 파이썬으로 평탄 ΛCDM의 공변 거리 $D_M(z)=\frac c{H_0}\int_0^z\frac{dz'}{E(z')}$를 계산하라. $h=0.67772$, $\Omega_m=0.30796$에서 $D_M(1)$은?

## 요약

- FLRW 우주는 척도인자 하나로 기술되고, $1+z=1/a$.
- 프리드만 방정식 $H^2/H_0^2=\sum\Omega_i(1+z)^{3(1+w_i)}$.
- 평탄 먼지+진공 우주의 나이는 (6.4.4)이고, 위상 $H_\Lambda t_0$는 $\Omega_m$만의 함수다.

## 이 절의 수치와 재현

| 수치 | 값 | 재현 |
|---|---|---|
| 나이, $H_\Lambda t_0$ | 13.81 Gyr, 0.7962 | `ce_rendering_cycle.cycle_geometry()` |
| $H_\Lambda$, $c/H_\Lambda$ | 56.38, 5317 Mpc | 같음 |
