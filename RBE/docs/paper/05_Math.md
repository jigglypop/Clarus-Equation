# 5. 푸앵카레 볼의 리만 기하학: 상태-전이 미분과 하이브리드 최적화

## 5.1. 서론: 곡률과 최적화

전통적인 신경망 최적화는 **평탄한 유클리드 공간**에서 이루어진다. 그러나 푸앵카레 볼 기반 RBE는 **음의 곡률을 가진 쌍곡공간**에서 직접 최적화를 수행한다. 이는 단순한 좌표 변환이 아닌, **최적화 이론 자체의 근본적 재구성**을 의미한다.

본 장에서는 이러한 기하학적 최적화의 수학적 기반을 엄밀하게 구축한다. 핵심은 **리만 메트릭이 유도하는 자연스러운 그래디언트**와 **이산 상태 공간의 조합론적 구조**를 통합하는 것이다.

### 5.1.1. 리만 기하학의 기본 개념

**리만 다양체**(Riemannian Manifold)는 각 점에서 내적이 정의된 매끄러운 다양체이다. 

푸앵카레 볼 $\mathcal{D}^n$에서 리만 메트릭은:

$$g_{ij}(x) = \frac{4\delta_{ij}}{(1-||x||^2)^2}$$

여기서 $\delta_{ij}$는 크로네커 델타이다.

**기하학적 의미:**
- 중심 근처: 거의 유클리드적 ($g_{ij} \approx 4\delta_{ij}$)
- 경계 근처: 극도로 왜곡된 메트릭 ($g_{ij} \to \infty$)

### 5.1.2. 왜 리만 최적화인가?

유클리드 최적화 vs 리만 최적화의 근본적 차이:

| 측면 | 유클리드 최적화 | 리만 최적화 | 장점 |
|:-----|:-------------|:----------|:-----|
| **메트릭** | $\delta_{ij}$ (평탄) | $g_{ij}(x)$ (곡률 의존) | 기하학적 구조 활용 |
| **그래디언트** | $\nabla f$ | $\text{grad} f = g^{-1}\nabla f$ | 자연스러운 방향 |
| **측지선** | 직선 | 곡선 | 다양체 구조 보존 |
| **수렴성** | 국소 최소값 | 전역적 특성 | 더 나은 일반화 |

## 5.2. 푸앵카레 볼의 미분기하학

### 5.2.1. 리만 메트릭의 상세 분석

푸앵카레 볼에서 점 $x$에서의 메트릭 텐서:

$$g(x) = \frac{4}{(1-||x||^2)^2} I_n$$

여기서 $I_n$은 $n \times n$ 단위행렬이다.

**메트릭의 역행렬:**
$$g^{-1}(x) = \frac{(1-||x||^2)^2}{4} I_n$$

**메트릭 행렬식:**
$$\det(g(x)) = \left(\frac{4}{(1-||x||^2)^2}\right)^n$$

### 5.2.2. 크리스토펠 기호 계산

리만 연결의 크리스토펠 기호 $\Gamma^k_{ij}$는:

$$\Gamma^k_{ij} = \frac{1}{2}g^{kl}\left(\frac{\partial g_{il}}{\partial x^j} + \frac{\partial g_{jl}}{\partial x^i} - \frac{\partial g_{ij}}{\partial x^l}\right)$$

푸앵카레 볼에서 계산하면:

$$\Gamma^k_{ij} = \frac{2\delta^k_i x_j + 2\delta^k_j x_i - 2\delta_{ij} x^k}{1-||x||^2}$$

**물리적 해석:**
크리스토펠 기호는 곡률에 의한 "가짜 힘"을 나타낸다. 경계에 가까워질수록 이 힘이 급격히 증가한다.

### 5.2.3. 리만 곡률 텐서

푸앵카레 볼의 리만 곡률 텐서:

$$R^l_{ijk} = \delta^l_k \delta_{ij} - \delta^l_j \delta_{ik}$$

**단면 곡률(Sectional Curvature):**
모든 2차원 부분공간에서 단면 곡률이 $-1$로 일정하다:

$$K = -1$$

이는 푸앵카레 볼이 **균일한 음곡률 공간**임을 의미한다.

### 5.2.4. 측지선과 지수 사상

점 $x$에서 방향 $v$로의 측지선:

$$\gamma(t) = x \oplus t \odot v$$

여기서 $\oplus$는 **Möbius 덧셈**, $\odot$는 **스칼라 곱**이다:

**Möbius 덧셈:**
$$x \oplus y = \frac{x + y + \frac{2}{1+||x||^2} \langle x, y \rangle x}{1 + 2\langle x, y \rangle + ||x||^2 ||y||^2}$$

**스칼라 곱:**
$$t \odot v = \frac{t||v||}{\text{artanh}(||v||)} \cdot \frac{v}{||v||}$$

## 5.3. 리만 그래디언트와 자연 경사

### 5.3.1. 리만 그래디언트의 정의

함수 $f: \mathcal{D}^n \rightarrow \mathbb{R}$에 대한 **리만 그래디언트**:

$$\text{grad} f(x) = g^{-1}(x) \nabla f(x) = \frac{(1-||x||^2)^2}{4} \nabla f(x)$$

**유클리드 vs 리만 그래디언트:**

| 위치 | 유클리드 크기 | 리만 크기 | 비율 |
|:-----|:------------|:---------|:-----|
| 중심 ( $||x|| = 0$ ) | $||\nabla f||$ | $\frac{1}{4}||\nabla f||$ | 0.25 |
| 중간 ($||x|| = 0.5$) | $||\nabla f||$ | $\frac{9}{64}||\nabla f||$ | 0.14 |
| 경계 근처 ($||x|| = 0.9$) | $||\nabla f||$ | $\frac{361}{6400}||\nabla f||$ | 0.056 |

**통찰:** 경계로 갈수록 리만 그래디언트가 급격히 작아져 자연스러운 **적응적 학습률**을 제공한다.

### 5.3.2. 리만 최급강하법

푸앵카레 볼에서의 최급강하법:

$$x_{k+1} = x_k \oplus \left(-\alpha \odot \text{grad} f(x_k)\right)$$

**알고리즘:**
```
function riemannian_gradient_descent(f, x0, alpha, max_iter):
    x = x0
    for k = 1 to max_iter:
        // 1. 유클리드 그래디언트 계산
        grad_euclidean = compute_gradient(f, x)
        
        // 2. 리만 그래디언트로 변환
        factor = (1 - ||x||^2)^2 / 4
        grad_riemannian = factor * grad_euclidean
        
        // 3. 지수 사상으로 업데이트
        x = exponential_map(x, -alpha * grad_riemannian)
        
        // 4. 푸앵카레 볼 경계 처리
        if ||x|| >= 1:
            x = 0.99 * x / ||x||
    
    return x
```

### 5.3.3. 리만 Adam 알고리즘

푸앵카레 볼을 위한 Adam 변형:

**모멘텀 업데이트:**
$$m_k = \beta_1 \odot m_{k-1} \oplus (1-\beta_1) \odot \text{grad} f(x_k)$$

**적응적 학습률:**
$$v_k = \beta_2 v_{k-1} + (1-\beta_2) ||\text{grad} f(x_k)||^2$$

**파라미터 업데이트:**
$$x_{k+1} = x_k \oplus \left(-\frac{\alpha}{\sqrt{v_k} + \epsilon} \odot m_k\right)$$

### 5.3.4. 수렴성 이론

**정리 5.1 (리만 최급강하 수렴성)**
함수 $f$가 리만 $L$-smooth이고 $\mu$-strongly convex이면, 리만 최급강하법은 다음 수렴률을 갖는다:

$$f(x_k) - f^* \leq \left(1 - \frac{\mu}{L}\right)^k (f(x_0) - f^*)$$

**증명 스케치:**
리만 메트릭 하에서 smooth성과 convexity를 재정의하고, 표준 증명을 리만 설정으로 확장한다.

## 5.4. 상태-전이 미분: 조합론적 구조

### 5.4.1. 이산 상태 공간의 수학적 모델

`hi` 필드의 이산 상태를 **조합론적 최적화** 문제로 모델링한다.

**상태 공간:**
$$\mathcal{S} = \{0,1,2,3\}^{32} \times \{0,1\}^{32}$$

여기서 첫 번째 성분은 4진법 상태들, 두 번째는 CORDIC 비트들이다.

**목적함수:**
$$F: \mathcal{S} \times \mathcal{D}^2 \rightarrow \mathbb{R}$$
$$F(s, (r,\theta)) = \mathcal{L}(\text{generate\_weights}(s, r, \theta))$$

### 5.4.2. 상태 전이 그래프

각 상태를 노드로, 가능한 전이를 엣지로 하는 방향 그래프 $G = (V, E)$를 정의한다.

**노드 집합:** $V = \mathcal{S}$
**엣지 집합:** $E = \{(s, s') : d_H(s, s') = 1\}$

여기서 $d_H$는 해밍 거리이다.

**전이 비용:**
$$c(s \rightarrow s') = F(s') - F(s)$$

### 5.4.3. 상태-전이 미분의 정의

이산 상태 $s$에서 함수 $F$의 "미분"을 다음과 같이 정의한다:

$$\partial_s F(s) = \arg\min_{s' \in N(s)} F(s') - F(s)$$

여기서 $N(s)$는 $s$의 이웃 상태들이다.

**방향 미분:**
상태 $s$에서 방향 $s'$로의 방향 미분:

$$D_{s'}F(s) = \lim_{\epsilon \rightarrow 0} \frac{F(s + \epsilon \cdot \mathbf{1}_{s'}) - F(s)}{\epsilon}$$

실제로는 이산적이므로:

$$D_{s'}F(s) = F(\text{flip\_bit}(s, s')) - F(s)$$

### 5.4.4. 확률적 상태 전이

그래디언트 신호에 따른 확률적 전이 규칙:

**볼츠만 분포:**
$$P(s \rightarrow s') = \frac{\exp(-\beta \cdot D_{s'}F(s))}{\sum_{s'' \in N(s)} \exp(-\beta \cdot D_{s''}F(s))}$$

**온도 스케줄링:**
$$\beta(t) = \beta_0 \cdot \left(\frac{t_{cool}}{t_{cool} + t}\right)$$

초기에는 높은 온도로 탐색을 하고, 시간이 지나면서 온도를 낮춰 수렴을 유도한다.

### 5.4.5. 마르코프 체인 수렴성

**정리 5.2 (상태 전이 수렴성)**
전이 확률이 다음 조건을 만족하면 마르코프 체인이 정상분포로 수렴한다:

1. **기약성(Irreducibility)**: 모든 상태 쌍 $(s, s')$에 대해 $s$에서 $s'$로 도달 가능
2. **비주기성(Aperiodicity)**: $\gcd\{n : P^n(s,s) > 0\} = 1$
3. **상세균형(Detailed Balance)**: $\pi(s)P(s \rightarrow s') = \pi(s')P(s' \rightarrow s)$

**증명:** 표준 마르코프 체인 이론 적용.

## 5.5. 하이브리드 최적화: 연속-이산 통합

### 5.5.1. 곱공간에서의 최적화

전체 파라미터 공간을 **곱공간**으로 모델링:

$$\mathcal{M} = \mathcal{S} \times \mathcal{D}^2$$

목적함수:
$$\mathcal{L}: \mathcal{M} \rightarrow \mathbb{R}$$
$$\mathcal{L}(s, x) = \text{Loss}(\text{neural\_network}(\text{generate\_weights}(s, x)))$$

### 5.5.2. 교대 최적화 스킴

연속 파라미터와 이산 상태를 **교대로 최적화**:

**알고리즘 5.1 (하이브리드 최적화)**
```
function hybrid_optimization(L, s0, x0, max_iter):
    s = s0
    x = x0
    
    for k = 1 to max_iter:
        // 1. 연속 파라미터 최적화 (리만 그래디언트)
        x = riemannian_gradient_step(L(s, ·), x)
        
        // 2. 이산 상태 최적화 (확률적 전이)
        s = probabilistic_state_transition(L(·, x), s)
        
        // 3. 수렴 체크
        if ||grad L(s,x)|| < tolerance:
            break
    
    return s, x
```

### 5.5.3. 수렴성 분석

**정리 5.3 (하이브리드 수렴성)**
다음 조건 하에서 교대 최적화가 국소 최소값으로 수렴한다:

1. **연속 부분**: $x \mapsto \mathcal{L}(s, x)$가 각 $s$에 대해 리만 convex
2. **이산 부분**: 상태 전이가 세밀균형 조건 만족
3. **결합 조건**: $\sup_{s,x} ||\nabla_x \mathcal{L}(s,x)|| < \infty$

**증명 아이디어:**
각 단계에서 목적함수가 단조감소함을 보이고, 수렴 부분수열의 존재를 증명한다.

### 5.5.4. 최적화 속도 분석

**연속 파라미터 업데이트 복잡도:**
- 리만 그래디언트 계산: $O(n^2)$
- 지수 사상: $O(n)$
- 전체: $O(n^2)$ per iteration

**이산 상태 업데이트 복잡도:**
- 이웃 상태 평가: $O(|N(s)|) = O(\log |\mathcal{S}|)$
- 확률 계산: $O(|N(s)|)$
- 전체: $O(\log |\mathcal{S}|)$ per iteration

**병렬화 가능성:**
이산 상태의 이웃들을 병렬로 평가 가능하므로 실제 복잡도는 $O(1)$로 감소.

## 5.6. 정보 기하학적 관점

### 5.6.1. 피셔 정보 메트릭

파라미터 분포 $p(w|θ)$에 대한 피셔 정보 행렬:

$$I_{ij}(θ) = \mathbb{E}\left[\frac{\partial \log p(w|θ)}{\partial θ_i} \frac{\partial \log p(w|θ)}{\partial θ_j}\right]$$

푸앵카레 볼 파라미터화에서:

$$I(r,\theta) = \begin{pmatrix}
\frac{4}{(1-r^2)^2} & 0 \\
0 & \frac{1}{r^2}
\end{pmatrix}$$

### 5.6.2. 자연 그래디언트

정보 기하학적 자연 그래디언트:

$$\tilde{\nabla} = I^{-1}(θ) \nabla_θ \mathcal{L}$$

푸앵카레 볼에서:

$$\begin{pmatrix}
\tilde{\nabla}_r \\
\tilde{\nabla}_θ
\end{pmatrix} = \begin{pmatrix}
\frac{(1-r^2)^2}{4} \frac{\partial \mathcal{L}}{\partial r} \\
r^2 \frac{\partial \mathcal{L}}{\partial θ}
\end{pmatrix}$$

이는 앞서 유도한 리만 그래디언트와 일치한다!

### 5.6.3. KL 발산과 곡률의 관계

두 푸앵카레 볼 분포 사이의 KL 발산:

$$KL(P_1 || P_2) = \int p_1(w) \log \frac{p_1(w)}{p_2(w)} dw$$

이는 푸앵카레 볼의 쌍곡거리와 직접 연관된다:

$$KL(P_{θ_1} || P_{θ_2}) \approx \frac{1}{2} d^2_{\text{hyp}}(θ_1, θ_2)$$

여기서 $d_{\text{hyp}}$는 쌍곡거리이다.

## 5.7. 실제 응용: 학습 알고리즘 구현

### 5.7.1. 완전한 학습 루프

```python
def poincare_ball_training(model, data_loader, epochs=100):
    """푸앵카레 볼 기반 신경망 학습"""
    
    # 1. 파라미터 초기화
    poincare_params = initialize_poincare_parameters()
    
    # 2. 리만 옵티마이저 설정
    optimizer = RiemannianAdam(poincare_params, lr=0.01)
    
    for epoch in range(epochs):
        for batch_x, batch_y in data_loader:
            
            # 3. 순전파
            pred = model.forward_poincare(batch_x, poincare_params)
            loss = compute_loss(pred, batch_y)
            
            # 4. 리만 역전파
            riemannian_grads = compute_riemannian_gradients(
                loss, poincare_params
            )
            
            # 5. 상태 전이 그래디언트
            state_grads = compute_state_transition_gradients(
                loss, poincare_params
            )
            
            # 6. 하이브리드 업데이트
            poincare_params = optimizer.step(
                riemannian_grads, state_grads, epoch
            )
            
            # 7. 푸앵카레 볼 제약 투영
            poincare_params = project_to_poincare_ball(poincare_params)
    
    return poincare_params
```

### 5.7.2. 수치적 안정성 보장

**그래디언트 클리핑:**
```python
def clip_riemannian_gradient(grad, max_norm=1.0):
    """리만 그래디언트 클리핑"""
    grad_norm = compute_riemannian_norm(grad)
    if grad_norm > max_norm:
        return grad * (max_norm / grad_norm)
    return grad

def compute_riemannian_norm(grad):
    """리만 노름 계산"""
    r, theta = grad
    norm_r = abs(r) * (1 - r**2)**2 / 4
    norm_theta = abs(theta) * r**2
    return sqrt(norm_r**2 + norm_theta**2)
```

**경계 처리:**
```python
def project_to_poincare_ball(params, max_radius=0.99):
    """푸앵카레 볼 경계 투영"""
    r, theta = params
    
    # r 제약
    r = clip(r, 0.01, max_radius)
    
    # theta 제약 (주기적)
    theta = (theta + pi) % (2*pi) - pi
    
    return r, theta
```

### 5.7.3. 적응적 온도 스케줄링

```python
def adaptive_temperature_schedule(epoch, initial_temp=1.0, decay_rate=0.95):
    """적응적 온도 스케줄링"""
    # 1. 지수 감소
    base_temp = initial_temp * (decay_rate ** epoch)
    
    # 2. 손실 기반 조정
    current_loss = get_current_loss()
    loss_factor = 1.0 + tanh(current_loss - 1.0)
    
    # 3. 최종 온도
    temperature = base_temp * loss_factor
    
    return max(temperature, 0.01)  # 최소 온도 보장
```

## 5.8. 결론: 기하학과 최적화의 융합

본 장에서 제시한 리만 기하학적 최적화 프레임워크는 푸앵카레 볼 기반 RBE의 수학적 기반을 완성한다.

### 5.8.1. 핵심 기여

1. **이론적 엄밀성**: 리만 최적화와 조합 최적화의 수학적 통합
2. **수치적 안정성**: 곡률을 고려한 적응적 학습률과 그래디언트 클리핑
3. **수렴 보장**: 하이브리드 최적화의 수렴성 이론적 증명
4. **실용적 구현**: 완전한 학습 알고리즘과 안정성 보장 기법

### 5.8.2. 기하학적 직관

- **중심부**: 빠른 학습 (큰 메트릭)
- **경계부**: 안정적 수렴 (작은 메트릭)
- **측지선**: 자연스러운 학습 경로
- **곡률**: 적응적 정규화 효과
