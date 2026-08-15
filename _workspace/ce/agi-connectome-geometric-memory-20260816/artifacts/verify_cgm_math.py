"""Exact/finite checks for the CGM math lane.

Only the Python standard library is used.  The finite enumerations are not
proofs of the general theorems; they are reproducible fixtures for the exact
counterexamples and theorem boundaries recorded in 11-math.md.
"""

from __future__ import annotations

from fractions import Fraction as F
import itertools
import json


def transpose(a):
    return [list(row) for row in zip(*a)]


def mm(a, b):
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def mv(a, v):
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def madd(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def eye(n):
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def inv(a):
    n = len(a)
    aug = [list(row) + ident for row, ident in zip(a, eye(n))]
    for col in range(n):
        pivot = next(row for row in range(col, n) if aug[row][col] != 0)
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            scale = aug[row][col]
            aug[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(aug[row], aug[col])
            ]
    return [row[n:] for row in aug]


def mpow(a, exponent):
    result = eye(len(a))
    base = a
    while exponent:
        if exponent & 1:
            result = mm(result, base)
        base = mm(base, base)
        exponent //= 2
    return result


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def outer(a, b):
    return [[x * y for y in b] for x in a]


def scc(vertices, edges):
    vertices = tuple(vertices)
    adjacency = {v: [] for v in vertices}
    for source, target in edges:
        adjacency[source].append(target)

    index = 0
    stack = []
    on_stack = set()
    indices = {}
    lowlink = {}
    components = []

    def visit(v):
        nonlocal index
        indices[v] = lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in adjacency[v]:
            if w not in indices:
                visit(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)
                if w == v:
                    break
            components.append(tuple(sorted(component)))

    for vertex in vertices:
        if vertex not in indices:
            visit(vertex)
    return tuple(sorted(components))


def condensation(vertices, edges):
    components = scc(vertices, edges)
    block = {vertex: i for i, component in enumerate(components) for vertex in component}
    quotient_edges = {
        (block[source], block[target])
        for source, target in edges
        if block[source] != block[target]
    }
    return components, quotient_edges


def acyclic(vertices, edges):
    adjacency = {v: [] for v in vertices}
    indegree = {v: 0 for v in vertices}
    for source, target in edges:
        adjacency[source].append(target)
        indegree[target] += 1
    ready = [v for v in vertices if indegree[v] == 0]
    seen = 0
    while ready:
        source = ready.pop()
        seen += 1
        for target in adjacency[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return seen == len(vertices)


def fraction_strings(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: fraction_strings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [fraction_strings(item) for item in value]
    return value


def main():
    results = {}

    # D1 and the one-step termination of same-semantics SCC iteration.
    vertices = tuple(range(4))
    possible_edges = tuple(itertools.permutations(vertices, 2))
    checked_graphs = 0
    for mask in range(1 << len(possible_edges)):
        edges = {
            edge for bit, edge in enumerate(possible_edges) if mask & (1 << bit)
        }
        components, quotient_edges = condensation(vertices, edges)
        quotient_vertices = tuple(range(len(components)))
        assert acyclic(quotient_vertices, quotient_edges)
        assert all(
            len(component) == 1
            for component in scc(quotient_vertices, quotient_edges)
        )
        checked_graphs += 1
    results["scc_exhaustive_n4"] = {
        "graphs_without_self_loops": checked_graphs,
        "all_condensations_acyclic": True,
        "all_second_sccs_singleton": True,
    }

    base_edges = {(0, 1), (1, 0), (2, 3), (3, 2), (1, 2)}
    base_components, base_condensation = condensation(vertices, base_edges)
    assert base_components == ((0, 1), (2, 3))
    assert base_condensation == {(0, 1)}
    scale_specific_edges = {(0, 1), (1, 0)}
    assert scc((0, 1), scale_specific_edges) == ((0, 1),)
    results["scc_hierarchy_boundary"] = {
        "level0_components": base_components,
        "same_semantics_condensation_edges": sorted(base_condensation),
        "same_semantics_second_scc": scc((0, 1), base_condensation),
        "new_scale_semantics_scc": scc((0, 1), scale_specific_edges),
    }

    # N1: an invertible latent coordinate change preserves every observation
    # while changing the off-diagonal support of the latent transition matrix.
    a = [[F(1, 2), F(0)], [F(0), F(1, 3)]]
    h = [[F(1), F(1)], [F(0), F(1)]]
    h_inv = inv(h)
    a_prime = mm(mm(h, a), h_inv)
    c = [[F(1), F(2)]]
    c_prime = mm(c, h_inv)
    x0 = [F(2), F(3)]
    x0_prime = mv(h, x0)
    observation_trace = []
    for time in range(7):
        y = mv(c, mv(mpow(a, time), x0))[0]
        y_prime = mv(c_prime, mv(mpow(a_prime, time), x0_prime))[0]
        assert y == y_prime
        observation_trace.append(y)
    assert a[0][1] == 0 and a_prime[0][1] != 0
    results["latent_similarity_counterexample"] = {
        "A": a,
        "A_prime": a_prime,
        "C": c,
        "C_prime": c_prime,
        "observations_t0_to_t6": observation_trace,
        "off_diagonal_support_changed": True,
    }

    # The same Gaussian covariance admits X -> Y and Y -> X factorizations.
    rho = F(1, 2)
    residual_variance = 1 - rho * rho
    covariance_xy = [[F(1), rho], [rho, rho * rho + residual_variance]]
    covariance_yx = [[rho * rho + residual_variance, rho], [rho, F(1)]]
    assert covariance_xy == covariance_yx == [[F(1), F(1, 2)], [F(1, 2), F(1)]]
    results["observational_causal_direction_counterexample"] = {
        "rho": rho,
        "residual_variance": residual_variance,
        "covariance_X_to_Y": covariance_xy,
        "covariance_Y_to_X": covariance_yx,
    }

    # Narrow fully observed LTI theorem fixture: full-row-rank design recovers
    # [A B] uniquely; a rank-deficient design admits an exact null direction.
    theta = [[F(1, 2), F(1, 4), F(1)], [F(0), F(1, 3), F(2)]]
    z = [
        [F(1), F(0), F(0), F(1)],
        [F(0), F(1), F(0), F(1)],
        [F(0), F(0), F(1), F(1)],
    ]
    y = mm(theta, z)
    recovered = mm(mm(y, transpose(z)), inv(mm(z, transpose(z))))
    assert recovered == theta
    z_bad = [[F(1), F(2)], [F(2), F(4)], [F(0), F(0)]]
    delta = [[F(2), F(-1), F(7)], [F(0), F(0), F(0)]]
    assert mm(delta, z_bad) == [[F(0), F(0)], [F(0), F(0)]]
    assert mm(theta, z_bad) == mm(madd(theta, delta), z_bad)
    results["fully_observed_lti_fixture"] = {
        "theta": theta,
        "recovered_theta": recovered,
        "rank_deficient_null_delta": delta,
        "rank_deficient_predictions_equal": True,
    }

    # SCC partitions are discontinuous in threshold and aggregation window.
    weighted_edges = {
        (0, 1): F(3, 5),
        (1, 0): F(3, 5),
        (1, 2): F(2, 5),
        (2, 1): F(2, 5),
    }
    high = {edge for edge, weight in weighted_edges.items() if weight >= F(1, 2)}
    low = {edge for edge, weight in weighted_edges.items() if weight >= F(3, 10)}
    assert scc((0, 1, 2), high) == ((0, 1), (2,))
    assert scc((0, 1, 2), low) == ((0, 1, 2),)
    events = [(1, 0, 1), (2, 1, 0), (3, 1, 2), (4, 2, 1)]
    short_edges = {(source, target) for time, source, target in events if time <= 2}
    long_edges = {(source, target) for time, source, target in events if time <= 4}
    assert scc((0, 1, 2), short_edges) == ((0, 1), (2,))
    assert scc((0, 1, 2), long_edges) == ((0, 1, 2),)
    results["scc_definition_sensitivity"] = {
        "threshold_1_over_2": scc((0, 1, 2), high),
        "threshold_3_over_10": scc((0, 1, 2), low),
        "window_t_le_2": scc((0, 1, 2), short_edges),
        "window_t_le_4": scc((0, 1, 2), long_edges),
    }

    # SCC membership (even plus block sum) is not automatically a predictive
    # sufficient statistic.  Exact sufficiency would require Q A = Abar Q.
    predictive_a = [[F(0), F(1)], [F(2), F(0)]]
    q = [[F(1), F(1)]]
    state_left = [F(1), F(0)]
    state_right = [F(0), F(1)]
    current_left = mv(q, state_left)
    current_right = mv(q, state_right)
    next_left = mv(q, mv(predictive_a, state_left))
    next_right = mv(q, mv(predictive_a, state_right))
    assert current_left == current_right == [F(1)]
    assert next_left == [F(2)] and next_right == [F(1)]
    assert mm(q, predictive_a) == [[F(2), F(1)]]
    results["scc_not_predictively_sufficient"] = {
        "A": predictive_a,
        "same_current_block_sum": current_left,
        "different_next_block_sums": [next_left, next_right],
        "Q_A": mm(q, predictive_a),
        "no_scalar_Abar_can_make_QA_equal_AbarQ": True,
    }

    # Geometry/weight gauge: only K = W^T g W is observed by quadratic costs.
    w = eye(2)
    metric = [[F(4), F(0)], [F(0), F(9)]]
    s = [[F(1), F(1)], [F(0), F(1)]]
    w_prime = mm(s, w)
    metric_prime = mm(mm(transpose(inv(s)), metric), inv(s))
    k = mm(mm(transpose(w), metric), w)
    k_prime = mm(mm(transpose(w_prime), metric_prime), w_prime)
    assert k == k_prime
    samples = [[F(1), F(2)], [F(-3), F(1)], [F(0), F(5)]]
    for sample in samples:
        assert dot(sample, mv(k, sample)) == dot(sample, mv(k_prime, sample))
    results["geometry_weight_gauge"] = {
        "W": w,
        "g": metric,
        "W_prime": w_prime,
        "g_prime": metric_prime,
        "observable_K": k,
        "all_sample_costs_equal": True,
    }

    # A static SPD quadratic form has no arrow: v and -v have equal cost.
    displacement = [F(2), F(-1)]
    reverse = [-value for value in displacement]
    forward_cost = dot(displacement, mv(metric, displacement))
    reverse_cost = dot(reverse, mv(metric, reverse))
    assert forward_cost == reverse_cost
    results["static_metric_direction_no_go"] = {
        "forward_cost": forward_cost,
        "reverse_cost": reverse_cost,
        "equal": True,
    }

    # Operational memory geometry from the finite-horizon controllability
    # Gramian.  For this exact example W_2 is invertible and the minimum energy
    # to reach x=(1,0) is x^T W_2^-1 x = 40.
    control_a = [[F(1, 2), F(0)], [F(0), F(1, 3)]]
    control_b = [[F(1)], [F(1)]]
    ab = mv(control_a, [F(1), F(1)])
    b_vector = [F(1), F(1)]
    gramian = madd(outer(b_vector, b_vector), outer(ab, ab))
    gramian_inv = inv(gramian)
    target = [F(1), F(0)]
    reachability = [[ab[0], b_vector[0]], [ab[1], b_vector[1]]]
    optimal_control = mv(transpose(reachability), mv(gramian_inv, target))
    reached = mv(reachability, optimal_control)
    energy = dot(optimal_control, optimal_control)
    metric_energy = dot(target, mv(gramian_inv, target))
    assert gramian == [[F(5, 4), F(7, 6)], [F(7, 6), F(10, 9)]]
    assert gramian_inv == [[F(40), F(-42)], [F(-42), F(45)]]
    assert optimal_control == [F(6), F(-2)]
    assert reached == target
    assert energy == metric_energy == F(40)
    rank_deficient_b = [F(1), F(0)]
    rank_deficient_ab = mv(control_a, rank_deficient_b)
    rank_deficient_gramian = madd(
        outer(rank_deficient_b, rank_deficient_b),
        outer(rank_deficient_ab, rank_deficient_ab),
    )
    assert rank_deficient_gramian[1] == [F(0), F(0)]
    results["controllability_gramian_metric"] = {
        "W_2": gramian,
        "g_2_inverse_gramian": gramian_inv,
        "target": target,
        "optimal_control": optimal_control,
        "minimum_energy": energy,
        "rank_deficient_W_2": rank_deficient_gramian,
        "rank_deficient_target_0_1_unreachable": True,
    }

    # H5 cannot promise strict improvement: identical context metrics collapse
    # exactly to the single-metric model for every mixture weight.
    identity = eye(2)
    for numerator in range(6):
        alpha = F(numerator, 5)
        mixture = [
            [alpha * x + (1 - alpha) * y for x, y in zip(row_x, row_y)]
            for row_x, row_y in zip(identity, identity)
        ]
        assert mixture == identity
    results["context_mixture_strict_improvement_counterexample"] = {
        "g_context_1": identity,
        "g_context_2": identity,
        "mixture_equals_single_for_all_alpha": True,
    }

    print(json.dumps(fraction_strings(results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
