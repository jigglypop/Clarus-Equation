"""Independent finite checks for the LGS mathematics lane.

Uses only exact ``Fraction`` arithmetic.  The exhaustive family has three
labelled vertices, each directed non-loop edge absent/weight-1/weight-2, and
then adds every directed edge at cost 1 or 2 (including an already present
edge, which is interpreted as a parallel candidate transition).
"""

from fractions import Fraction
from itertools import product


INF = None
N = 3
ARCS = [(i, j) for i in range(N) for j in range(N) if i != j]


def add(x, y):
    return INF if x is INF or y is INF else x + y


def less(x, y):
    return x is not INF and (y is INF or x < y)


def apsp(edges):
    d = [[INF] * N for _ in range(N)]
    for i in range(N):
        d[i][i] = Fraction(0)
    for i, j, w in edges:
        if d[i][j] is INF or w < d[i][j]:
            d[i][j] = w
    for k in range(N):
        for i in range(N):
            for j in range(N):
                via = add(d[i][k], d[k][j])
                if less(via, d[i][j]):
                    d[i][j] = via
    return d


def verify_t1():
    cases = 0
    existing_lower = 0
    for assignment in product((None, 1, 2), repeat=len(ARCS)):
        edges = [(i, j, Fraction(w)) for (i, j), w in zip(ARCS, assignment) if w]
        d = apsp(edges)
        for u, v in ARCS:
            for a_int in (1, 2):
                a = Fraction(a_int)
                updated = apsp(edges + [(u, v, a)])
                if any(i == u and j == v and w < a for i, j, w in edges):
                    existing_lower += 1
                for i in range(N):
                    for j in range(N):
                        rhs = d[i][j]
                        candidate = add(add(d[i][u], a), d[v][j])
                        if less(candidate, rhs):
                            rhs = candidate
                        assert updated[i][j] == rhs, (
                            assignment, (u, v, a), (i, j), updated[i][j], rhs
                        )
                        assert not less(updated[i][j], updated[i][j])
                cases += 1
    return cases, existing_lower


def fixture_checks():
    # A->X->Y->B, plus P->A and B->Q: one arc changes many pairs, not all.
    edges = [(0, 1, Fraction(1)), (1, 2, Fraction(1)), (2, 3, Fraction(1)),
             (4, 0, Fraction(1)), (3, 5, Fraction(1))]
    # Separate local Floyd--Warshall because this fixture has six vertices.
    m = 6
    d = [[None] * m for _ in range(m)]
    for i in range(m): d[i][i] = Fraction(0)
    for i, j, w in edges: d[i][j] = w
    for k in range(m):
        for i in range(m):
            for j in range(m):
                if d[i][k] is not None and d[k][j] is not None:
                    z = d[i][k] + d[k][j]
                    if d[i][j] is None or z < d[i][j]: d[i][j] = z
    before_pq, before_ab = d[4][5], d[0][3]
    edges.append((0, 3, Fraction(1)))
    after = [[None] * m for _ in range(m)]
    for i in range(m): after[i][i] = Fraction(0)
    for i, j, w in edges:
        if after[i][j] is None or w < after[i][j]: after[i][j] = w
    for k in range(m):
        for i in range(m):
            for j in range(m):
                if after[i][k] is not None and after[k][j] is not None:
                    z = after[i][k] + after[k][j]
                    if after[i][j] is None or z < after[i][j]: after[i][j] = z
    assert (before_ab, after[0][3]) == (Fraction(3), Fraction(1))
    assert (before_pq, after[4][5]) == (Fraction(5), Fraction(3))
    assert after[1][2] == Fraction(1)  # untouched pair


if __name__ == "__main__":
    cases, existing_lower = verify_t1()
    fixture_checks()
    print("LGS math verification: PASS")
    print(f"T1 exhaustive cases: {cases}")
    print(f"Cases with an already-existing lower-cost u->v arc: {existing_lower}")
    print("Fixtures: many-pairs change and an untouched pair: PASS")
