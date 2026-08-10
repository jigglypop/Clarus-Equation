import numpy as np

from reality_stone.clarus.connection_holonomy import (
    parallel_transport_sphere,
    spherical_triangle_area,
    spherical_triangle_holonomy,
)


def test_octant_holonomy_equals_spherical_area() -> None:
    triangle = np.eye(3)
    assert abs(spherical_triangle_area(triangle) - np.pi / 2.0) < 1e-12
    assert abs(spherical_triangle_holonomy(triangle) - np.pi / 2.0) < 1e-12


def test_parallel_transport_remains_tangent_and_preserves_norm() -> None:
    start = np.array([1.0, 0.0, 0.0])
    end = np.array([0.0, 1.0, 0.0])
    tangent = np.array([0.0, 0.0, 1.0])
    transported = parallel_transport_sphere(start, end, tangent)
    assert abs(float(transported @ end)) < 1e-12
    assert abs(np.linalg.norm(transported) - 1.0) < 1e-12
