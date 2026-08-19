//! Gate-A-only numerical kernel for the frozen NRM3-D contract.
//! It deliberately contains no dataset loader or scientific outcome command.

use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

pub type Mat3 = Matrix3<f64>;
pub type Vec3 = Vector3<f64>;

pub const SPD_MIN: f64 = 1.0e-10;
pub const SYMMETRY_TOL: f64 = 1.0e-13;

#[derive(Debug, Clone, PartialEq)]
pub enum GeometryError {
    NonFinite(&'static str),
    NotSymmetric { relative: f64 },
    NotSpd { min_eigenvalue: f64, condition: f64 },
    SingularPullback { rank: usize },
    InvalidStep,
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for GeometryError {}

fn frob(a: &Mat3) -> f64 {
    a.norm()
}
fn finite(a: &Mat3) -> bool {
    a.iter().all(|x| x.is_finite())
}
fn sym(a: Mat3) -> Mat3 {
    (a + a.transpose()) * 0.5
}

/// Validated symmetric positive definite 3-by-3 covariant tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Sym3(Mat3);

impl Sym3 {
    pub fn try_new(value: Mat3) -> Result<Self, GeometryError> {
        if !finite(&value) {
            return Err(GeometryError::NonFinite("matrix"));
        }
        let antisym = &value - value.transpose();
        let relative = frob(&antisym) / frob(&value).max(1.0);
        if relative > SYMMETRY_TOL {
            return Err(GeometryError::NotSymmetric { relative });
        }
        let value = sym(value);
        let eig = SymmetricEigen::new(value);
        let min = eig.eigenvalues.min();
        let max = eig.eigenvalues.max();
        let condition = if min > 0.0 { max / min } else { f64::INFINITY };
        if min <= SPD_MIN || !condition.is_finite() || condition > 1.0e10 {
            return Err(GeometryError::NotSpd {
                min_eigenvalue: min,
                condition,
            });
        }
        Ok(Self(value))
    }
    pub fn matrix(&self) -> Mat3 {
        self.0
    }
    pub fn eigenvalues(&self) -> Vector3<f64> {
        SymmetricEigen::new(self.0).eigenvalues
    }
    fn spectral(&self, f: impl Fn(f64) -> f64) -> Result<Self, GeometryError> {
        let eig = SymmetricEigen::new(self.0);
        let d = Mat3::from_diagonal(&eig.eigenvalues.map(f));
        Self::try_new(eig.eigenvectors * d * eig.eigenvectors.transpose())
    }
    pub fn log(&self) -> Result<Mat3, GeometryError> {
        let eig = SymmetricEigen::new(self.0);
        let result = eig.eigenvectors
            * Mat3::from_diagonal(&eig.eigenvalues.map(f64::ln))
            * eig.eigenvectors.transpose();
        if finite(&result) {
            Ok(sym(result))
        } else {
            Err(GeometryError::NonFinite("matrix logarithm"))
        }
    }
    pub fn sqrt(&self) -> Result<Self, GeometryError> {
        self.spectral(f64::sqrt)
    }
    pub fn inverse(&self) -> Result<Self, GeometryError> {
        self.spectral(|x| 1.0 / x)
    }
    pub fn cholesky(&self) -> Result<Mat3, GeometryError> {
        self.0
            .cholesky()
            .map(|c| c.l())
            .ok_or(GeometryError::NotSpd {
                min_eigenvalue: self.eigenvalues().min(),
                condition: f64::INFINITY,
            })
    }
    pub fn from_cholesky(lower: Mat3) -> Result<Self, GeometryError> {
        if !finite(&lower)
            || (0..3).any(|i| lower[(i, i)] <= 0.0)
            || (0..3).any(|i| (i + 1..3).any(|j| lower[(i, j)] != 0.0))
        {
            return Err(GeometryError::NotSpd {
                min_eigenvalue: 0.0,
                condition: f64::INFINITY,
            });
        }
        Sym3::try_new(lower * lower.transpose())
    }
    pub fn inverse_sqrt(&self) -> Result<Self, GeometryError> {
        self.spectral(|x| 1.0 / x.sqrt())
    }
}

/// Exponential of a symmetric (not necessarily SPD) endomorphism in an orthonormal frame.
pub fn symmetric_exp(s: Mat3) -> Result<Sym3, GeometryError> {
    if !finite(&s) {
        return Err(GeometryError::NonFinite("exponent"));
    }
    let relative = frob(&(s - s.transpose())) / frob(&s).max(1.0);
    if relative > SYMMETRY_TOL {
        return Err(GeometryError::NotSymmetric { relative });
    }
    let e = SymmetricEigen::new(sym(s));
    let d = Mat3::from_diagonal(&e.eigenvalues.map(f64::exp));
    Sym3::try_new(e.eigenvectors * d * e.eigenvectors.transpose())
}

pub fn sym3_basis() -> [Mat3; 6] {
    let r2 = 2.0_f64.sqrt();
    let r3 = 3.0_f64.sqrt();
    let r6 = 6.0_f64.sqrt();
    [
        Mat3::identity() / r3,
        Mat3::new(1.0 / r2, 0.0, 0.0, 0.0, -1.0 / r2, 0.0, 0.0, 0.0, 0.0),
        Mat3::new(1.0 / r6, 0.0, 0.0, 0.0, 1.0 / r6, 0.0, 0.0, 0.0, -2.0 / r6),
        Mat3::new(0.0, 1.0 / r2, 0.0, 1.0 / r2, 0.0, 0.0, 0.0, 0.0, 0.0),
        Mat3::new(0.0, 0.0, 1.0 / r2, 0.0, 0.0, 0.0, 1.0 / r2, 0.0, 0.0),
        Mat3::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / r2, 0.0, 1.0 / r2, 0.0),
    ]
}
pub fn sym3_coefficients(s: Mat3) -> [f64; 6] {
    let b = sym3_basis();
    std::array::from_fn(|i| s.component_mul(&b[i]).sum())
}

/// Equation (15): metric from a coframe and a full six-component frame tensor.
pub fn coframe_metric(coframe: Mat3, frame_s: Mat3) -> Result<Sym3, GeometryError> {
    if coframe.determinant() <= 0.0 {
        return Err(GeometryError::SingularPullback {
            rank: matrix_rank(coframe),
        });
    }
    let e = symmetric_exp(frame_s)?;
    Sym3::try_new(coframe.transpose() * e.matrix() * coframe)
}

/// Equation (16), represented in a g0-orthonormal frame; avoids a coordinate principal square root.
pub fn intrinsic_relative_log(g0_coframe: Mat3, gt: &Sym3) -> Result<Mat3, GeometryError> {
    if g0_coframe.determinant() <= 0.0 {
        return Err(GeometryError::SingularPullback {
            rank: matrix_rank(g0_coframe),
        });
    }
    let inv = g0_coframe
        .try_inverse()
        .ok_or(GeometryError::SingularPullback {
            rank: matrix_rank(g0_coframe),
        })?;
    let relative = Sym3::try_new(inv.transpose() * gt.matrix() * inv)?;
    relative.log()
}

pub fn pullback(jacobian: Mat3, ambient_metric: &Sym3) -> Result<Sym3, GeometryError> {
    if matrix_rank(jacobian) < 3 {
        return Err(GeometryError::SingularPullback {
            rank: matrix_rank(jacobian),
        });
    }
    Sym3::try_new(jacobian.transpose() * ambient_metric.matrix() * jacobian)
}

pub fn chart_transform_covariant(
    metric: &Sym3,
    j_new_from_old: Mat3,
) -> Result<Sym3, GeometryError> {
    let inverse = j_new_from_old
        .try_inverse()
        .ok_or(GeometryError::SingularPullback {
            rank: matrix_rank(j_new_from_old),
        })?;
    Sym3::try_new(inverse.transpose() * metric.matrix() * inverse)
}

pub fn curve_length(
    metric: impl Fn(Vec3) -> Result<Sym3, GeometryError>,
    points: &[Vec3],
) -> Result<f64, GeometryError> {
    if points.len() < 2 {
        return Ok(0.0);
    }
    let mut length = 0.0;
    for segment in points.windows(2) {
        let delta = segment[1] - segment[0];
        let midpoint = (segment[0] + segment[1]) * 0.5;
        let g = metric(midpoint)?.matrix();
        let ds2 = delta.dot(&(g * delta));
        if !ds2.is_finite() || ds2 < 0.0 {
            return Err(GeometryError::NonFinite("curve length"));
        }
        length += ds2.sqrt();
        if !length.is_finite() {
            return Err(GeometryError::NonFinite("accumulated curve length"));
        }
    }
    Ok(length)
}

fn matrix_rank(a: Mat3) -> usize {
    nalgebra::linalg::SVD::new(a, false, false)
        .singular_values
        .iter()
        .filter(|x| **x > 1.0e-12)
        .count()
}

fn axis_step(axis: usize, h: f64) -> Vec3 {
    match axis {
        0 => Vec3::new(h, 0.0, 0.0),
        1 => Vec3::new(0.0, h, 0.0),
        2 => Vec3::new(0.0, 0.0, h),
        _ => unreachable!("3D axis"),
    }
}

pub fn folded_r(y: Vec3) -> Vec3 {
    let (u, v, w) = (y.x, y.y, y.z);
    let p = std::f64::consts::PI;
    let f = 0.25 * (p * u).sin() * (p * v).sin();
    let fx = 0.25 * p * (p * u).cos() * (p * v).sin();
    let fy = 0.25 * p * (p * u).sin() * (p * v).cos();
    let raw = Vec3::new(-fx, -fy, 1.0);
    let n = raw / raw.norm();
    Vec3::new(u, v, f) + 0.05 * w * n
}

/// Analytic Dr for (11), including the normal derivative.
pub fn folded_dr(y: Vec3) -> Mat3 {
    let (u, v, w) = (y.x, y.y, y.z);
    let p = std::f64::consts::PI;
    let su = Vec3::new(1.0, 0.0, 0.25 * p * (p * u).cos() * (p * v).sin());
    let sv = Vec3::new(0.0, 1.0, 0.25 * p * (p * u).sin() * (p * v).cos());
    let fx = su.z;
    let fy = sv.z;
    let fxx = -0.25 * p * p * (p * u).sin() * (p * v).sin();
    let fxy = 0.25 * p * p * (p * u).cos() * (p * v).cos();
    let fyy = fxx;
    let raw = Vec3::new(-fx, -fy, 1.0);
    let norm = raw.norm();
    let n = raw / norm;
    let raw_u = Vec3::new(-fxx, -fxy, 0.0);
    let raw_v = Vec3::new(-fxy, -fyy, 0.0);
    let n_u = raw_u / norm - raw * raw.dot(&raw_u) / norm.powi(3);
    let n_v = raw_v / norm - raw * raw.dot(&raw_v) / norm.powi(3);
    Mat3::from_columns(&[su + 0.05 * w * n_u, sv + 0.05 * w * n_v, 0.05 * n])
}

/// Principal curvatures of the mid-surface in (11), with the upward normal.
pub fn folded_principal_curvatures(y: Vec3) -> Result<(f64, f64), GeometryError> {
    let (u, v) = (y.x, y.y);
    let p = std::f64::consts::PI;
    let fu = 0.25 * p * (p * u).cos() * (p * v).sin();
    let fv = 0.25 * p * (p * u).sin() * (p * v).cos();
    let fuu = -0.25 * p * p * (p * u).sin() * (p * v).sin();
    let fuv = 0.25 * p * p * (p * u).cos() * (p * v).cos();
    let fvv = fuu;
    let a = nalgebra::Matrix2::new(1.0 + fu * fu, fu * fv, fu * fv, 1.0 + fv * fv);
    let b = nalgebra::Matrix2::new(fuu, fuv, fuv, fvv) / (1.0 + fu * fu + fv * fv).sqrt();
    let shape = a
        .try_inverse()
        .ok_or(GeometryError::SingularPullback { rank: 0 })?
        * b;
    let discriminant =
        (shape[(0, 0)] - shape[(1, 1)]).powi(2) + 4.0 * shape[(0, 1)] * shape[(1, 0)];
    if discriminant < 0.0 || !discriminant.is_finite() {
        return Err(GeometryError::NonFinite("principal curvature"));
    }
    let root = discriminant.sqrt();
    Ok((
        (shape[(0, 0)] + shape[(1, 1)] + root) * 0.5,
        (shape[(0, 0)] + shape[(1, 1)] - root) * 0.5,
    ))
}

pub fn folded_j_perp(y: Vec3) -> Result<f64, GeometryError> {
    let (u, v) = (y.x, y.y);
    let p = std::f64::consts::PI;
    let su = Vec3::new(1.0, 0.0, 0.25 * p * (p * u).cos() * (p * v).sin());
    let sv = Vec3::new(0.0, 1.0, 0.25 * p * (p * u).sin() * (p * v).cos());
    let area = su.cross(&sv).norm();
    if area <= 0.0 || !area.is_finite() {
        return Err(GeometryError::SingularPullback { rank: 0 });
    }
    Ok(folded_dr(y).determinant().abs() / (0.05 * area))
}

pub fn folded_nonneighbor_min_distance(points: &[Vec3]) -> f64 {
    let mut min = f64::INFINITY;
    for (left, y) in points.iter().enumerate() {
        let iz = left % 9;
        let iy = (left / 9) % 17;
        let ix = left / (9 * 17);
        for (right, other) in points.iter().enumerate().skip(left + 1) {
            let jz = right % 9;
            let jy = (right / 9) % 17;
            let jx = right / (9 * 17);
            if ix.abs_diff(jx).max(iy.abs_diff(jy)).max(iz.abs_diff(jz)) > 1 {
                min = min.min((folded_r(*y) - folded_r(*other)).norm());
            }
        }
    }
    min
}

pub fn folded_h(y: Vec3) -> Result<Sym3, GeometryError> {
    pullback(folded_dr(y), &Sym3::try_new(Mat3::identity())?)
}

pub fn flat_map(y: Vec3, zeta: [f64; 6]) -> Vec3 {
    let p = std::f64::consts::PI;
    let s1 = |x: f64| (p * x).sin();
    let s2 = |x: f64| (2.0 * p * x).sin();
    Vec3::new(
        y.x + zeta[0] * s1(y.x) * s1(y.y) * s1(y.z) + zeta[1] * s2(y.x) * s1(y.y) * s1(y.z),
        y.y + zeta[2] * s1(y.x) * s2(y.y) * s1(y.z) + zeta[3] * s1(y.x) * s1(y.y) * s2(y.z),
        y.z + zeta[4] * s2(y.x) * s1(y.y) * s1(y.z) + zeta[5] * s1(y.x) * s2(y.y) * s1(y.z),
    )
}

pub fn flat_map_jacobian(y: Vec3, zeta: [f64; 6]) -> Mat3 {
    let p = std::f64::consts::PI;
    let s1 = |x: f64| (p * x).sin();
    let c1 = |x: f64| p * (p * x).cos();
    let s2 = |x: f64| (2.0 * p * x).sin();
    let c2 = |x: f64| 2.0 * p * (2.0 * p * x).cos();
    let (u, v, w) = (y.x, y.y, y.z);
    Mat3::new(
        1.0 + zeta[0] * c1(u) * s1(v) * s1(w) + zeta[1] * c2(u) * s1(v) * s1(w),
        zeta[0] * s1(u) * c1(v) * s1(w) + zeta[1] * s2(u) * c1(v) * s1(w),
        zeta[0] * s1(u) * s1(v) * c1(w) + zeta[1] * s2(u) * s1(v) * c1(w),
        zeta[2] * c1(u) * s2(v) * s1(w) + zeta[3] * c1(u) * s1(v) * s2(w),
        1.0 + zeta[2] * s1(u) * c2(v) * s1(w) + zeta[3] * s1(u) * c1(v) * s2(w),
        zeta[2] * s1(u) * s2(v) * c1(w) + zeta[3] * s1(u) * s1(v) * c2(w),
        zeta[4] * c2(u) * s1(v) * s1(w) + zeta[5] * c1(u) * s2(v) * s1(w),
        zeta[4] * s2(u) * c1(v) * s1(w) + zeta[5] * s1(u) * c2(v) * s1(w),
        1.0 + zeta[4] * s2(u) * s1(v) * c1(w) + zeta[5] * s1(u) * s2(v) * c1(w),
    )
}

pub fn flat_metric(y: Vec3, zeta: [f64; 6]) -> Result<Sym3, GeometryError> {
    let f = flat_map(y, zeta);
    pullback(flat_map_jacobian(y, zeta), &folded_h(f)?)
}

/// Independently assembled pullback: compose ambient Jacobians before forming Gram matrix.
pub fn flat_metric_expected(y: Vec3, zeta: [f64; 6]) -> Result<Sym3, GeometryError> {
    let f = flat_map(y, zeta);
    let composed = folded_dr(f) * flat_map_jacobian(y, zeta);
    Sym3::try_new(composed.transpose() * composed)
}

pub fn curved_metric(y: Vec3) -> Result<Sym3, GeometryError> {
    let (x, yy, z) = (y.x, y.y, y.z);
    let c = Mat3::new(0.20, 0.10, -0.08, 0.10, -0.15, 0.07, -0.08, 0.07, 0.12);
    let b = Mat3::new(
        x.powi(3),
        yy.powi(3),
        z.powi(3),
        yy.powi(3),
        (x + yy + z).powi(3),
        x.powi(3),
        z.powi(3),
        x.powi(3),
        (x - yy).powi(3),
    );
    symmetric_exp(c + Mat3::identity() * (0.14 * (x * x + yy * yy + z * z)) + b * 0.02)
}

fn derivative4(
    f: &impl Fn(Vec3) -> Result<Mat3, GeometryError>,
    x: Vec3,
    axis: usize,
    h: f64,
) -> Result<Mat3, GeometryError> {
    if h <= 0.0 || !h.is_finite() {
        return Err(GeometryError::InvalidStep);
    }
    let d = axis_step(axis, h);
    Ok((-f(x + 2.0 * d)? + 8.0 * f(x + d)? - 8.0 * f(x - d)? + f(x - 2.0 * d)?) / (12.0 * h))
}

fn christoffel(
    f: &impl Fn(Vec3) -> Result<Mat3, GeometryError>,
    x: Vec3,
    h: f64,
) -> Result<[[[f64; 3]; 3]; 3], GeometryError> {
    let g = f(x)?;
    let inv = g
        .try_inverse()
        .ok_or(GeometryError::SingularPullback { rank: 0 })?;
    let dg = [
        derivative4(f, x, 0, h)?,
        derivative4(f, x, 1, h)?,
        derivative4(f, x, 2, h)?,
    ];
    let mut gamma = [[[0.0; 3]; 3]; 3];
    for k in 0..3 {
        for i in 0..3 {
            for j in 0..3 {
                gamma[k][i][j] = (0..3)
                    .map(|l| 0.5 * inv[(k, l)] * (dg[i][(l, j)] + dg[j][(l, i)] - dg[l][(i, j)]))
                    .sum();
            }
        }
    }
    Ok(gamma)
}

#[derive(Debug, Clone, Serialize)]
pub struct Curvature3 {
    pub scalar: f64,
    pub ricci_invariant_norm: f64,
    pub riemann_invariant_norm: f64,
    pub riemann_direct_norm: f64,
    pub riemann_mixed_difference: f64,
}

pub fn curvature3(
    f: impl Fn(Vec3) -> Result<Sym3, GeometryError>,
    x: Vec3,
    h: f64,
) -> Result<Curvature3, GeometryError> {
    let wrapped = |p| Ok(f(p)?.matrix());
    let g = f(x)?;
    let inv = g
        .matrix()
        .try_inverse()
        .ok_or(GeometryError::SingularPullback { rank: 0 })?;
    let gamma = christoffel(&wrapped, x, h)?;
    let dg = |axis: usize| -> Result<[[[f64; 3]; 3]; 3], GeometryError> {
        let d = axis_step(axis, h);
        let a = christoffel(&wrapped, x + 2.0 * d, h)?;
        let b = christoffel(&wrapped, x + d, h)?;
        let c = christoffel(&wrapped, x - d, h)?;
        let e = christoffel(&wrapped, x - 2.0 * d, h)?;
        let mut out = [[[0.0; 3]; 3]; 3];
        for r in 0..3 {
            for i in 0..3 {
                for j in 0..3 {
                    out[r][i][j] = (-a[r][i][j] + 8.0 * b[r][i][j] - 8.0 * c[r][i][j] + e[r][i][j])
                        / (12.0 * h);
                }
            }
        }
        Ok(out)
    };
    let dgamma = [dg(0)?, dg(1)?, dg(2)?];
    let mut ricci = Mat3::zeros();
    let mut riemann = [[[[0.0; 3]; 3]; 3]; 3];
    for sigma in 0..3 {
        for nu in 0..3 {
            for rho in 0..3 {
                for mu in 0..3 {
                    let mut r = dgamma[mu][rho][nu][sigma] - dgamma[nu][rho][mu][sigma];
                    for lambda in 0..3 {
                        r += gamma[rho][mu][lambda] * gamma[lambda][nu][sigma]
                            - gamma[rho][nu][lambda] * gamma[lambda][mu][sigma];
                    }
                    riemann[rho][sigma][mu][nu] = r;
                }
                ricci[(sigma, nu)] += riemann[rho][sigma][rho][nu];
            }
        }
    }
    let scalar = (inv * ricci).trace();
    let mut ricci_sq = 0.0;
    let mut direct_riemann_sq = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    ricci_sq += inv[(a, c)] * inv[(b, d)] * ricci[(a, b)] * ricci[(c, d)];
                    for e in 0..3 {
                        for ff in 0..3 {
                            for gg in 0..3 {
                                for hh in 0..3 {
                                    direct_riemann_sq += g.matrix()[(a, e)]
                                        * inv[(b, ff)]
                                        * inv[(c, gg)]
                                        * inv[(d, hh)]
                                        * riemann[a][b][c][d]
                                        * riemann[e][ff][gg][hh];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // In 3D Weyl vanishes: |Rm|^2 = 4|Ric|^2 - R^2. Keep the direct tensor above
    // for index-level auditing, but use this algebraically equivalent invariant to
    // avoid a second independent eight-index accumulation becoming the tolerance bottleneck.
    let riemann_sq = 4.0 * ricci_sq - scalar * scalar;
    if !scalar.is_finite()
        || ricci_sq < -1e-12
        || riemann_sq < -1e-12
        || !ricci_sq.is_finite()
        || !riemann_sq.is_finite()
        || !direct_riemann_sq.is_finite()
    {
        Err(GeometryError::NonFinite("scalar curvature"))
    } else {
        if direct_riemann_sq < -1e-12 {
            return Err(GeometryError::NonFinite(
                "negative direct Riemann contraction",
            ));
        }
        let direct_norm = direct_riemann_sq.max(0.0).sqrt();
        let identity_difference = (direct_norm - riemann_sq.max(0.0).sqrt()).abs();
        Ok(Curvature3 {
            scalar,
            ricci_invariant_norm: ricci_sq.max(0.0).sqrt(),
            riemann_invariant_norm: riemann_sq.max(0.0).sqrt(),
            riemann_direct_norm: direct_norm,
            riemann_mixed_difference: identity_difference,
        })
    }
}

pub fn scalar_curvature(
    f: impl Fn(Vec3) -> Result<Sym3, GeometryError>,
    x: Vec3,
    h: f64,
) -> Result<f64, GeometryError> {
    Ok(curvature3(f, x, h)?.scalar)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandidateConfig {
    pub id: String,
    pub parameter_budget: u8,
    pub route: String,
}
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FitInput {
    pub circuit_id: u32,
    pub features: Vec<f64>,
    pub paths: Vec<[f64; 3]>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Prediction {
    pub circuit_id: u32,
    pub nll: f64,
}
/// The fitter cannot receive generator labels or true metric fields.
pub trait Fitter: Send + Sync {
    fn fit_predict(
        &self,
        candidate: &CandidateConfig,
        input: &FitInput,
    ) -> Result<Prediction, GeometryError>;
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexedRecord {
    pub index: usize,
    pub digest: String,
}

/// The exact UTF-8 tuple rule frozen in 11-math.md, expanded into a ChaCha20 seed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalSeedTuple {
    pub version: u8,
    pub master_seed: String,
    pub route: String,
    pub generator: String,
    pub dataset: u32,
    pub split: String,
    pub circuit: u32,
    pub condition: String,
    pub force: String,
    pub path: u32,
    pub candidate: String,
}

impl CanonicalSeedTuple {
    pub fn canonical_utf8(&self) -> String {
        let fields = [
            self.master_seed.as_str(),
            self.route.as_str(),
            self.generator.as_str(),
            self.split.as_str(),
            self.condition.as_str(),
            self.force.as_str(),
            self.candidate.as_str(),
        ];
        if fields
            .iter()
            .any(|v| v.len() > 64 || !v.is_ascii() || v.contains('|'))
        {
            return String::new();
        }
        let mut out = format!("NRM3-SEED-v{}", self.version);
        for field in fields {
            out.push_str(&format!(":{}:{}", field.len(), field));
        }
        out.push_str(&format!(":{}:{}:{}", self.dataset, self.circuit, self.path));
        out
    }
}

pub fn rng_for_seed(seed: &CanonicalSeedTuple) -> ChaCha20Rng {
    let canonical = seed.canonical_utf8();
    assert!(!canonical.is_empty(), "invalid canonical seed fields");
    let hash = blake3::hash(canonical.as_bytes());
    ChaCha20Rng::from_seed(*hash.as_bytes())
}

pub fn deterministic_records(n: usize) -> Vec<IndexedRecord> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = rng_for_seed(&CanonicalSeedTuple {
                version: 1,
                master_seed: "0x4e524d3344504643".into(),
                route: "R-KERNEL-3D".into(),
                generator: "Gate-A".into(),
                dataset: i as u32,
                split: "fixture".into(),
                circuit: i as u32,
                condition: "none".into(),
                force: "none".into(),
                path: 0,
                candidate: "kernel".into(),
            });
            IndexedRecord {
                index: i,
                digest: blake3::hash(format!("NRM3-GATE-A:{i}:{}", rng.next_u64()).as_bytes())
                    .to_hex()
                    .to_string(),
            }
        })
        .collect()
}
pub fn canonical_parallel_aggregate(thread_count: usize) -> Result<Vec<u8>, GeometryError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .map_err(|_| GeometryError::NonFinite("rayon pool"))?;
    let records = pool.install(|| deterministic_records(64));
    let records_json =
        serde_json::to_string(&records).map_err(|_| GeometryError::NonFinite("serialization"))?;
    let mut aggregate = BTreeMap::new();
    aggregate.insert(
        "aggregate_digest",
        blake3::hash(records_json.as_bytes()).to_hex().to_string(),
    );
    aggregate.insert("records", records_json);
    serde_json::to_vec(&aggregate).map_err(|_| GeometryError::NonFinite("serialization"))
}
pub fn config_hash() -> String {
    blake3::hash(b"NRM3-D1|Gate-A-r6|f64|Matrix3|riemann3d|atlas-absdet-0.025|sigma-0.04|tau-k-0.25|jperp-0.5625|oracle-manifest|no-outcomes|20260819")
        .to_hex()
        .to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FixtureReport {
    pub name: &'static str,
    pub value: f64,
    pub threshold: f64,
    pub pass: bool,
}
pub fn gate_a_fixtures() -> Result<Vec<FixtureReport>, GeometryError> {
    let s = Mat3::new(0.2, 0.1, -0.08, 0.1, -0.15, 0.07, -0.08, 0.07, 0.12);
    let e = symmetric_exp(s)?;
    let rt = e.log()?;
    let exp_log = (rt - s).norm() / s.norm();
    let chol = e.cholesky()?;
    let chol_error = (Sym3::from_cholesky(chol)?.matrix() - e.matrix()).norm() / e.matrix().norm();
    let coeff = sym3_coefficients(s);
    let basis = sym3_basis();
    let rebuilt = (0..6).fold(Mat3::zeros(), |acc, i| acc + basis[i] * coeff[i]);
    let basis_error = (rebuilt - s).norm() / s.norm();
    let spatial_recovery = [
        Vec3::new(-0.2, 0.1, -0.1),
        Vec3::new(0.1, -0.15, 0.2),
        Vec3::new(0.2, 0.2, -0.15),
    ]
    .iter()
    .map(|p| {
        let c = [
            0.12 + p.x,
            -0.08 + p.y,
            0.06 + p.z,
            0.04 + p.x * p.y,
            -0.03 + p.y * p.z,
            0.05 + p.x * p.z,
        ];
        let b = sym3_basis();
        let l = (0..6).fold(Mat3::zeros(), |a, i| a + b[i] * c[i]);
        let e = folded_dr(*p);
        let g = coframe_metric(e, l)?;
        let recovered = intrinsic_relative_log(e, &g)?;
        Ok::<f64, GeometryError>(
            sym3_coefficients(recovered)
                .iter()
                .zip(c)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max),
        )
    })
    .collect::<Result<Vec<_>, _>>()?
    .into_iter()
    .fold(0.0, f64::max);
    let affine = Mat3::new(1.2, 0.1, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 1.1);
    let g = curved_metric(Vec3::new(0.1, -0.12, 0.07))?;
    let lhs = chart_transform_covariant(&g, affine)?;
    let inv = affine.try_inverse().unwrap();
    let rhs = Sym3::try_new(inv.transpose() * g.matrix() * inv)?;
    let chart = (lhs.matrix() - rhs.matrix()).norm() / rhs.matrix().norm();
    let q = Mat3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let e0 = folded_dr(Vec3::new(0.1, -0.1, 0.0));
    let j = Mat3::new(1.1, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0);
    let s0 = Mat3::new(0.1, 0.02, 0.01, 0.02, -0.1, 0.03, 0.01, 0.03, 0.04);
    let g0 = coframe_metric(e0, s0)?;
    let ep = q * e0 * j.try_inverse().unwrap();
    let gp = coframe_metric(ep, q * s0 * q.transpose())?;
    let combined_error =
        (gp.matrix() - chart_transform_covariant(&g0, j)?.matrix()).norm() / g0.matrix().norm();
    let relative_covariance = (intrinsic_relative_log(ep, &gp)?
        - q * intrinsic_relative_log(e0, &g0)? * q.transpose())
    .norm();
    let bad_orientation = if coframe_metric(-e0, s0).is_err() {
        0.0
    } else {
        1.0
    };
    let bad_relative_orientation = if intrinsic_relative_log(-e0, &g0).is_err() {
        0.0
    } else {
        1.0
    };
    let singular_coframe = if coframe_metric(Mat3::zeros(), s0).is_err() {
        0.0
    } else {
        1.0
    };
    let rank_deficient_pullback = if pullback(Mat3::zeros(), &g0).is_err() {
        0.0
    } else {
        1.0
    };
    let nonfinite_spd = if Sym3::try_new(Mat3::from_element(f64::NAN)).is_err() {
        0.0
    } else {
        1.0
    };
    let non_lower_cholesky =
        if Sym3::from_cholesky(Mat3::new(1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)).is_err() {
            0.0
        } else {
            1.0
        };
    let nonlinear_bad = {
        let bad_z = [2.0, -2.0, 2.0, -2.0, 2.0, -2.0];
        let j = flat_map_jacobian(Vec3::new(0.25, 0.25, 0.25), bad_z);
        let sigma = nalgebra::linalg::SVD::new(j, false, false)
            .singular_values
            .min();
        if j.determinant() < 0.25 || sigma < 0.25 {
            0.0
        } else {
            1.0
        }
    };
    let curve: Vec<Vec3> = (0..=128)
        .map(|i| {
            let t = i as f64 / 128.0;
            Vec3::new(
                -0.2 + 0.4 * t,
                0.08 * (2.0 * std::f64::consts::PI * t).sin(),
                -0.1 + 0.2 * t * t,
            )
        })
        .collect();
    let euclidean = Sym3::try_new(Mat3::identity())?;
    let old_length = curve_length(|_| Ok(euclidean.clone()), &curve)?;
    let curve_new: Vec<Vec3> = curve.iter().map(|p| affine * p).collect();
    let chart_length = curve_length(
        |_| chart_transform_covariant(&euclidean, affine),
        &curve_new,
    )?;
    let length_error = (old_length - chart_length).abs() / old_length;
    let z = [0.004, -0.003, 0.0035, -0.0025, 0.003, -0.002];
    let audit_points: Vec<Vec3> = (-6..=6)
        .flat_map(|i| {
            (-6..=6).flat_map(move |j| {
                (-2..=2).map(move |k| Vec3::new(i as f64 / 8.0, j as f64 / 8.0, k as f64 / 4.0))
            })
        })
        .collect();
    let atlas_points: Vec<Vec3> = (-8..=8)
        .flat_map(|i| {
            (-8..=8).flat_map(move |j| {
                (-4..=4).map(move |k| Vec3::new(i as f64 / 8.0, j as f64 / 8.0, k as f64 / 4.0))
            })
        })
        .collect();
    let min_fold_det = atlas_points
        .par_iter()
        .map(|p| folded_dr(*p).determinant().abs())
        .reduce(|| f64::INFINITY, f64::min);
    let min_signed_fold_det = atlas_points
        .par_iter()
        .map(|p| folded_dr(*p).determinant())
        .reduce(|| f64::INFINITY, f64::min);
    let min_fold_sigma = atlas_points
        .par_iter()
        .map(|p| {
            nalgebra::linalg::SVD::new(folded_dr(*p), false, false)
                .singular_values
                .min()
        })
        .reduce(|| f64::INFINITY, f64::min);
    let min_j_perp = atlas_points
        .par_iter()
        .map(|p| folded_j_perp(*p))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let max_tau_curvature = atlas_points
        .par_iter()
        .map(|p| folded_principal_curvatures(*p).map(|(k1, k2)| 0.05 * k1.abs().max(k2.abs())))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(0.0, f64::max);
    let nonneighbor_distance = folded_nonneighbor_min_distance(&atlas_points);
    let h_riemann = audit_points
        .par_iter()
        .map(|p| curvature3(folded_h, *p, 0.002).map(|c| c.riemann_invariant_norm))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(0.0, f64::max);
    let h_direct = audit_points
        .par_iter()
        .map(|p| {
            curvature3(folded_h, *p, 0.002)
                .map(|c| (c.riemann_direct_norm, c.riemann_mixed_difference))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold((0.0_f64, 0.0_f64), |a, b| (a.0.max(b.0), a.1.max(b.1)));
    let flat_riemann = audit_points
        .par_iter()
        .map(|p| curvature3(|q| flat_metric(q, z), *p, 0.002).map(|c| c.riemann_invariant_norm))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(0.0, f64::max);
    let flat_direct = audit_points
        .par_iter()
        .map(|p| {
            curvature3(|q| flat_metric(q, z), *p, 0.002)
                .map(|c| (c.riemann_direct_norm, c.riemann_mixed_difference))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold((0.0_f64, 0.0_f64), |a, b| (a.0.max(b.0), a.1.max(b.1)));
    // Check the direct contraction and the 3D identity at each sampled point,
    // rather than comparing independent maxima after reduction.
    let mixed_audit = |metric: &dyn Fn(Vec3) -> Result<Sym3, GeometryError>| -> Result<(f64, f64), GeometryError> {
        let points = audit_points.iter().map(|p| {
            curvature3(|q| metric(q), *p, 0.002).map(|c| {
                let tol = 2e-6 + 1e-8 * c.riemann_direct_norm.max(c.riemann_invariant_norm);
                (c.riemann_mixed_difference, c.riemann_mixed_difference - tol)
            })
        }).collect::<Result<Vec<_>, _>>()?;
        Ok(points.into_iter().fold((0.0_f64, f64::NEG_INFINITY), |a,b| (a.0.max(b.0), a.1.max(b.1))))
    };
    let h_mixed_pointwise = mixed_audit(&folded_h)?;
    let flat_mixed_pointwise = mixed_audit(&|q| flat_metric(q, z))?;
    let flat_j_min = atlas_points
        .par_iter()
        .map(|p| flat_map_jacobian(*p, z).determinant())
        .reduce(|| f64::INFINITY, f64::min);
    let flat_sigma_min = atlas_points
        .par_iter()
        .map(|p| {
            nalgebra::linalg::SVD::new(flat_map_jacobian(*p, z), false, false)
                .singular_values
                .min()
        })
        .reduce(|| f64::INFINITY, f64::min);
    let flat_tensor_error = atlas_points
        .par_iter()
        .map(|p| {
            let a = flat_metric(*p, z)?;
            let b = flat_metric_expected(*p, z)?;
            Ok::<f64, GeometryError>((a.matrix() - b.matrix()).norm() / b.matrix().norm())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(0.0, f64::max);
    let curved = curvature3(curved_metric, Vec3::zeros(), 0.002)?;
    let curved_mixed_margin = curved.riemann_mixed_difference
        - (2e-6
            + 1e-8
                * curved
                    .riemann_direct_norm
                    .max(curved.riemann_invariant_norm));
    let observed = curved.scalar;
    let oracle = -0.56 * SymmetricEigen::new(s).eigenvalues.map(|x| (-x).exp()).sum();
    let curved_err = (observed - oracle).abs() / oracle.abs();
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1);
    let serial = canonical_parallel_aggregate(1)?;
    let parallel = canonical_parallel_aggregate(max_threads)?;
    let deterministic = if serial == parallel { 0.0 } else { 1.0 };
    Ok(vec![
        FixtureReport {
            name: "symmetric_exp_log",
            value: exp_log,
            threshold: 1e-11,
            pass: exp_log <= 1e-11,
        },
        FixtureReport {
            name: "cholesky_reconstruction",
            value: chol_error,
            threshold: 1e-12,
            pass: chol_error <= 1e-12,
        },
        FixtureReport {
            name: "six_component_basis_reconstruction",
            value: basis_error,
            threshold: 1e-13,
            pass: basis_error <= 1e-13,
        },
        FixtureReport {
            name: "six_component_spatial_exp_coframe_log_max",
            value: spatial_recovery,
            threshold: 1e-11,
            pass: spatial_recovery <= 1e-11,
        },
        FixtureReport {
            name: "affine_chart",
            value: chart,
            threshold: 1e-10,
            pass: chart <= 1e-10,
        },
        FixtureReport {
            name: "combined_coframe_chart_gauge",
            value: combined_error,
            threshold: 1e-10,
            pass: combined_error <= 1e-10,
        },
        FixtureReport {
            name: "relative_log_combined_chart_gauge",
            value: relative_covariance,
            threshold: 1e-10,
            pass: relative_covariance <= 1e-10,
        },
        FixtureReport {
            name: "bad_coframe_orientation_rejected",
            value: bad_orientation,
            threshold: 0.0,
            pass: bad_orientation == 0.0,
        },
        FixtureReport {
            name: "bad_relative_log_orientation_rejected",
            value: bad_relative_orientation,
            threshold: 0.0,
            pass: bad_relative_orientation == 0.0,
        },
        FixtureReport {
            name: "singular_coframe_rejected",
            value: singular_coframe,
            threshold: 0.0,
            pass: singular_coframe == 0.0,
        },
        FixtureReport {
            name: "rank_deficient_pullback_rejected",
            value: rank_deficient_pullback,
            threshold: 0.0,
            pass: rank_deficient_pullback == 0.0,
        },
        FixtureReport {
            name: "nonfinite_spd_rejected",
            value: nonfinite_spd,
            threshold: 0.0,
            pass: nonfinite_spd == 0.0,
        },
        FixtureReport {
            name: "non_lower_triangular_cholesky_rejected",
            value: non_lower_cholesky,
            threshold: 0.0,
            pass: non_lower_cholesky == 0.0,
        },
        FixtureReport {
            name: "nonlinear_DF_det_sigma_failure_detected",
            value: nonlinear_bad,
            threshold: 0.0,
            pass: nonlinear_bad == 0.0,
        },
        FixtureReport {
            name: "sampled_curve_length_chart",
            value: length_error,
            threshold: 1e-9,
            pass: length_error <= 1e-9,
        },
        // A lower-bound check only: never a projection or a clamp.
        FixtureReport {
            name: "F-H_atlas_det_floor",
            value: min_fold_det,
            threshold: 0.025,
            pass: min_fold_det >= 0.025,
        },
        FixtureReport {
            name: "F-H_atlas_signed_det_min",
            value: min_signed_fold_det,
            threshold: 0.0,
            pass: min_signed_fold_det > 0.0,
        },
        FixtureReport {
            name: "F-H_atlas_sigma_min",
            value: min_fold_sigma,
            threshold: 0.04,
            pass: min_fold_sigma >= 0.04,
        },
        FixtureReport {
            name: "F-H_atlas_tau_max_principal_curvature",
            value: max_tau_curvature,
            threshold: 0.25,
            pass: max_tau_curvature <= 0.25,
        },
        FixtureReport {
            name: "F-H_atlas_J_perp_min",
            value: min_j_perp,
            threshold: 0.5625,
            pass: min_j_perp >= 0.5625,
        },
        FixtureReport {
            name: "F-H_nonneighbor_physical_separation_min",
            value: nonneighbor_distance,
            threshold: 0.02,
            pass: nonneighbor_distance > 0.02,
        },
        FixtureReport {
            name: "F-H_riemann_invariant_norm",
            value: h_riemann,
            threshold: 1e-5,
            pass: h_riemann <= 1e-5,
        },
        FixtureReport {
            name: "F-H_riemann_direct_norm",
            value: h_direct.0,
            threshold: 1e-5,
            pass: h_direct.0 <= 1e-5,
        },
        FixtureReport {
            name: "F-FLAT_riemann_invariant_norm",
            value: flat_riemann,
            threshold: 1e-5,
            pass: flat_riemann <= 1e-5,
        },
        FixtureReport {
            name: "F-H_riemann_mixed_difference",
            value: h_mixed_pointwise.0,
            threshold: 2.0000001e-6,
            pass: h_mixed_pointwise.1 <= 0.0,
        },
        FixtureReport {
            name: "F-H_riemann_mixed_max_pointwise_margin",
            value: h_mixed_pointwise.1,
            threshold: 0.0,
            pass: h_mixed_pointwise.1 <= 0.0,
        },
        FixtureReport {
            name: "F-FLAT_riemann_direct_norm",
            value: flat_direct.0,
            threshold: 1e-5,
            pass: flat_direct.0 <= 1e-5,
        },
        FixtureReport {
            name: "F-FLAT_riemann_mixed_difference",
            value: flat_mixed_pointwise.0,
            threshold: 2.0000001e-6,
            pass: flat_mixed_pointwise.1 <= 0.0,
        },
        FixtureReport {
            name: "F-FLAT_riemann_mixed_max_pointwise_margin",
            value: flat_mixed_pointwise.1,
            threshold: 0.0,
            pass: flat_mixed_pointwise.1 <= 0.0,
        },
        FixtureReport {
            name: "F-FLAT_det_DF_min",
            value: flat_j_min,
            threshold: 0.25,
            pass: flat_j_min >= 0.25,
        },
        FixtureReport {
            name: "F-FLAT_sigma_DF_min",
            value: flat_sigma_min,
            threshold: 0.25,
            pass: flat_sigma_min >= 0.25,
        },
        FixtureReport {
            name: "F-FLAT_tensor_law_relative",
            value: flat_tensor_error,
            threshold: 1e-10,
            pass: flat_tensor_error <= 1e-10,
        },
        FixtureReport {
            name: "F-CURVED_origin_relative",
            value: curved_err,
            threshold: 1e-3,
            pass: curved_err <= 1e-3 && observed.abs() >= 1e-2,
        },
        FixtureReport {
            name: "F-CURVED_origin_scalar",
            value: observed,
            threshold: 0.0,
            pass: observed.is_finite(),
        },
        FixtureReport {
            name: "F-CURVED_riemann_invariant_norm",
            value: curved.riemann_invariant_norm,
            threshold: 1e-2,
            pass: curved.riemann_invariant_norm > 1e-2,
        },
        FixtureReport {
            name: "F-CURVED_riemann_direct_norm",
            value: curved.riemann_direct_norm,
            threshold: 1e-2,
            pass: curved.riemann_direct_norm > 1e-2,
        },
        FixtureReport {
            name: "F-CURVED_riemann_mixed_difference",
            value: curved.riemann_mixed_difference,
            threshold: 2.0000001e-6,
            pass: curved.riemann_mixed_difference
                <= 2e-6
                    + 1e-8
                        * curved
                            .riemann_direct_norm
                            .max(curved.riemann_invariant_norm),
        },
        FixtureReport {
            name: "F-CURVED_riemann_mixed_margin",
            value: curved_mixed_margin,
            threshold: 0.0,
            pass: curved_mixed_margin <= 0.0,
        },
        FixtureReport {
            name: "serial_rayon",
            value: deterministic,
            threshold: 0.0,
            pass: deterministic == 0.0,
        },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_non_spd() {
        assert!(Sym3::try_new(Mat3::new(1., 2., 0., 2., 1., 0., 0., 0., 1.)).is_err());
    }
    #[test]
    fn all_six_components_survive_coframe() {
        let g = coframe_metric(
            Mat3::identity(),
            Mat3::new(0.2, 0.1, 0.07, 0.1, -0.1, 0.05, 0.07, 0.05, 0.03),
        )
        .unwrap();
        assert!(
            g.matrix()[(0, 1)].abs() > 1e-4
                && g.matrix()[(0, 2)].abs() > 1e-4
                && g.matrix()[(1, 2)].abs() > 1e-4
        );
    }
    #[test]
    fn flat_map_has_full_rank() {
        assert_eq!(
            matrix_rank(flat_map_jacobian(
                Vec3::new(0.13, -0.19, 0.21),
                [0.004, -0.003, 0.0035, -0.0025, 0.003, -0.002]
            )),
            3
        );
    }
    #[test]
    fn parallel_is_index_stable() {
        assert_eq!(deterministic_records(32), deterministic_records(32));
    }
    #[test]
    fn chacha_tuple_seed_is_stable() {
        let seed = CanonicalSeedTuple {
            version: 1,
            master_seed: "seed".into(),
            route: "route".into(),
            generator: "generator".into(),
            dataset: 0,
            split: "fit".into(),
            circuit: 0,
            condition: "c".into(),
            force: "f".into(),
            path: 0,
            candidate: "m".into(),
        };
        let mut left = rng_for_seed(&seed);
        let mut right = rng_for_seed(&seed);
        assert_eq!(left.next_u64(), right.next_u64());
    }
    #[test]
    fn raised_first_index_uses_metric_not_inverse() {
        let g = Mat3::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0);
        let inv = g.try_inverse().unwrap();
        // Only R^0_{1 0 1}=1: g_00 (g^11)^2 = 2/9.
        let correct = g[(0, 0)] * inv[(1, 1)] * inv[(0, 0)] * inv[(1, 1)];
        let legacy = inv[(0, 0)] * inv[(1, 1)] * inv[(0, 0)] * inv[(1, 1)];
        assert!(
            (correct - 1.0 / 9.0).abs() <= 1e-14,
            "contracted paired symmetry gives 1/9"
        );
        assert!((legacy - 1.0 / 36.0).abs() <= 1e-14);
        assert!((correct - legacy).abs() > 1e-3);
    }
}
