"""같은 최하위 양의 작용을 직접 적분하거나 전체 상위 커널로 전달한다.

Finite-depth post hoc diagnostic. Different mesh depths are supplied model families,
not a demonstrated autonomous RG trajectory or a physical fixed point.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np

from conditional_composition import reference as r

HERE=Path(__file__).resolve().parent


def edges(cell):
    return [tuple(sorted(edge)) for edge in combinations(cell,2)]


def _positive_precision(value):
    """유한 대칭 양의 정부호인 정밀도만 허용한다."""
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("precision must be square")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
        raise ValueError("precision must be finite and symmetric")
    matrix = (matrix + matrix.T) / 2
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("precision must be positive definite") from exc
    return matrix


def _validated_depth_step(depth, step):
    if isinstance(depth, bool) or not isinstance(depth, int) or depth not in (1, 2, 3):
        raise ValueError("depth must be an integer from 1 to 3")
    if not math.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive")


def assemble(cells,kernels,boundary):
    if len(cells) != len(kernels):
        raise ValueError("each cell requires exactly one kernel")
    if len(set(boundary)) != len(boundary):
        raise ValueError("boundary edges must be unique")
    ordered=list(boundary)
    ordered.extend(sorted(set(edge for cell in cells for edge in edges(cell))-set(boundary)))
    index={edge:i for i,edge in enumerate(ordered)}
    matrix=np.zeros((len(ordered),len(ordered)))
    for cell,kernel in zip(cells,kernels):
        indices=[index[edge] for edge in edges(cell)]
        kernel=np.asarray(kernel,dtype=float)
        if kernel.shape != (len(indices),len(indices)) or not np.isfinite(kernel).all():
            raise ValueError("kernel must be finite and match the cell edges")
        kernel = _positive_precision(kernel)
        matrix[np.ix_(indices,indices)]+=kernel
    return matrix


def eliminate(matrix,retained):
    """Return Schur precision and log Gaussian integral in unique-edge Lebesgue measure."""
    matrix = _positive_precision(matrix)
    if isinstance(retained, bool) or not isinstance(retained, int) or not 0 <= retained <= len(matrix):
        raise ValueError("retained must be an integer between zero and the matrix size")
    a=matrix[:retained,:retained]
    b=matrix[:retained,retained:]
    d=matrix[retained:,retained:]
    factor=np.linalg.cholesky(d)
    kernel=a-b @ np.linalg.solve(d,b.T)
    log_weight=0.5*len(d)*math.log(2*math.pi)-float(np.log(np.diag(factor)).sum())
    return (kernel+kernel.T)/2,log_weight


def depth_result(depth,step):
    _validated_depth_step(depth, step)
    points=r.points_from_squared(np.full(10,2.0))
    levels=[[tuple(range(5))]]
    for _ in range(depth):
        levels.append(r.refine(levels[-1],points))
    cells=levels[-1]
    kappas=r.equal_split_kappas(cells,tuple(range(5)),np.full(10,np.pi))
    kernels=[]
    minimum=math.inf
    for cell,kappa in zip(cells,kappas):
        lengths=r.cell_lengths(cell,points)
        h=r.richardson_hessian(lambda x:r.simplex_action(x,kappa),lengths,step)
        w,v=np.linalg.eigh(h)
        minimum=min(minimum,float(np.min(np.abs(w))))
        kernels.append((v*np.abs(w)) @ v.T)
    root_edges=edges(levels[0][0])
    direct_matrix=assemble(cells,kernels,root_edges)
    direct,log_direct=eliminate(direct_matrix,10)
    factors=[0.0]*len(cells)
    for level in range(depth-1,-1,-1):
        next_kernels=[]
        next_factors=[]
        for i,parent in enumerate(levels[level]):
            group=slice(5*i,5*i+5)
            matrix=assemble(levels[level+1][group],kernels[group],edges(parent))
            reduced,local_factor=eliminate(matrix,10)
            next_kernels.append(reduced)
            next_factors.append(local_factor+sum(factors[group]))
        kernels,factors=next_kernels,next_factors
    residual=float(np.linalg.norm(kernels[0]-direct)/np.linalg.norm(direct))
    log_residual=float(factors[0]-log_direct)
    if not math.isfinite(residual) or not math.isfinite(log_residual) or residual>1e-8 or abs(log_residual)>1e-7:
        raise RuntimeError(f"same-measure composition failed: {residual}, {log_residual}")
    covariance=np.linalg.inv(direct)
    return {
        "depth":depth,"leaf_cells":len(cells),"glued_edges":len(direct_matrix),
        "integrated_edges":len(direct_matrix)-10,"min_abs_leaf_hessian_eigenvalue":minimum,
        "kernel_composition_residual":residual,"log_weight_composition_residual":log_residual,
        "log_integral_factor":log_direct,"boundary_precision":direct.tolist(),
        "boundary_covariance":covariance.tolist(),
        "boundary_covariance_eigenvalues":np.linalg.eigvalsh(covariance).tolist(),
    }


def run(max_depth=3,step=0.00002):
    _validated_depth_step(max_depth, step)
    results=[]
    for depth in range(1,max_depth+1):
        result=depth_result(depth,step)
        if results:
            previous=np.asarray(results[-1]["boundary_covariance"])
            w,v=np.linalg.eigh(previous)
            inverse_root=(v*w**-0.5) @ v.T
            current=np.asarray(result["boundary_covariance"])
            result["covariance_ratio_to_previous"]=np.linalg.eigvalsh(inverse_root @ current @ inverse_root).tolist()
        results.append(result)
    return {"scope":"finite-depth supplied Wick model family; unique glued-edge Lebesgue measure; kappa=1",
            "python":sys.version.split()[0],"numpy":np.__version__,"fd_step":step,
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reference_sha256":hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest(),
            "depths":results,"physical_fixed_point_proved":False,"common_metric_selection_proved":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-depth",type=int,choices=(1,2,3),default=3)
    parser.add_argument("--fd-step",type=float,default=0.00002)
    parser.add_argument("--output",type=Path,default=HERE/"recursive_kernel.json")
    args=parser.parse_args()
    if not math.isfinite(args.fd_step) or args.fd_step<=0:
        parser.error("fd-step must be finite and positive")
    result=run(args.max_depth,args.fd_step)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    summary=[{k:v for k,v in row.items() if k not in ("boundary_precision","boundary_covariance")} for row in result["depths"]]
    print(json.dumps(summary,indent=2,allow_nan=False))
