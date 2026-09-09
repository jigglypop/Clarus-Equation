"""Tree-level type-I seesaw matching and conditional rank-two mass readout.

All mass matrices and their q=Q/Mpl derivatives use the same mass unit/frame.
The UV matrices are supplied, not derived from the dimension measure.
"""
import json
from pathlib import Path
import numpy as np


def match(dirac, heavy, dirac_slope, heavy_slope):
    d, n, dc, nc = [np.asarray(x, complex) for x in (dirac, heavy, dirac_slope, heavy_slope)]
    if d.ndim != 2 or d.shape[0] != 3 or d.shape[1] not in (2,3):
        raise ValueError("3x2 or 3x3 Dirac block required")
    if n.shape != (d.shape[1],)*2 or dc.shape != d.shape or nc.shape != n.shape:
        raise ValueError("matching matrix and slope shapes required")
    if not all(np.isfinite(x).all() for x in (d,n,dc,nc)):
        raise ValueError("finite matrices required")
    for x in (n,nc):
        if not np.allclose(x,x.T,rtol=1e-12,atol=1e-12*np.linalg.norm(x)):
            raise ValueError("heavy Majorana block and slope must be symmetric")
    rdt = np.linalg.solve(n,d.T)
    mass = -d@rdt
    slope = -dc@rdt-d@np.linalg.solve(n,dc.T)+d@np.linalg.solve(n,nc@rdt)
    return {"mass": mass, "slope": slope,
            "heavy_mixing_norm": float(np.linalg.norm(rdt,2)),
            "approximation": "tree_level_leading_seesaw_requires_small_heavy_mixing"}


def rank_two_masses(solar, atmospheric, ordering):
    """m_lightest=0 and NuFIT convention: atm=dm31(NO), dm32(IO), signed."""
    if not np.isfinite([solar,atmospheric]).all() or solar <= 0:
        raise ValueError("positive finite solar splitting required")
    if ordering == "NO" and atmospheric > solar:
        return np.sqrt([0,solar,atmospheric])
    if ordering == "IO" and atmospheric < -solar:
        return np.sqrt([-atmospheric-solar,-atmospheric,0])
    raise ValueError("ordering and signed atmospheric splitting inconsistent")


def report():
    # Table 1, article v2; both analysis variants retained without summing them.
    source = json.loads(Path(__file__).with_name("nufit60_pmns_intervals.json").read_text())
    atm = {"IC19_NO": .002534,"IC19_IO": -.002510,
           "IC24_SK_NO": .002513,"IC24_SK_IO": -.002484}
    rows = {}
    for name,value in atm.items():
        masses = rank_two_masses(.0000749,value,name.rsplit('_',1)[1])
        rows[name] = {"solar_eV2": .0000749,"signed_atmospheric_eV2": value,
                      "masses_eV": masses.tolist(),"sum_eV": float(masses.sum())}
    return {"source": source["source"],"source_html_sha256": source["source_html_sha256"],
            "rows":rows,"assumption":"exactly_two_heavy_neutrinos_tree_level_no_extra_mass_operator",
            "status":"conditional_transform_of_external_splittings_not_their_prediction",
            "joint_rmse":None,"scientific_success":False}


if __name__ == '__main__':
    result=report()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,row in result['rows'].items():
        print(name,row['masses_eV'],row['sum_eV'])
