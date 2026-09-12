"""Independent symbolic heavy-source elimination and high-precision audit.

No import of the full-tensor generator or its hand-written amplitude formula.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import sympy as sy


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'verify/ce_record_conversion_scattering.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    for name,digest in data['source_sha256'].items():
        assert hashlib.sha256((root/'verify'/name).read_bytes()).hexdigest()==digest,name
    chapter=next((root/'paper/06_QFT_재설계').glob('80_*.md'))
    frozen=chapter.read_text(encoding='utf-8').split('## 80.2')[0]
    assert hashlib.sha256(frozen.encode()).hexdigest()==data['preregistration_sha256']
    x,xb,y,yb,a,M,ma,mb,sh = sy.symbols('x xb y yb a M ma mb sh')
    r,rb=x*x+y*y,xb*xb+yb*yb
    density=ma*x*xb+mb*y*yb
    leff=sy.expand(-a*a*r*rb+a*a*(M*r-2*density)*(M*rb-2*density)/(M*M-sh))
    poly=sy.Poly(leff,x,xb,y,yb)
    incoming=[x*x,x*xb,xb*xb]
    outgoing=[yb*yb,y*yb,y*y]
    factor=[2,1,2]
    amplitude=sy.Matrix([[sy.factor(poly.coeff_monomial(i*o)*factor[col]*factor[row])
                         for col,i in enumerate(incoming)] for row,o in enumerate(outgoing)])
    expected=4*a*a/(M*M-sh)*sy.Matrix([[sh,-M*ma,0],[-M*mb,2*ma*mb,-M*mb],[0,-M*ma,sh]])
    assert sy.simplify(amplitude-expected)==sy.zeros(3)
    determinant=sy.factor(amplitude.det())
    assert sy.simplify(determinant+128*a**6*ma*mb*sh/(M*M-sh)**2)==0
    assert amplitude[2,0]==amplitude[0,2]==0
    assert sy.limit(amplitude[0,0],sh,0)==0
    symmetry=np.array([2.,1.,2.])
    rows=[]
    largest={'full_tensor_relative_error':0.,'closed_formula_relative_error':0.,
             'cross_section_relative_error':0.,'conditional_output_scaled_error':0.,
             'partial_cut_relative_error':0.}
    for row in data['rows']:
        alpha=0 if row['incoming_species']=='+' else 1
        masses=[sy.Rational(1,10000)*(5+sy.pi/2),sy.Rational(1,10000)*(5-sy.pi/2)]
        emin=sy.Rational(str(row['energy_over_V']))
        subs={a:1/(2*sy.sqrt(15)),M:1,ma:masses[alpha],mb:masses[1-alpha],sh:emin**2}
        high=amplitude.subs(subs).evalf(80)
        exact=np.array(high,float)
        mask=exact!=0
        full=np.array(row['full_tensor_amplitude_matrix_real'])
        recorded=np.array(row['amplitude_matrix_real'])
        full_error=float(np.max(abs((full[mask]-exact[mask])/exact[mask])))
        closed_error=float(np.max(abs((recorded[mask]-exact[mask])/exact[mask])))
        assert np.max(abs(full[~mask]))<1e-12
        ba=sy.sqrt(1-4*masses[alpha]**2/emin**2)
        bb=sy.sqrt(1-4*masses[1-alpha]**2/emin**2)
        flux=(bb/(16*sy.pi*emin**2*ba)).evalf(80)
        sigma=sy.Matrix([[flux*high[i,j]**2/(2 if i!=1 else 1) for j in range(3)] for i in range(3)])
        sig=np.array(sigma,float)
        got=np.array(row['sigma_matrix_times_V2'])
        sigma_error=float(np.max(abs((got[mask]-sig[mask])/sig[mask])))
        d,z=float(high[0,0]),float(high[1,0])
        c=float(flux)
        e=c*np.array([[d*d/2+z*z,z*z],[z*z,d*d/2+z*z]])
        map_data=row['selected_map']
        assert np.max(abs(e-np.array(map_data['cross_section_effect_times_V2'])))/np.max(abs(e))<1e-8
        # Integral of the two-body Lorentz phase space, including final symmetry.
        spectral=np.array([[d*d/2+z*z,z*z],[z*z,d*d/2+z*z]])*float(bb)/(8*np.pi)
        cut_error=float(np.max(abs(spectral-np.array(map_data['required_beta_cut_in_2_Im_forward_loop'])))/np.max(abs(spectral)))
        phase_error=0.
        for phase in map_data['phase_preparations']:
            phi=sy.pi*sy.Rational(str(phase['phase_over_pi']))
            # Coherent sum first, then the sum over three orthogonal final channels.
            vec=sy.Matrix([high[0,0]/2,high[1,0]*(1+sy.exp(sy.I*phi))/sy.sqrt(2),
                           high[2,2]*sy.exp(sy.I*phi)/2])
            norm=sy.re((sy.conjugate(vec).T*vec)[0]).evalf(80)
            sigphase=float(flux*norm)
            density=(vec*sy.conjugate(vec).T/norm).evalf(60)
            density_np=np.array(density,complex)
            got_rho=np.array(phase['conditional_density_real'])+1j*np.array(phase['conditional_density_imag'])
            phase_error=max(phase_error,abs(phase['sigma_total_times_V2']/sigphase-1),
                            float(np.max(abs(density_np-got_rho))))
        for key,value in zip(largest,[full_error,closed_error,sigma_error,phase_error,cut_error]):
            largest[key]=max(largest[key],value)
        assert max(full_error,closed_error,sigma_error,phase_error,cut_error)<1e-8
        rows.append({'energy_over_V':float(emin),'incoming_species':row['incoming_species'],
                     'M_PP_PP_80_digit':str(high[0,0]),'sigma_PP_input_total_V2':float(sum(sigma[:,0]))})
    result={'candidate':'CE-BR2-independent-audit','preregistration_sha256':data['preregistration_sha256'],
        'generator_result_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'sympy':sy.__version__},
        'amplitude_from_nonlocal_source':[[str(amplitude[i,j]) for j in range(3)] for i in range(3)],
        'amplitude_determinant':str(determinant),'PP_PP_constant_soft_term_zero':True,
        'max_errors':largest,'rows':rows,'audit_passed':True,'scientific_success':False,'full_joint_rmse':None}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'audit_passed':True,'max_errors':largest,'matrix_determinant':str(determinant)}))


if __name__=='__main__':
    main()
