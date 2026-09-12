"""CE-BR2: on-shell tree conversion between both light record species.

All scalar mediator vertices are extracted from the original 82-component
action. Selected cross-section effects are not complete detector instruments.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from ce_singlet_record_observability import build_model
from ce_supersymmetric_record import component_matrices, f_terms


ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT/'paper/06_QFT_재설계/80_두_기록_종류의_산란과_선택_기록.md'
PREREG = '99d1ca28394cccb3e4b00b6fe3b6703622865e52c3d60de8dd705cc2f3680871'
M0, HEAVY, A2 = 1e-4, 1., 1/60
MASSES = M0*np.array([5+np.pi/2,5-np.pi/2])
ENERGIES = [.0015,.003,.01,.05]
TOL, ZERO_TOL = 1e-8, 1e-12
SYMMETRY = np.array([2.,1.,2.])


def maximum(x):
    return float(np.max(np.abs(x)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vertices():
    _,_,reps,matrix,cubic,vacuum = build_model(M0)
    fv,w0 = f_terms(vacuum,matrix,cubic)
    e = np.zeros((82,2),complex)
    e[35,:] = 1/np.sqrt(2)
    e[59,:] = np.array([1,-1])/np.sqrt(2)
    ext = np.column_stack([e[:,0],1j*e[:,0],e[:,1],1j*e[:,1]])/np.sqrt(2)
    internal = np.column_stack([np.eye(82),1j*np.eye(82)])/np.sqrt(2)
    wi,we = w0@internal,w0@ext
    cex = np.einsum('ijk,ja,kb->iab',cubic,ext,ext,optimize=True)
    cie = np.einsum('ijk,ja,kb->iab',cubic,internal,ext,optimize=True)
    first = np.einsum('ki,kej->ije',we.conj(),cie,optimize=True)
    third = np.einsum('ke,kij->ije',wi.conj(),cex,optimize=True)
    g = 2*np.real(first+first.transpose(1,0,2)+third)
    quartic = np.zeros((4,4,4,4))
    for i,j,k,l in np.ndindex((4,)*4):
        quartic[i,j,k,l] = 2*np.real(np.vdot(cex[:,i,j],cex[:,k,l])
            +np.vdot(cex[:,i,k],cex[:,j,l])+np.vdot(cex[:,i,l],cex[:,j,k]))
    hs,_,_,_,_ = component_matrices(vacuum,reps,matrix,cubic)
    eig,u = np.linalg.eigh(hs)
    geig = np.einsum('ije,ef->ijf',g,u,optimize=True)
    gauge = np.einsum('ia,kij,jb->kab',ext.conj(),reps,ext,optimize=True)
    active = [11,93]
    outside = np.setdiff1d(np.arange(164),active)
    err = {
        'vacuum_F_absolute':maximum(fv),
        'unwanted_scalar_mediator_vertex_absolute':maximum(g[:,:,outside]),
        'mixed_mass_species_cubic_vertex_absolute':maximum(g[:2,2:]),
        'light_record_gauge_and_D_vertex_absolute':maximum(gauge),
        'heavy_singlet_mass_squared_absolute':maximum(hs[np.ix_(active,active)]-np.eye(2)),
        'heavy_singlet_mass_mixing_absolute':maximum(hs[np.ix_(active,outside)]),
    }
    assert max(err.values())<ZERO_TOL,err
    return quartic,geig,eig,{'errors':err,'real_scalar_mediators_checked':164,
        'active_mediators':['Re Sigma_Y','Im Sigma_Y'],
        'D_and_gauge_exchange_absent_for_these_external_states':True,
        'fermion_exchange_not_a_tree_four_scalar_topology':True}


def pair_polarizations(species):
    p = np.zeros(4,complex)
    p[2*species:2*species+2] = np.array([1,-1j])/np.sqrt(2)
    anti = p.conj()
    return [(p,p),(p,anti),(anti,anti)]


def amplitude(quartic,g,m2,energy,alpha,cosine):
    beta = 1-alpha
    ma,mb = MASSES[[alpha,beta]]
    shat = energy**2
    pa,pb = np.sqrt(shat/4-np.array([ma,mb])**2)
    that = ma*ma+mb*mb-shat/2+2*pa*pb*cosine
    uhat = 2*(ma*ma+mb*mb)-shat-that
    terms = {k:np.zeros((3,3),complex) for k in ['contact','s','t','u']}
    for col,(v1,v2) in enumerate(pair_polarizations(alpha)):
        for row,(out1,out2) in enumerate(pair_polarizations(beta)):
            v3,v4 = out1.conj(),out2.conj()
            terms['contact'][row,col] = -np.einsum('i,j,k,l,ijkl',v1,v2,v3,v4,quartic)
            for label,inv,left,right in [
                ('s',shat,(v1,v2),(v3,v4)),
                ('t',that,(v1,v3),(v2,v4)),
                ('u',uhat,(v1,v4),(v2,v3))]:
                gl = np.einsum('i,j,ije->e',*left,g)
                gr = np.einsum('i,j,ije->e',*right,g)
                terms[label][row,col] = -np.sum(gl*gr/(inv-m2))
    return sum(terms.values()),terms


def independent_formula(energy,ma,mb):
    shat = energy**2
    return 4*A2/(HEAVY**2-shat)*np.array([
        [shat,-HEAVY*ma,0],[-HEAVY*mb,2*ma*mb,-HEAVY*mb],[0,-HEAVY*ma,shat]])


def angular_checks(matrix,energy,ma,mb):
    shat = energy**2
    ba,bb = np.sqrt(1-4*np.array([ma,mb])**2/shat)
    pa,pb = energy*np.array([ba,bb])/2
    expected = bb/(16*np.pi*shat*ba)*abs(matrix)**2/SYMMETRY[:,None]
    errors = []
    shell = 0.
    for ncos,nphi in [(16,8),(32,12)]:
        nodes,weights = np.polynomial.legendre.leggauss(ncos)
        angular = np.zeros((3,3))
        for ct,weight in zip(nodes,weights):
            for phi in 2*np.pi*np.arange(nphi)/nphi:
                n = np.array([np.sqrt(1-ct*ct)*np.cos(phi),np.sqrt(1-ct*ct)*np.sin(phi),ct])
                pin = np.array([[energy/2,0,0,pa],[energy/2,0,0,-pa]])
                pout = np.array([np.r_[energy/2,pb*n],np.r_[energy/2,-pb*n]])
                shell = max(shell,maximum(pin.sum(axis=0)-pout.sum(axis=0))/energy,
                    maximum(pin[:,0]**2-np.sum(pin[:,1:]**2,axis=1)-ma*ma)/shat,
                    maximum(pout[:,0]**2-np.sum(pout[:,1:]**2,axis=1)-mb*mb)/shat)
                angular += weight*2*np.pi/nphi*bb/(64*np.pi**2*shat*ba)*abs(matrix)**2/SYMMETRY[:,None]
        width = 4*pa*pb
        # Independent integral of d sigma/dt over its physical interval.
        dt = abs(matrix)**2/(64*np.pi*shat*pa*pa)/SYMMETRY[:,None]
        tint = sum(weights)*width/2*dt
        nonzero = expected>0
        errors += [maximum((angular[nonzero]-expected[nonzero])/expected[nonzero]),
                   maximum((tint[nonzero]-expected[nonzero])/expected[nonzero])]
    assert max(errors+[shell])<TOL
    return expected,{'relative_integration_error':max(errors),'normalized_shell_and_momentum_error':shell}


def selected_map(matrix,energy,ma,mb):
    shat = energy**2
    ba,bb = np.sqrt(1-4*np.array([ma,mb])**2/shat)
    weight = bb/(16*np.pi*shat*ba)
    b = matrix[:,[0,2]]/np.sqrt(SYMMETRY[:,None])
    gram = b.conj().T@b
    effect = weight*gram
    cut = bb/(8*np.pi)*gram
    # The cut is a required contribution to 2 Im of the forward loop amplitude.
    assert maximum(cut-2*shat*ba*effect)/maximum(cut)<TOL
    eig = np.linalg.eigvalsh(effect)
    assert eig[0]>0
    rows = []
    for phase in [0.,np.pi/2,np.pi,3*np.pi/2]:
        state = np.array([1,np.exp(1j*phase)])/np.sqrt(2)
        output = b@state
        sigma = float(weight*np.vdot(output,output).real)
        norm = np.vdot(output,output).real
        rho = np.outer(output,output.conj())/norm
        particle_antiparticle = float(weight*abs(output[1])**2)
        d,x = matrix[0,0],matrix[1,0]
        direct = weight*(abs(d)**2/2+abs(x)**2*(1+np.cos(phase)))
        err = max(abs(sigma/direct-1),abs(np.trace(rho)-1),maximum(rho@rho-rho))
        assert err<TOL
        rows.append({'phase_over_pi':float(phase/np.pi),'sigma_total_times_V2':sigma,
            'sigma_particle_antiparticle_times_V2':particle_antiparticle,'conditional_channel_probabilities':np.diag(rho).real.tolist(),
            'conditional_density_real':rho.real.tolist(),'conditional_density_imag':rho.imag.tolist(),
            'normalization_and_coherence_error':float(err)})
    return {'input_basis':['particle_pair','antiparticle_pair'],
        'effect_units':'V^-2; not a dimensionless event probability',
        'cross_section_effect_times_V2':effect.real.tolist(),
        'cross_section_effect_eigenvalues_times_V2':eig.tolist(),
        'required_beta_cut_in_2_Im_forward_loop':cut.real.tolist(),
        'phase_preparations':rows,'full_no_click_instrument_constructed':False}


def main():
    started = time.perf_counter()
    frozen = CHAPTER.read_text(encoding='utf-8').split('## 80.2')[0]
    assert hashlib.sha256(frozen.encode()).hexdigest()==PREREG,'Preregistration changed'
    q,g,m2,vertex_report = vertices()
    rows = []
    max_amp_error,max_zero,max_reciprocity = 0.,0.,0.
    for energy in ENERGIES:
        numerical = []
        for alpha in [0,1]:
            ma,mb = MASSES[[alpha,1-alpha]]
            expected = independent_formula(energy,ma,mb)
            allowed = expected!=0
            for cosine in [-.8,-.2,0.,.4,.9]:
                measured,terms = amplitude(q,g,m2,energy,alpha,cosine)
                error = maximum((measured[allowed]-expected[allowed])/expected[allowed])
                zero = max(maximum(measured[~allowed]),maximum(terms['t']),maximum(terms['u']))
                assert error<TOL and zero<ZERO_TOL,(energy,alpha,error,zero,measured,expected)
                max_amp_error,max_zero = max(max_amp_error,error),max(max_zero,zero)
            numerical.append(measured)
            cross,integration = angular_checks(expected,energy,ma,mb)
            selected = selected_map(expected,energy,ma,mb)
            naive_sigma = np.sqrt((1-4*mb*mb/energy**2)/(1-4*ma*ma/energy**2))/(32*np.pi*energy**2)*(4*A2)**2
            rows.append({'energy_over_V':energy,'incoming_species':'+' if alpha==0 else '-',
                'outgoing_species':'-' if alpha==0 else '+','m_in_over_V':float(ma),'m_out_over_V':float(mb),
                'amplitude_matrix_real':expected.tolist(),
                'full_tensor_amplitude_matrix_real':measured.real.tolist(),
                'PP_to_PP_contact':float(terms['contact'][0,0].real),
                'PP_to_PP_heavy_exchange':float(terms['s'][0,0].real),
                'sigma_matrix_times_V2':cross.tolist(),
                'PP_input_total_conversion_sigma_times_V2':float(cross[:,0].sum()),
                'PP_input_particle_antiparticle_to_identical_ratio':float(cross[1,0]/cross[0,0]),
                'PP_to_PP_sigma_over_contact_only':float(cross[0,0]/naive_sigma),
                'leading_heavy_expansion_amplitude_relative_correction':float(energy**2/(1-energy**2)),
                'integration':integration,'selected_map':selected})
        max_reciprocity = max(max_reciprocity,maximum(numerical[0]-numerical[1].T))
    assert max_reciprocity<ZERO_TOL
    files=[Path(__file__),ROOT/'verify/ce_singlet_record_observability.py',
           ROOT/'verify/ce_supersymmetric_record.py',ROOT/'verify/ce_simple_group_record.py']
    result={'candidate':'CE-BR2','preregistration_sha256':PREREG,
        'source_sha256':{p.name:digest(p) for p in files},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'parameters':{'m0_over_V':M0,'M_Sigma_over_V':HEAVY,'a_squared':A2,
                      'm_plus_over_V':float(MASSES[0]),'m_minus_over_V':float(MASSES[1])},
        'vertex_audit':vertex_report,
        'full_tensor_amplitude_relative_error':max_amp_error,
        'forbidden_or_t_u_channel_absolute_error':max_zero,
        'reverse_species_transpose_absolute_error':max_reciprocity,
        'pair_channel_order':['SS','S_Sbar','Sbar_Sbar'],
        'rows':rows,'scope':{'perturbative_order':'tree two-scalar species conversion only',
            'all_CE_scattering_channels_included':False,'loop_forward_phase_computed':False,
            'spatial_classical_preparation_reused':False,'physical_no_click_instrument':False},
        'verification_passed':True,'scientific_success':False,'full_joint_rmse':None,
        'runtime_seconds':time.perf_counter()-started}
    target=Path(__file__).with_suffix('.json')
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'rows':len(rows),'full_tensor_relative_error':max_amp_error,'zero_channel_error':max_zero,
        'seconds':result['runtime_seconds'],'verification_passed':True,
        'first_row_conversion_sigma_V2':rows[0]['PP_input_total_conversion_sigma_times_V2'],
        'first_row_particle_antiparticle_to_identical_ratio':rows[0]['PP_input_particle_antiparticle_to_identical_ratio']}))


if __name__=='__main__':
    main()
