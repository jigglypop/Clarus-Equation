"""Independent implementation of ZIP RT31 equations 6,7,12,13.

Conformal xi=1/6, stated R^2 matching, supplied Gaussian preparation. The
quadrature is this implementation's choice; no original source code assumed.
"""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def run(cutoff=24.,order=384,dynamic_h=False,end_time=25.,*,
        f=.1,MR=3.,eps=.35,theta0=.7,velocity0=6.,checkpoint_path=None,restart_path=None):
    # All masses here are measured in sqrt(s), with s=1 in the equations.
    if not (f>0 and MR>0 and 0<eps<.5 and cutoff>0 and order>=16 and end_time>0):
        raise ValueError('Require positive scales, 0<epsilon/s<1/2 and adequate quadrature.')
    L=3*MR**2+1/(16*np.pi**2);ca=1/(160*np.pi**2)
    nodes,w=leggauss(order);k=(nodes+1)*cutoff/2
    weights=w*cutoff/2*k*k/(2*np.pi**2)
    def spectrum(theta):
        angles=(theta+2*np.pi*np.arange(3))/3
        return 1+2*eps*np.cos(angles),-2*eps/3*np.sin(angles)
    xr,_=spectrum(np.pi)
    ref=np.sum(xr*xr*(np.log(xr)-1.5))/(32*np.pi**2)
    def potential(x,A):
        return (np.sum(x*x*(np.log(x)-1.5))/(32*np.pi**2)-ref,
                np.sum(A*x*(np.log(x)-1))/(16*np.pi**2))
    def Hubble(rho):
        disc=L*L-4*ca*rho
        assert disc>0 and rho>0
        return np.sqrt(2*rho/(L+np.sqrt(disc)))
    x0,A0=spectrum(theta0);U0,_=potential(x0,A0)
    omega0=np.sqrt(k[None,:]**2+x0[:,None])
    def preparation(H):
        rate=(2*H*x0[:,None]+A0[:,None]*velocity0)/(4*omega0**2)
        v=-rate/(2*omega0);n=v*v;u=-n
        return np.array([n,u,v])
    def initial_rho(H):
        n=preparation(H)[0]
        return .5*f*f*velocity0**2+U0+np.sum(weights[None,:]*2*omega0*n)
    # Scale the root variable: absolute H tolerances otherwise fail for widely
    # separated Planck and particle scales. Stay on the low-curvature branch.
    rho0=initial_rho(0.)
    if rho0<=0:raise ValueError('This expanding preparation requires positive initial energy.')
    Hscale=np.sqrt(rho0/L)
    def constraint_scaled(h):
        H=h*Hscale
        return (L*H*H-ca*H**4-initial_rho(H))/rho0
    upper=2.
    Hturn=np.sqrt(L/(2*ca))
    while constraint_scaled(upper)<=0 and upper*Hscale<Hturn:
        upper=min(2*upper,Hturn/Hscale)
    if constraint_scaled(upper)<=0:raise ValueError('No low-curvature initial root in this preparation.')
    H0=Hscale*brentq(constraint_scaled,0.,upper,xtol=1e-13)
    state0=np.r_[theta0,velocity0,0.,preparation(H0).ravel()]
    if dynamic_h:state0=np.r_[state0,H0]
    start_time=0.
    parameters=np.array([f,MR,eps,theta0,velocity0,cutoff,order,float(dynamic_h)])
    if restart_path is not None:
        with np.load(restart_path,allow_pickle=False) as saved:
            if not np.array_equal(saved['parameters'],parameters):
                raise ValueError('Restart requires identical model, initial preparation and quadrature.')
            if not (np.array_equal(saved['k'],k) and np.array_equal(saved['weights'],weights)):
                raise ValueError('Restart momentum grid differs; explicit state remapping required.')
            state0=saved['state'].copy();start_time=float(saved['time'])
        if end_time<=start_time:raise ValueError('End time must follow saved time.')
    def quantities(y):
        theta,velocity,loga=y[:3];a=np.exp(loga)
        n,u,v=y[3:-1 if dynamic_h else None].reshape(3,3,order)
        x,A=spectrum(theta);U,U1=potential(x,A)
        omega=np.sqrt(k[None,:]**2+a*a*x[:,None])
        rhoex=np.sum(weights[None,:]*2*omega*n)/a**4
        pex=np.sum(weights[None,:]*(2*omega*n-2*a*a*x[:,None]/omega*(n+u)))/(3*a**4)
        J=np.sum(weights[None,:]*A[:,None]/omega*(n+u))/a**2
        rho=.5*f*f*velocity**2+U+rhoex
        H=y[-1] if dynamic_h else Hubble(rho)
        return a,n,u,v,x,A,omega,rhoex,pex,J,U1,H
    def rhs(t,y):
        a,n,u,v,x,A,omega,re,pe,J,U1,H=quantities(y)
        velocity=y[1]
        rate=a*a*(2*H*x[:,None]+A[:,None]*velocity)/(4*omega**2)
        dn=2*rate*u
        du=rate*(1+2*n)+2*omega/a*v
        dv=-2*omega/a*u
        result=np.r_[velocity,-3*H*velocity-(U1+J)/(f*f),H,np.array([dn,du,dv]).ravel()]
        if dynamic_h:result=np.r_[result,-3*(f*f*velocity**2+re+pe)/(2*L-4*ca*H*H)]
        return result
    times=np.linspace(start_time,end_time,max(51,int((end_time-start_time)*10)+1))
    sol=solve_ivp(rhs,(start_time,end_time),state0,method='DOP853',rtol=2e-9,atol=2e-12,t_eval=times)
    assert sol.success,sol.message
    if checkpoint_path is not None:
        destination=Path(checkpoint_path);destination.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(destination,state=sol.y[:,-1],time=end_time,k=k,weights=weights,
                            parameters=parameters,schema_version=1)
    results=[]
    constraints=[]
    for y in sol.y.T:
        a,n,u,v,x,A,omega,re,pe,J,U1,H=quantities(y)
        invariant=float(np.max(abs((1+2*n)**2-4*(u*u+v*v)-1)))
        U,_=potential(x,A)
        rho=.5*f*f*y[1]**2+U+re
        constraints.append(float(abs(L*H*H-ca*H**4-rho)/rho))
        numbers=2*np.sum(weights[None,:]*n,axis=1)
        cold_energy=float(numbers@np.sqrt(x)/a**3)
        gas_pressure=float(np.sum(weights[None,:]*2*k[None,:]**2*n/omega)/(3*a**4))
        coherence_pressure=float(-2*np.sum(weights[None,:]*x[:,None]*u/omega)/(3*a**2))
        particle_force=float(np.sum(weights[None,:]*A[:,None]*n/omega)/a**2)
        coherence_force=float(np.sum(weights[None,:]*A[:,None]*u/omega)/a**2)
        assert abs((gas_pressure+coherence_pressure)-pe)<1e-12
        assert abs(particle_force+coherence_force-J)<1e-12
        assert cold_energy<=re*(1+1e-10)
        rate=a*a*(2*H*x[:,None]+A[:,None]*y[1])/(4*omega**2)
        number_derivatives=4*np.sum(weights[None,:]*rate*u,axis=1)
        results.append(dict(a=float(a),H=float(H),theta=float(y[0]),theta_velocity=float(y[1]),rho_ex=float(re),
                            gaussian_invariant_max_abs=invariant,
                            state_moments=dict(comoving_particle_numbers=numbers.tolist(),
                              number_fractions=(numbers/numbers.sum()).tolist(),
                              comoving_number_time_derivatives=number_derivatives.tolist(),
                              rest_mass_energy=cold_energy,kinetic_energy_fraction=float(1-cold_energy/re),
                              particle_gas_pressure=gas_pressure,coherence_pressure=coherence_pressure,
                              total_state_pressure=float(pe),particle_force=particle_force,
                              coherence_force=coherence_force,total_state_force=float(J))))
    assert max(constraints)<1e-7
    windows=[]
    midpoint=start_time+.5*(end_time-start_time)
    threequarters=start_time+.75*(end_time-start_time)
    for start,end in [(midpoint,threequarters),(threequarters,end_time)]:
        mask=(times>=start)&(times<=end);tt=times[mask]
        rows=[r for r,keep in zip(results,mask) if keep]
        integrate=lambda values:float(np.trapezoid(values,tt))
        energy=integrate([r['rho_ex'] for r in rows])
        m=[r['state_moments'] for r in rows]
        pressure=integrate([r['total_state_pressure'] for r in m])
        particle_force=integrate([r['particle_force'] for r in m])
        coherence_force=integrate([r['coherence_force'] for r in m])
        windows.append(dict(time_start=start,time_end=end,pressure_over_energy_integrals=pressure/energy,
                            kinetic_energy_fraction_integrated=1-integrate([r['rest_mass_energy'] for r in m])/energy,
                            integrated_particle_force=particle_force,integrated_coherence_force=coherence_force,
                            initial_number_fractions=m[0]['number_fractions'],final_number_fractions=m[-1]['number_fractions'],
                            total_comoving_number_change_fraction=float(sum(m[-1]['comoving_particle_numbers'])/sum(m[0]['comoving_particle_numbers'])-1)))
    return dict(cutoff=cutoff,quadrature_order=order,dynamic_H=dynamic_h,
                start_time=start_time,end_time=end_time,time_windows=windows,
                restarted=restart_path is not None,
                initial=results[0],final=results[-1],friedmann_max_relative=max(constraints),nfev=sol.nfev,
                supplied_inputs=dict(s=1,epsilon=eps,f=f,M_R=MR,theta0=theta0,theta_dot0=velocity0,Lambda_R=0),
                dimensionless_ratios=dict(f_over_M_R=f/MR,sqrt_s_over_M_R=1/MR),
                fitted_parameters=[],observational_rmse=None)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--cutoff',type=float,default=24)
    parser.add_argument('--order',type=int,default=384);parser.add_argument('--dynamic-h',action='store_true')
    parser.add_argument('--end-time',type=float,default=25.);args=parser.parse_args()
    result=run(args.cutoff,args.order,args.dynamic_h,args.end_time)
    suffix='_dynamicH' if args.dynamic_h else ''
    if args.end_time!=25:suffix+=f'_t{args.end_time:g}'
    path=Path(__file__).with_name(f'ce_rt31_flrw_K{args.cutoff:g}_n{args.order}{suffix}.json')
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
