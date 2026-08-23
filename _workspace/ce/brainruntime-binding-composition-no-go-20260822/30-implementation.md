# BA-TR16 implementation

BA-TR16 generates fresh BA-TR10 source codes, trains four atomic associations
with the frozen BA-TR15 local packet compensation, and uses the cyclic
nonidentity target map `pi=(1,2,3,0)`.

The simultaneous probes combine one source from `{0,1}` with one source from
`{2,3}`. No such pair is presented during learning. The expected output is the
two-element set `{pi(i),pi(j)}` at call 6.

The runtime is unchanged. Its max-relative hidden competition is also checked
as a pure algebraic function and can have at most one positive component.
Independent-union probes clone the same sealed snapshot twice, recall each atom
separately, and combine their target activations offline. This confirms that
both atomic memories exist even when simultaneous routing fails.

