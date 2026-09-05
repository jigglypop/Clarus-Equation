import sys
sys.path.insert(0, r'C:/dev/ce/Clarus-Equation')
import numpy as np
from examples.physics.gravity.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment
t = geometric_self_dual_triple(np.eye(4))
print("triple", np.asarray(t).shape, type(t))
print("gram", np.asarray(plebanski_gram(t)).shape)
a = optimal_internal_alignment(t, geometric_self_dual_triple(np.eye(4) + 0.01 * np.ones((4, 4))))
print("aligned", np.asarray(a.aligned_candidate).shape, [f for f in dir(a) if not f.startswith("_")])
