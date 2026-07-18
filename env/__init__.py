"""LIBERO vectorized-env package.

Kept import-light on purpose: this ``__init__`` pulls in **nothing** heavy, so a
forkserver/spawn worker importing ``env.subproc_libero_env`` does not drag in
mujoco (in-process env) or the JAX training stack. Import the concrete classes
from their submodules:

    from env.subproc_libero_env import SubprocVectorizedLiberoEnv
    from env.vectorized_libero_env import VectorizedLiberoEnv
"""
