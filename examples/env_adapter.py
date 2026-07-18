"""Bridge from dsrl_pi0's flat `variant` argparse namespace to the env_cfg
shape that ``env.subproc_libero_env.SubprocVectorizedLiberoEnv`` expects.

Kept in ``examples/`` (not in the ``env`` package) so the env package stays
import-light and independent of dsrl_pi0's config conventions.
"""

from types import SimpleNamespace

from env.subproc_libero_env import SubprocVectorizedLiberoEnv


def _variant_get(variant, key, default=None):
    """Read from either an argparse.Namespace or a dict-like variant."""
    if hasattr(variant, "get") and not isinstance(variant, SimpleNamespace):
        return variant.get(key, default)
    return getattr(variant, key, default)


def build_env_cfg(variant):
    """Project the flat dsrl_pi0 variant onto the attributes the vec-env reads."""
    task_ids_raw = str(_variant_get(variant, "task_ids", "") or "").strip()
    task_ids = [int(x) for x in task_ids_raw.split(",") if x.strip()] if task_ids_raw else []

    egl_raw = str(_variant_get(variant, "egl_device_ids", "") or "").strip()
    egl_device_ids = [int(x) for x in egl_raw.split(",") if x.strip()] if egl_raw else None

    reset_mode = _variant_get(variant, "reset_mode", None)
    if isinstance(reset_mode, str) and not reset_mode.strip():
        reset_mode = None

    mujoco_gl = _variant_get(variant, "mujoco_gl", None)
    if isinstance(mujoco_gl, str) and not mujoco_gl.strip():
        mujoco_gl = None

    return SimpleNamespace(
        num_envs=int(_variant_get(variant, "num_envs", 1)),
        num_steps_wait=int(_variant_get(variant, "num_steps_wait", 10)),
        resolution=int(_variant_get(variant, "env_resolution", 256)),
        task_suite=_variant_get(variant, "task_suite"),
        task_ids=task_ids,
        task_id=_variant_get(variant, "task_id", None),
        reset_mode=reset_mode,
        random_reset=bool(_variant_get(variant, "random_reset", True)),
        pin_reset=bool(_variant_get(variant, "pin_reset", False)),
        mujoco_gl=mujoco_gl,
        egl_device_ids=egl_device_ids,
    )


def make_libero_vec_env(variant, base_seed=None):
    """Build a SubprocVectorizedLiberoEnv from the dsrl_pi0 variant."""
    env_cfg = build_env_cfg(variant)
    seed = int(_variant_get(variant, "seed", 0)) if base_seed is None else int(base_seed)
    return SubprocVectorizedLiberoEnv(env_cfg, base_seed=seed)
