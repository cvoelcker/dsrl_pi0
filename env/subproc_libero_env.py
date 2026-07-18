"""Subprocess-vectorized LIBERO envs + shared, parallel-aware seeding.

Each env runs in its own process with its own GL/EGL context, eliminating the
process-global GL current-context framebuffer bleed that silently corrupts
observations when multiple tasks render in one process (the failure that makes
per-task success degrade as more tasks share a run). All RNG lives in the main
process; workers are purely mechanical (build / reset / step / render).

Kept intentionally lightweight at import time (no jax / openpi / mujoco at module
top level) so freshly-started forkserver/spawn workers do not drag in the whole
training stack. ``libero`` is imported lazily: task metadata in the main process,
the simulator inside each worker (after MUJOCO_GL is set). The seeding helpers
here are the single source of truth, shared with the in-process env class.
"""

import multiprocessing as mp
import os
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

_OBS_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def filter_obs(obs):
    return {k: obs[k] for k in _OBS_KEYS if k in obs}


def build_slot_rngs(seed, num_envs):
    """One independent RNG substream per env slot.

    ``SeedSequence.spawn`` produces decorrelated children, so parallel slots
    never collapse to the same draw while the whole set stays reproducible from
    ``seed``.
    """
    children = np.random.SeedSequence(seed).spawn(num_envs)
    return [np.random.default_rng(c) for c in children]


def resolve_reset_mode(env_cfg):
    """``env.reset_mode`` wins; else map legacy ``random_reset``.

    True (og) -> "random"; False (curated init states) -> "curated_uniform".
    """
    mode = getattr(env_cfg, "reset_mode", None)
    if mode:
        return mode
    return "random" if getattr(env_cfg, "random_reset", True) else "curated_uniform"


def reset_recipe(reset_mode, slot_rng, num_init_states, pin_reset, base_seed, idx):
    """Resolve one reset into a mechanical recipe a worker can execute.

    Returns dict with env_seed (int|None), init_state_idx (int|None), warmup (bool).

    - pin_reset (diagnostic): fixed init_states[0] + slot-derived seed, identical
      regardless of task count — used to A/B whether init randomization moved
      success.
    - curated_uniform: draw an init state uniformly from the task's ~50 official
      LIBERO init states; set_init_state fully specifies the layout, so no
      placement seed is needed. Settle with a warmup.
    - random (og): fresh per-episode object-placement seed, plain reset(), no
      curated state, no warmup.

    The draw comes from the slot's substream, which advances every episode and is
    never re-created on task rebind — the fix for the diversity-collapse bug where
    rebinding replayed the same layout.
    """
    if pin_reset:
        return {"env_seed": base_seed + 1000 * idx, "init_state_idx": 0, "warmup": True}
    if reset_mode == "curated_uniform":
        return {
            "env_seed": None,
            "init_state_idx": int(slot_rng.integers(num_init_states)),
            "warmup": True,
        }
    if reset_mode == "random":
        return {
            "env_seed": int(slot_rng.integers(1 << 31)),
            "init_state_idx": None,
            "warmup": False,
        }
    raise ValueError(f"unknown reset_mode {reset_mode!r}")


def load_task_metadata(env_cfg):
    """Resolve task ids + per-task metadata (description, bddl_file, init_states).

    Runs in the main process only; imports libero lazily. Precedence:
    task_ids > task_id > whole suite.
    """
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[env_cfg.task_suite]()
    if env_cfg.task_ids:
        task_ids = list(env_cfg.task_ids)
    elif env_cfg.task_id is not None:
        task_ids = [env_cfg.task_id]
    else:
        task_ids = list(range(suite.n_tasks))

    bddl_root = pathlib.Path(get_libero_path("bddl_files"))
    tasks = {}
    for tid in task_ids:
        task = suite.get_task(tid)
        tasks[tid] = {
            "description": task.language,
            "bddl_file": str(bddl_root / task.problem_folder / task.bddl_file),
            "init_states": suite.get_task_init_states(tid),
        }
    return task_ids, tasks

def _worker_main(conn, resolution, num_steps_wait, gl_backend, egl_device_id):
    """One env slot in its own process, driven by commands over ``conn``.

    Sets its own GL backend/device BEFORE importing robosuite/mujoco so every
    worker owns an isolated offscreen context (no shared current-context bleed).
    """
    os.environ["MUJOCO_GL"] = gl_backend
    if egl_device_id is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_id)
        os.environ.setdefault("EGL_DEVICE_ID", str(egl_device_id))

    from libero.libero.envs import OffScreenRenderEnv

    env = None
    cur_task = None
    init_states = None

    def _warmup(obs):
        for _ in range(num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        return obs

    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                break
            elif cmd == "bind":
                tid = payload["task_id"]
                if cur_task == tid and env is not None:
                    conn.send(("ok", None))
                    continue
                if env is not None:
                    env.close()
                    env = None
                    cur_task = None
                try:
                    env = OffScreenRenderEnv(
                        bddl_file_name=payload["bddl_file"],
                        camera_heights=resolution,
                        camera_widths=resolution,
                    )
                except BaseException:
                    # Render-context creation can fail (e.g. GL framebuffer
                    # incomplete, error 0x8cdd, when the GPU is out of headroom).
                    # Report it up the pipe instead of dying silently and
                    # leaving the parent with an opaque EOFError.
                    import traceback
                    conn.send(("err", traceback.format_exc()))
                    continue
                cur_task = tid
                init_states = payload["init_states"]
                conn.send(("ok", None))
            elif cmd == "reset":
                if payload["env_seed"] is not None:
                    env.seed(payload["env_seed"])
                obs = env.reset()
                if payload["init_state_idx"] is not None:
                    obs = env.set_init_state(init_states[payload["init_state_idx"]])
                if payload["warmup"]:
                    obs = _warmup(obs)
                conn.send(("obs", filter_obs(obs)))
            elif cmd == "step":
                obs, reward, done, _ = env.step(payload)
                conn.send(("step", (filter_obs(obs), float(reward), bool(done), {})))
            else:
                conn.send(("err", f"unknown cmd {cmd!r}"))
    finally:
        if env is not None:
            env.close()
        conn.close()

class SubprocVectorizedLiberoEnv:
    
    def __init__(self, env_cfg, base_seed: int = 0, num_envs: int = None):
        self.num_envs = num_envs if num_envs is not None else env_cfg.num_envs
        self.num_steps_wait = env_cfg.num_steps_wait
        self.resolution = env_cfg.resolution
        self.reset_mode = resolve_reset_mode(env_cfg)
        self.pin_reset = getattr(env_cfg, "pin_reset", False)
        self._base_seed = base_seed

        self.task_ids, self.tasks = load_task_metadata(env_cfg)
        self.env_task_ids = [None] * self.num_envs

        self._task_rng = np.random.default_rng(base_seed)
        self._task_queue = []
        self.slot_rngs = build_slot_rngs(base_seed, self.num_envs)

        gl_backend = getattr(env_cfg, "mujoco_gl", None) or os.environ.get("MUJOCO_GL", "egl")
        # Per-worker EGL device only if explicitly requested (multi-GPU); else let
        # mujoco pick.
        egl_devices = getattr(env_cfg, "egl_device_ids", None)

        try:
            self._ctx = mp.get_context("forkserver")
        except ValueError:
            self._ctx = mp.get_context("spawn")

        self._conns = []
        self._procs = []
        for i in range(self.num_envs):
            parent_conn, child_conn = self._ctx.Pipe()
            dev = egl_devices[i % len(egl_devices)] if egl_devices else None
            proc = self._ctx.Process(
                target=_worker_main,
                args=(child_conn, self.resolution, self.num_steps_wait, gl_backend, dev),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent keeps only its end
            self._conns.append(parent_conn)
            self._procs.append(proc)

    @property
    def task_descriptions(self):
        return [self.tasks[tid]["description"] for tid in self.env_task_ids]

    def _next_tasks(self, n):
        out = []
        while len(out) < n:
            if not self._task_queue:
                self._task_queue = self._task_rng.permutation(self.task_ids).tolist()
            out.append(self._task_queue.pop())
        return out

    def reseed_perturbations(self, seed=None):
        """Rebuild per-slot substreams so a following block of resets is
        reproducible (eval draws the same init sequence every evaluation)."""
        self.slot_rngs = build_slot_rngs(
            self._base_seed if seed is None else seed, self.num_envs
        )

    def _bind(self, idx, task_id):
        """Bind slot idx to task_id (rebuild worker env if changed)."""
        if self.env_task_ids[idx] == task_id:
            return
        self._conns[idx].send(("bind", {
            "task_id": task_id,
            "bddl_file": self.tasks[task_id]["bddl_file"],
            "init_states": self.tasks[task_id]["init_states"],
        }))
        self._await_bind_ack(idx, task_id)
        self.env_task_ids[idx] = task_id

    def _await_bind_ack(self, idx, task_id):
        """Block on a slot's bind ack, raising the worker's traceback on error.

        Turns a worker-side render-context failure into a legible exception in
        the main process instead of an opaque EOFError from a dead pipe.
        """
        try:
            tag, info = self._conns[idx].recv()
        except EOFError:
            raise RuntimeError(
                f"env worker slot {idx} died while binding task {task_id} "
                "(pipe closed before ack)"
            ) from None
        if tag != "ok":
            raise RuntimeError(
                f"env worker slot {idx} failed to bind task {task_id}:\n{info}"
            )

    def _recipe(self, idx):
        tid = self.env_task_ids[idx]
        return reset_recipe(
            self.reset_mode, self.slot_rngs[idx],
            len(self.tasks[tid]["init_states"]),
            self.pin_reset, self._base_seed, idx,
        )

    def reset(self, idx, task_id=None):
        if task_id is not None:
            self._bind(idx, task_id)
        self._conns[idx].send(("reset", self._recipe(idx)))
        tag, obs = self._conns[idx].recv()
        assert tag == "obs"
        return obs

    def reset_all(self, task_ids=None):
        """Reset every slot (fresh shuffled task assignment if task_ids is None).

        Binds all slots first, then fires all resets and gathers — env compute
        runs concurrently across worker processes.
        """
        if task_ids is None:
            task_ids = self._next_tasks(self.num_envs)
        # Bind phase (parallel): send all rebinds, then collect acks.
        pending = []
        for i, tid in enumerate(task_ids):
            if self.env_task_ids[i] != tid:
                self._conns[i].send(("bind", {
                    "task_id": tid,
                    "bddl_file": self.tasks[tid]["bddl_file"],
                    "init_states": self.tasks[tid]["init_states"],
                }))
                self.env_task_ids[i] = tid
                pending.append((i, tid))
        for i, tid in pending:
            self._await_bind_ack(i, tid)
        # Reset phase (parallel).
        for i in range(self.num_envs):
            self._conns[i].send(("reset", self._recipe(i)))
        out = []
        for i in range(self.num_envs):
            tag, obs = self._conns[i].recv()
            assert tag == "obs"
            out.append(obs)
        return out

    def step(self, idx, action):
        self._conns[idx].send(("step", np.asarray(action)))
        tag, result = self._conns[idx].recv()
        assert tag == "step"
        return result

    def step_all(self, actions):
        """Batched step. ``actions``: dict {slot_idx: action}. Returns
        {slot_idx: (obs, reward, done, info)}, running workers concurrently."""
        for i, a in actions.items():
            self._conns[i].send(("step", np.asarray(a)))
        results = {}
        for i in actions:
            tag, result = self._conns[i].recv()
            assert tag == "step"
            results[i] = result
        return results

    def close(self):
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        for conn in self._conns:
            conn.close()
