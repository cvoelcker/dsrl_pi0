#! /usr/bin/env python
import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs from https://github.com/huggingface/gym-aloha/tree/main?tab=readme-ov-file#-gpu-rendering-egl
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import copy

import jax
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
import gym_aloha
from gym.spaces import Dict, Box

from libero.libero import benchmark

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim import trajwise_alternating_training_loop
from examples.env_adapter import make_libero_vec_env
from examples.checkpoint_utils import (
    checkpoint_exists,
    load_agent_and_buffer,
    load_run_state,
)
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))

def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            if variant.env == 'libero':
                state_dim = 8
            elif variant.env == 'aloha_cube':
                state_dim = 14
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        if variant.get('use_language', False) and variant.get('language_dim', 0) > 0:
            obs_dict['language_emb'] = Box(low=-np.inf, high=np.inf,
                                           shape=(variant.language_dim, 1), dtype=np.float32)
        if variant.get('use_policy_reps', False) and variant.get('policy_rep_dim', 0) > 0:
            obs_dict['policy_reps'] = Box(low=-np.inf, high=np.inf,
                                          shape=(variant.policy_rep_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32) # 32 is the noise action space of pi 0


def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
        
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)

    # If run_id is set, use $EXP/<run_id> as both the output/checkpoint dir
    # and the wandb id — that way resuming picks up the same wandb run.
    run_id = variant.get('run_id', '') or ''
    if run_id:
        outputdir = os.path.join(os.environ['EXP'], run_id)
        resume_run = os.path.isdir(outputdir) and checkpoint_exists(outputdir)
        expname = run_id
    else:
        outputdir = os.path.join(os.environ['EXP'], expname)
        resume_run = False
    variant.outputdir = outputdir
    variant.ckpt_dir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir, '(resume=%s)' % resume_run)
    
    tasks = None
    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[variant.task_suite]()

        # Which tasks to train on: an explicit comma-separated --task_ids list,
        # else every task in the (custom) suite. This is the multitask setup
        # translated from steering-with-failures' EnvConfig.task_ids.
        num_suite_tasks = task_suite.get_num_tasks()
        if str(variant.get('task_ids', '')).strip():
            task_ids = [int(x) for x in str(variant.task_ids).split(',') if x.strip() != '']
        else:
            task_ids = list(range(num_suite_tasks))
        print(f'Libero suite {variant.task_suite}: {num_suite_tasks} tasks, training on {task_ids}')

        # Task metadata only; the actual envs live inside the subprocess-vectorized
        # worker pool built below. We no longer construct per-task in-process envs.
        tasks = []
        for tid in task_ids:
            task = task_suite.get_task(tid)
            tasks.append({'task_id': tid, 'description': task.language})
            print(f'  task {tid}: {task.language}')

        # Per-task language embeddings for critic/actor conditioning.
        variant.use_language = bool(variant.get('use_language', 0))
        variant.use_muse = bool(variant.get('use_muse', 0))
        if variant.use_language:
            if variant.use_muse:
                from jaxrl2.networks.language import MUSEEncoder
                enc = MUSEEncoder()
                variant.language_dim = enc.embed_dim  # MUSE is 512-d
                unique = sorted({t['description'] for t in tasks})
                desc_to_emb = dict(zip(unique, enc.encode(unique)))
                for t in tasks:
                    t['language_emb'] = np.asarray(desc_to_emb[t['description']], dtype=np.float32)
                print(f'Computed MUSE embeddings for {len(unique)} unique task description(s)')
            else:
                # One-hot task ids; language_dim must cover the number of tasks.
                variant.language_dim = max(int(variant.get('language_dim', 0)), len(tasks))
                eye = np.eye(variant.language_dim, dtype=np.float32)
                for k, t in enumerate(tasks):
                    t['language_emb'] = eye[k]
        else:
            variant.language_dim = 0

        # Subprocess-vectorized Libero envs replace the per-task in-process envs;
        # `env`/`eval_env` are the same vec-env instance (rollout + eval share workers).
        vec_env = make_libero_vec_env(variant)
        env = vec_env
        eval_env = vec_env
        variant.task_description = tasks[0]['description']
        variant.env_max_reward = 1
        variant.max_timesteps = 400
    elif variant.env == 'aloha_cube':
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        eval_env = copy.deepcopy(env)
        variant.env_max_reward = 4
        variant.max_timesteps = 400
        variant.use_language = False
        variant.language_dim = 0
        

    # Preserve wandb group across resume by pulling it from the saved state.
    saved_state = load_run_state(outputdir) if resume_run else None
    if saved_state is not None and saved_state.get('wandb_group'):
        group_name = saved_state['wandb_group']
    else:
        group_name = variant.prefix + '_' + variant.launch_group_id
    variant.wandb_group = group_name
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(
        variant.prefix != '', variant, variant.wandb_project,
        experiment_id=expname, output_dir=wandb_output_dir,
        group_name=group_name, resume=resume_run,
    )

    # Load the base pi0 policy first so we can size the policy-feature obs key.
    if variant.env == 'libero':
        config = openpi_config.get_config("pi0_libero")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    elif variant.env == 'aloha_cube':
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    else:
        raise NotImplementedError()
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Loaded pi0 policy from %s", checkpoint_dir)

    # Optionally condition the critic/actor on the base pi0 model's features
    # (psi = state-only prefix representation), mirroring steering's use_policy_reps.
    variant.use_policy_reps = bool(variant.get('use_policy_reps', 0))
    variant.critic_rep = variant.get('critic_rep', 'psi')
    if variant.use_policy_reps:
        if variant.critic_rep != 'psi':
            raise NotImplementedError(
                f"critic_rep={variant.critic_rep!r} not supported here; only 'psi' "
                "(state-only) fits the observation-key conditioning. 'phi' depends on "
                "the sampled action/timestep and would require threading actions into "
                "the base model call.")
        variant.policy_rep_dim = int(getattr(agent_dp._model, f"{variant.critic_rep}_dim"))
        print(f"Conditioning on base pi0 features: critic_rep={variant.critic_rep}, dim={variant.policy_rep_dim}")
    else:
        variant.policy_rep_dim = 0

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)

    agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)

    # Hard cap at 1M transitions — the buffer is now a fixed FIFO ring (see
    # ReplayBuffer.insert), so pick the smaller of the naive "one insert per
    # multi_grad_step" estimate and the cap. This bounds host memory to
    # ~1M × per-transition bytes regardless of max_steps or num_envs.
    online_buffer_size = min(variant.max_steps // variant.multi_grad_step, 1_000_000)
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)

    start_step = 0
    start_total_env_steps = 0
    start_traj_idx = 0
    if resume_run:
        load_agent_and_buffer(outputdir, agent, online_replay_buffer)
        start_step = int(saved_state.get('step', 0))
        start_total_env_steps = int(saved_state.get('total_env_steps', 0))
        start_traj_idx = int(saved_state.get('traj_idx', 0))
        print(f'[checkpoint] resuming at step={start_step}, '
              f'env_steps={start_total_env_steps}, traj_idx={start_traj_idx}')

    trajwise_alternating_training_loop(
        variant, agent, env, eval_env, online_replay_buffer, replay_buffer,
        wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp, tasks=tasks,
        start_step=start_step, start_total_env_steps=start_total_env_steps,
        start_traj_idx=start_traj_idx,
    )
 