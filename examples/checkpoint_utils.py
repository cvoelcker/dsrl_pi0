"""Checkpointing for train_sim.

Layout under ``ckpt_dir``:
    agent/           flax checkpoint dir for PixelSACLearner._save_dict
    replay_buffer.pkl
    state.json       {step, total_env_steps, traj_idx, wandb_run_id, wandb_group}
"""
import json
import os

from flax.training import checkpoints


AGENT_SUBDIR = 'agent'
REPLAY_FILE = 'replay_buffer.pkl'
STATE_FILE = 'state.json'


def _paths(ckpt_dir):
    return (
        os.path.join(ckpt_dir, AGENT_SUBDIR),
        os.path.join(ckpt_dir, REPLAY_FILE),
        os.path.join(ckpt_dir, STATE_FILE),
    )


def checkpoint_exists(ckpt_dir):
    _, _, state_path = _paths(ckpt_dir)
    return os.path.isfile(state_path)


def save_run_checkpoint(ckpt_dir, agent, replay_buffer, step,
                        total_env_steps, traj_idx, wandb_run_id, wandb_group):
    agent_dir, replay_path, state_path = _paths(ckpt_dir)
    os.makedirs(agent_dir, exist_ok=True)
    # keep=1: overwrite the previous checkpoint each interval. Bump if you want
    # a longer history.
    checkpoints.save_checkpoint(
        agent_dir, agent._save_dict, step,
        prefix='checkpoint_', overwrite=True, keep=3,
    )
    replay_buffer.save(replay_path)
    state = dict(
        step=int(step),
        total_env_steps=int(total_env_steps),
        traj_idx=int(traj_idx),
        wandb_run_id=wandb_run_id,
        wandb_group=wandb_group,
    )
    tmp = state_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f)
    os.replace(tmp, state_path)
    print(f'[checkpoint] saved step {step} to {ckpt_dir}')


def load_run_state(ckpt_dir):
    """Return the state.json contents, or None if there is no checkpoint."""
    if not checkpoint_exists(ckpt_dir):
        return None
    _, _, state_path = _paths(ckpt_dir)
    with open(state_path, 'r') as f:
        return json.load(f)


def load_agent_and_buffer(ckpt_dir, agent, replay_buffer):
    agent_dir, replay_path, _ = _paths(ckpt_dir)
    agent.restore_checkpoint(agent_dir)
    replay_buffer.restore(replay_path)
    print(f'[checkpoint] restored agent + replay buffer from {ckpt_dir}')
