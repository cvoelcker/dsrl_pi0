import os
# Stop JAX from preallocating ~90% of VRAM
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import sys
from examples.train_sim import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=50,help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='libero', help='name of environment')
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='checkpoint interval.', type=int)
    parser.add_argument('--run_id', default='', help='Unique run id. Checkpoints live under $EXP/<run_id>; if that folder exists, the run resumes (agent + replay buffer + wandb run) instead of starting fresh. Empty => no checkpointing.', type=str)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--wandb_project', default='cql_sim_online', help='wandb project')
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)
    # --- Multitask Libero + language conditioning ---
    parser.add_argument('--task_suite', default='libero_paper', help='Libero benchmark suite to train on (multitask).')
    parser.add_argument('--task_ids', default='', help='Comma-separated Libero task ids to use. Empty = all tasks in the suite.')
    parser.add_argument('--use_language', default=1, help='Condition the critic/actor on a per-task language embedding.', type=int)
    parser.add_argument('--use_muse', default=1, help='Use 512-d MUSE embeddings of the task description (else one-hot task ids).', type=int)
    parser.add_argument('--language_dim', default=512, help='Language embedding dim (512 for MUSE; must be >= num tasks for one-hot).', type=int)
    parser.add_argument('--use_policy_reps', default=1, help='Condition the critic/actor on the base pi0 model features (policy representation).', type=int)
    parser.add_argument('--critic_rep', default='psi', help="Which base pi0 representation to read: 'psi' (state-only prefix mean).", type=str)
    # --- Subprocess-vectorized Libero env (env.subproc_libero_env) ---
    parser.add_argument('--num_envs', default=16, help='Number of parallel Libero env workers (SubprocVectorizedLiberoEnv).', type=int)
    parser.add_argument('--num_steps_wait', default=10, help='Warmup dummy steps after reset (settle sim before recording obs).', type=int)
    parser.add_argument('--env_resolution', default=256, help='Camera height/width for Libero rendering.', type=int)
    parser.add_argument('--reset_mode', default='', help="'random' (per-episode object placement seed) or 'curated_uniform' (draw from official init states). Empty => derived from --random_reset.", type=str)
    parser.add_argument('--random_reset', default=1, help='Legacy: 1 => reset_mode=random, 0 => curated_uniform. Ignored if --reset_mode is set.', type=int)
    parser.add_argument('--pin_reset', default=0, help='Diagnostic: pin every slot to init_states[0] with a slot-derived seed.', type=int)
    parser.add_argument('--mujoco_gl', default='', help='Force MUJOCO_GL backend inside each worker (e.g. egl). Empty => inherit environment / default egl.', type=str)
    parser.add_argument('--egl_device_ids', default='', help='Comma-separated EGL device ids to round-robin across workers (multi-GPU). Empty => let mujoco pick.', type=str)

    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr= 3e-4,
        temp_lr=3e-4,
        hidden_dims= (128, 128, 128),
        cnn_features= (32, 32, 32, 32),
        cnn_strides= (2, 1, 1, 1),
        cnn_padding= 'VALID',
        latent_dim= 50,
        discount= 0.999,
        tau= 0.005,
        critic_reduction = 'mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy='auto',
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
    
