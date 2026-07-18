from tqdm import tqdm
import numpy as np
import wandb
import jax
import jax.numpy as jnp
from openpi_client import image_tools
import math
import PIL

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def obs_to_img(obs, variant):
    '''
    Convert raw observation to resized image for DSRL actor/critic
    '''
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == 'aloha_cube':
        curr_image = obs["pixels"]["top"]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0: 
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image

def build_obs_dict(curr_image, qpos, variant, language_emb=None, policy_reps=None):
    '''
    Assemble the DSRL actor/critic observation dict, optionally with the
    per-task language embedding and the base pi0 policy features (matches the
    DummyEnv observation space).
    '''
    obs_dict = {'pixels': curr_image[np.newaxis, ..., np.newaxis]}
    if variant.add_states:
        obs_dict['state'] = qpos[np.newaxis, ..., np.newaxis]
    if variant.get('use_language', False) and language_emb is not None:
        obs_dict['language_emb'] = np.asarray(
            language_emb, dtype=np.float32)[np.newaxis, ..., np.newaxis]
    if variant.get('use_policy_reps', False) and policy_reps is not None:
        obs_dict['policy_reps'] = np.asarray(
            policy_reps, dtype=np.float32)[np.newaxis, ..., np.newaxis]
    return obs_dict


# Cache one jitted psi-extractor per loaded pi0 policy.
_PSI_FN_CACHE = {}


def _psi_fn(agent_dp):
    key = id(agent_dp)
    if key not in _PSI_FN_CACHE:
        from openpi.shared import nnx_utils
        _PSI_FN_CACHE[key] = nnx_utils.module_jit(agent_dp._model.get_state_representations)
    return _PSI_FN_CACHE[key]


def _batched_pi0_observation(agent_dp, obs_pi_list):
    """Stack ``len(obs_pi_list)`` raw pi0 inputs into one batched Observation.

    Returns ``(observation, stacked_state_np)``. Input transforms are run per
    slot (they may not be batch-safe) before stacking.
    """
    from openpi.models import model as _model
    per_slot_inputs = [agent_dp._input_transform(jax.tree.map(lambda x: x, obs))
                       for obs in obs_pi_list]
    stacked = jax.tree.map(
        lambda *xs: jnp.stack([jnp.asarray(x) for x in xs], axis=0),
        *per_slot_inputs,
    )
    return _model.Observation.from_dict(stacked), np.asarray(stacked["state"])


def batched_policy_reps(agent_dp, obs_pi_list):
    """Batched psi (state-only prefix mean) across slots. Returns list of
    (psi_dim,) float32 arrays, one per slot."""
    observation, _ = _batched_pi0_observation(agent_dp, obs_pi_list)
    psi_out = _psi_fn(agent_dp)(observation)
    psi_batched = np.asarray(psi_out[0] if isinstance(psi_out, tuple) else psi_out)
    return [np.asarray(psi_batched[b], dtype=np.float32)
            for b in range(psi_batched.shape[0])]


def batched_pi0_sample(agent_dp, obs_pi_list, noise_batch, rng_key):
    """Batched pi0 diffusion. ``noise_batch``: (B, 50, action_dim). Returns
    a list of B numpy action arrays with `_output_transform` applied per slot.

    Batch dim is always ``len(obs_pi_list)`` — pass a padded list so the jitted
    ``sample_actions`` doesn't recompile when slots finish mid-episode.
    """
    observation, stacked_state = _batched_pi0_observation(agent_dp, obs_pi_list)
    actions_batched = np.asarray(
        agent_dp._sample_actions(rng_key, observation, noise=jnp.asarray(noise_batch))
    )  # (B, H, A)
    actions_out = []
    for b in range(actions_batched.shape[0]):
        out = agent_dp._output_transform({
            "state": stacked_state[b],
            "actions": actions_batched[b],
        })
        actions_out.append(out["actions"])
    return actions_out


def get_policy_reps(agent_dp, obs_pi_zero):
    '''
    Read the base pi0 model's psi representation (state-only prefix mean) for a
    single observation, mirroring steering's Pi05PsiExtractor. `obs_pi_zero` is
    the same raw dict passed to `agent_dp.infer`; we run it through the policy's
    input transform and build an Observation exactly as `Policy.infer` does.
    '''
    from openpi.models import model as _model
    inputs = agent_dp._input_transform(jax.tree.map(lambda x: x, obs_pi_zero))
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(inputs)
    psi_rep = _psi_fn(agent_dp)(observation)[0]  # (1, psi_dim)
    return np.asarray(psi_rep[0], dtype=np.float32)  # (psi_dim,)


def obs_to_pi_zero_input(obs, variant, task_description=None):
    if task_description is None:
        task_description = variant.task_description
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        
        obs_pi_zero = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                    }
    elif variant.env == 'aloha_cube':
        img = np.ascontiguousarray(obs["pixels"]["top"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        obs_pi_zero = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2,0,1))}
        }
    else:
        raise NotImplementedError()
    return obs_pi_zero

def obs_to_qpos(obs, variant):
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env == 'aloha_cube':
        qpos = obs["agent_pos"]
    else:
        raise NotImplementedError()
    return qpos

def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None, tasks=None,
                                       start_step=0, start_total_env_steps=0, start_traj_idx=0):
    from examples.checkpoint_utils import save_run_checkpoint
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    # Libero uses the subprocess-vectorized env (env == vec_env, no per-task envs);
    # other envs (aloha) keep the single-env task registry.
    use_vec = 'libero' in variant.env
    if tasks is None:
        tasks = [{'env': env, 'description': variant.get('task_description'), 'language_emb': None}]
    tasks_by_id = {t['task_id']: t for t in tasks if 'task_id' in t} if use_vec else None

    total_env_steps = start_total_env_steps
    traj_idx = start_traj_idx
    i = start_step
    if i == 0:
        wandb_logger.log({'num_online_samples': 0}, step=i)
        wandb_logger.log({'num_online_trajs': 0}, step=i)
        wandb_logger.log({'env_steps': 0}, step=i)

    with tqdm(total=variant.max_steps, initial=i) as pbar:
        while i <= variant.max_steps:
            if use_vec:
                # One batched rollout across `num_envs` slots; task rotation is
                # handled inside the vec env's _next_tasks.
                trajs = collect_traj_vec(variant, agent, env, tasks_by_id, i, agent_dp)
            else:
                # Round-robin over tasks so the multitask critic sees every task evenly.
                task = tasks[traj_idx % len(tasks)]
                traj_idx += 1
                trajs = [collect_traj(variant, agent, task, i, agent_dp)]

            add_online_data_to_buffer(variant, trajs, online_replay_buffer)
            batch_env_steps = sum(tr['env_steps'] for tr in trajs)
            total_env_steps += batch_env_steps
            batch_query_steps = sum(len(tr['rewards']) for tr in trajs)
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', online_replay_buffer._traj_counter)
            print('total env steps:', total_env_steps)

            # Aggregate logging values across the batch (means for vec, identity for single).
            batch_return = float(np.mean([tr['episode_return'] for tr in trajs]))
            batch_success = float(np.mean([float(tr['is_success']) for tr in trajs]))

            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = batch_query_steps * variant.multi_grad_step

            if len(online_replay_buffer) > variant.start_online_updates:
                for _ in range(num_gradsteps):
                    # perform first visualization before updating
                    if i == 0:
                        print('performing evaluation for initial checkpoint')
                        if perform_control_evals:
                            perform_control_eval(agent, tasks, i, variant, wandb_logger, agent_dp, eval_env=eval_env)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # online perform update once we have some amount of online trajs
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1


                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'episode_return (exploration)': batch_return,
                            'is_success (exploration)': batch_success,
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': online_replay_buffer._traj_counter}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval(agent, tasks, i, variant, wandb_logger, agent_dp, eval_env=eval_env)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        save_run_checkpoint(
                            variant.ckpt_dir, agent, online_replay_buffer,
                            step=i, total_env_steps=total_env_steps,
                            traj_idx=traj_idx,
                            wandb_run_id=variant.get('run_id', '') or None,
                            wandb_group=variant.get('wandb_group', ''),
                        )

            
def add_online_data_to_buffer(variant, traj_or_trajs, online_replay_buffer):
    """Insert one trajectory or a list of trajectories (from the vec-env) into
    the buffer, incrementing the traj counter once per trajectory."""
    trajs = traj_or_trajs if isinstance(traj_or_trajs, list) else [traj_or_trajs]
    for traj in trajs:
        _insert_single_traj(variant, traj, online_replay_buffer)


def _insert_single_traj(variant, traj, online_replay_buffer):
    discount_horizon = variant.query_freq
    actions = np.array(traj['actions']) # (T, chunk_size, action_dim )
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)

        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()

def collect_traj(variant, agent, task, i, agent_dp=None):
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    env = task['env']
    task_description = task.get('description')
    language_emb = task.get('language_emb')

    agent._rng, rng = jax.random.split(agent._rng)

    if 'libero' in variant.env:
        obs = env.reset()
    elif 'aloha' in variant.env:
        obs, _ = env.reset()

    image_list = [] # for visualization
    rewards = []
    action_list = []
    obs_list = []

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)

        if t % query_frequency == 0:

            assert agent_dp is not None
            # we then use the noise to sample the action from diffusion model
            rng, key = jax.random.split(rng)
            qpos = obs_to_qpos(obs, variant)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant, task_description)
            policy_reps = get_policy_reps(agent_dp, obs_pi_zero) if variant.get('use_policy_reps', False) else None
            obs_dict = build_obs_dict(curr_image, qpos, variant, language_emb, policy_reps)
            if i == 0:
                # for initial round of data collection, we sample from standard gaussian noise
                noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                noise_repeat = jax.numpy.repeat(noise[:, -1:, :], 50 - noise.shape[1], axis=1)
                noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                actions_noise = noise[0, :agent.action_chunk_shape[0], :]
            else:
                # sac agent predicts the noise for diffusion model
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]
            
            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
            action_list.append(actions_noise)
            obs_list.append(obs_dict)
     
        action_t = actions[t % query_frequency]
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated
            
        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

    # add last observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    if variant.get('use_policy_reps', False):
        obs_pi_zero = obs_to_pi_zero_input(obs, variant, task_description)
        policy_reps = get_policy_reps(agent_dp, obs_pi_zero)
    else:
        policy_reps = None
    obs_dict = build_obs_dict(curr_image, qpos, variant, language_emb, policy_reps)
    obs_list.append(obs_dict)
    image_list.append(curr_image)
    
    # per episode
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards!=None])
    is_success = (reward == env_max_reward)
    print(f'Rollout Done: {episode_return=}, Success: {is_success}')
    
    
    '''
    We use sparse -1/0 reward to train the SAC agent.
    '''
    if is_success:
        query_steps = len(action_list)
        rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
        masks = np.concatenate([np.ones(query_steps - 1), [0]])
    else:
        query_steps = len(action_list)
        rewards = -np.ones(query_steps)
        masks = np.ones(query_steps)

    return {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1 
    }

def collect_traj_vec(variant, agent, vec_env, tasks_by_id, i, agent_dp, task_ids=None, rng=None):
    """Batched exploration rollout across `vec_env.num_envs` slots.

    Returns a list of per-slot trajectory dicts with the same schema as
    `collect_traj` (adds a 'task_id' field). Slots whose episode ended before
    any pi0 query are dropped.
    """
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    N = vec_env.num_envs

    if rng is None:
        agent._rng, rng = jax.random.split(agent._rng)

    initial_obs = vec_env.reset_all(task_ids=task_ids)
    slot_task_ids = list(vec_env.env_task_ids)

    slot_obs_dicts = [[] for _ in range(N)]
    slot_actions = [[] for _ in range(N)]
    slot_rewards = [[] for _ in range(N)]
    slot_images = [[] for _ in range(N)]
    slot_done = [False] * N
    slot_last_reward = [0.0] * N
    slot_last_obs = list(initial_obs)
    slot_action_chunk = [None] * N

    want_policy_reps = bool(variant.get('use_policy_reps', False))
    # Padded to constant batch = N so pi0's jitted sample_actions doesn't
    # recompile when slots finish mid-episode; padded (done) slots' outputs
    # are discarded.
    action_dim = agent.action_chunk_shape[-1]

    for t in tqdm(range(max_timesteps)):
        if t % query_frequency == 0:
            # 1) Build pi0 inputs for every slot (padded slots use their stale
            #    last obs so the batch dim stays fixed at N).
            obs_pi_batch = [
                obs_to_pi_zero_input(slot_last_obs[s], variant,
                                     tasks_by_id[slot_task_ids[s]].get('description'))
                for s in range(N)
            ]

            # 2) Batched psi so the SAC actor sees the real policy_reps below.
            psi_per_slot = (batched_policy_reps(agent_dp, obs_pi_batch)
                            if want_policy_reps else [None] * N)

            # 3) Per-slot SAC actor → noise (cheap; keeps sac_obs shape stable
            #    while pi0 does the heavy diffusion).
            per_slot_meta = [None] * N  # (obs_dict, actions_noise)
            noise_batch_np = np.zeros((N, 50, action_dim), dtype=np.float32)
            for slot in range(N):
                if slot_done[slot]:
                    continue
                obs = slot_last_obs[slot]
                curr_image = obs_to_img(obs, variant)
                qpos = obs_to_qpos(obs, variant)
                language_emb = tasks_by_id[slot_task_ids[slot]].get('language_emb')
                obs_dict = build_obs_dict(curr_image, qpos, variant, language_emb, psi_per_slot[slot])

                rng, key = jax.random.split(rng)
                if i == 0:
                    noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                    noise_repeat = jax.numpy.repeat(noise[:, -1:, :], 50 - noise.shape[1], axis=1)
                    noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)  # (1, 50, A)
                    actions_noise = np.asarray(noise[0, :agent.action_chunk_shape[0], :])
                    noise_np = np.asarray(noise[0])  # (50, A)
                else:
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    tail = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                    noise_np = np.concatenate([actions_noise, tail], axis=0)  # (50, A)
                noise_batch_np[slot] = noise_np
                per_slot_meta[slot] = (obs_dict, actions_noise)

            # 4) One batched pi0 diffusion across all N slots.
            rng, sample_key = jax.random.split(rng)
            actions_out = batched_pi0_sample(agent_dp, obs_pi_batch, noise_batch_np, sample_key)

            # 5) Commit to active slots; padded (done) slots discarded.
            for slot in range(N):
                if slot_done[slot]:
                    continue
                obs_dict, actions_noise = per_slot_meta[slot]
                slot_action_chunk[slot] = actions_out[slot]
                slot_actions[slot].append(actions_noise)
                slot_obs_dicts[slot].append(obs_dict)

        # Record images at each timestep for slots still running.
        for slot in range(N):
            if not slot_done[slot]:
                slot_images[slot].append(obs_to_img(slot_last_obs[slot], variant))

        actions_to_send = {}
        for slot in range(N):
            if slot_done[slot]:
                continue
            actions_to_send[slot] = slot_action_chunk[slot][t % query_frequency]

        if not actions_to_send:
            break

        results = vec_env.step_all(actions_to_send)
        for slot, (obs, reward, done, _) in results.items():
            slot_last_obs[slot] = obs
            slot_rewards[slot].append(reward)
            slot_last_reward[slot] = reward
            if done:
                slot_done[slot] = True

        if all(slot_done):
            break

    # Final observation per slot (mirrors single-env append after the loop).
    # Batched psi so the closing obs matches the query-step conditioning path.
    final_obs_pi_batch = [
        obs_to_pi_zero_input(slot_last_obs[s], variant,
                             tasks_by_id[slot_task_ids[s]].get('description'))
        for s in range(N)
    ] if want_policy_reps else None
    final_psi = (batched_policy_reps(agent_dp, final_obs_pi_batch)
                 if want_policy_reps else [None] * N)
    for slot in range(N):
        obs = slot_last_obs[slot]
        curr_image = obs_to_img(obs, variant)
        qpos = obs_to_qpos(obs, variant)
        language_emb = tasks_by_id[slot_task_ids[slot]].get('language_emb')
        obs_dict = build_obs_dict(curr_image, qpos, variant, language_emb, final_psi[slot])
        slot_obs_dicts[slot].append(obs_dict)
        slot_images[slot].append(curr_image)

    trajs = []
    for slot in range(N):
        query_steps = len(slot_actions[slot])
        if query_steps == 0:
            continue
        env_rewards = np.array(slot_rewards[slot], dtype=np.float32)
        episode_return = float(np.sum(env_rewards))
        is_success = bool(slot_last_reward[slot] == env_max_reward)
        if is_success:
            r = np.concatenate([-np.ones(query_steps - 1), [0]])
            m = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            r = -np.ones(query_steps)
            m = np.ones(query_steps)
        trajs.append({
            'observations': slot_obs_dicts[slot],
            'actions': slot_actions[slot],
            'rewards': r,
            'masks': m,
            'env_rewards': env_rewards,
            'is_success': is_success,
            'episode_return': episode_return,
            'images': slot_images[slot],
            'env_steps': len(env_rewards),
            'task_id': slot_task_ids[slot],
        })
    return trajs


def perform_control_eval(agent, tasks, i, variant, wandb_logger, agent_dp=None, eval_env=None):
    if 'libero' in variant.env and eval_env is not None:
        return _perform_control_eval_vec(agent, tasks, i, variant, wandb_logger, agent_dp, eval_env)

    query_frequency = variant.query_freq
    print('query frequency', query_frequency)
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    # Accept a single env for backward compatibility.
    if not isinstance(tasks, (list, tuple)):
        tasks = [{'env': tasks, 'description': variant.get('task_description'), 'language_emb': None}]

    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    rng = jax.random.PRNGKey(variant.seed+456)

    # variant.eval_episodes rollouts per task, so every task is evaluated evenly.
    for task in tasks:
        env = task['env']
        task_description = task.get('description')
        language_emb = task.get('language_emb')
        task_id = task.get('task_id')
        task_successes = []

        for rollout_id in range(variant.eval_episodes):
            if 'libero' in variant.env:
                obs = env.reset()
            elif 'aloha' in variant.env:
                obs, _ = env.reset()

            image_list = [] # for visualization
            rewards = []

            for t in tqdm(range(max_timesteps)):
                curr_image = obs_to_img(obs, variant)

                if t % query_frequency == 0:
                    qpos = obs_to_qpos(obs, variant)

                    rng, key = jax.random.split(rng)
                    assert agent_dp is not None

                    obs_pi_zero = obs_to_pi_zero_input(obs, variant, task_description)
                    policy_reps = get_policy_reps(agent_dp, obs_pi_zero) if variant.get('use_policy_reps', False) else None
                    obs_dict = build_obs_dict(curr_image, qpos, variant, language_emb, policy_reps)

                    if i == 0:
                        # for initial evaluation, we sample from standard gaussian noise to evaluate the base policy's performance
                        noise = jax.random.normal(rng, (1, 50, 32))
                    else:
                        actions_noise = agent.sample_actions(obs_dict)
                        actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                        noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                        noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]

                    actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]

                action_t = actions[t % query_frequency]

                if 'libero' in variant.env:
                    obs, reward, done, _ = env.step(action_t)
                elif 'aloha' in variant.env:
                    obs, reward, terminated, truncated, _ = env.step(action_t)
                    done = terminated or truncated

                rewards.append(reward)
                image_list.append(curr_image)
                if done:
                    break

            # per episode
            episode_lens.append(t + 1)
            rewards = np.array(rewards)
            episode_return = np.sum(rewards)
            episode_returns.append(episode_return)
            episode_highest_reward = np.max(rewards)
            highest_rewards.append(episode_highest_reward)
            is_success = (reward == env_max_reward)
            success_rates.append(is_success)
            task_successes.append(is_success)

            tag = f'task{task_id}_' if task_id is not None else ''
            print(f'Rollout {tag}{rollout_id} : {episode_return=}, Success: {is_success}')
            video = np.stack(image_list).transpose(0, 3, 1, 2)
            wandb_logger.log({f'eval_video/{tag}{rollout_id}': wandb.Video(video, fps=50)}, step=i)

        if task_id is not None and len(tasks) > 1:
            wandb_logger.log(
                {f'evaluation/success_rate_task{task_id}': float(np.mean(task_successes))}, step=i)

    num_rollouts = len(success_rates)
    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

def _perform_control_eval_vec(agent, tasks, i, variant, wandb_logger, agent_dp, vec_env):
    """Vec-env eval: for each task, pin all slots to that task and run enough
    batches to gather `variant.eval_episodes` rollouts. Trims overshoot from
    the last batch so per-task episode counts match the single-env path."""
    env_max_reward = variant.env_max_reward
    tasks_by_id = {t['task_id']: t for t in tasks if 'task_id' in t}
    N = vec_env.num_envs

    vec_env.reseed_perturbations(seed=variant.seed + 456)
    rng = jax.random.PRNGKey(variant.seed + 456)

    episode_returns, highest_rewards, success_rates, episode_lens = [], [], [], []

    for task in tasks:
        tid = task['task_id']
        task_description = task.get('description')
        task_successes = []
        rollout_id = 0
        while len(task_successes) < variant.eval_episodes:
            rng, sub = jax.random.split(rng)
            batch = collect_traj_vec(
                variant, agent, vec_env, tasks_by_id, i, agent_dp,
                task_ids=[tid] * N, rng=sub,
            )
            for tr in batch:
                if len(task_successes) >= variant.eval_episodes:
                    break
                env_rewards = tr['env_rewards']
                episode_returns.append(float(np.sum(env_rewards)))
                highest_rewards.append(float(np.max(env_rewards)) if len(env_rewards) else 0.0)
                success_rates.append(bool(tr['is_success']))
                task_successes.append(bool(tr['is_success']))
                episode_lens.append(tr['env_steps'])
                print(f'Rollout task{tid}_{rollout_id} ({task_description}): '
                      f'episode_return={episode_returns[-1]}, Success: {tr["is_success"]}')
                video = np.stack(tr['images']).transpose(0, 3, 1, 2)
                wandb_logger.log({f'eval_video/task{tid}_{rollout_id}': wandb.Video(video, fps=50)}, step=i)
                rollout_id += 1

        if len(tasks) > 1:
            wandb_logger.log(
                {f'evaluation/success_rate_task{tid}': float(np.mean(task_successes))}, step=i)

    num_rollouts = len(success_rates)
    success_rate = float(np.mean(success_rates))
    avg_return = float(np.mean(episode_returns))
    avg_episode_len = float(np.mean(episode_lens))
    summary = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    for r in range(env_max_reward + 1):
        more_or_equal_r = int((np.array(highest_rewards) >= r).sum())
        rate = more_or_equal_r / num_rollouts
        wandb_logger.log({f'evaluation/Reward >= {r}': rate}, step=i)
        summary += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {rate * 100}%\n'
    print(summary)


def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger):
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(variant, trajs)
    wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)
  
