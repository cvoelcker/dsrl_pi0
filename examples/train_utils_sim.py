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
        _PSI_FN_CACHE[key] = nnx_utils.module_jit(agent_dp._model.get_psi_representation)
    return _PSI_FN_CACHE[key]


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
                                       perform_control_evals=True, shard_fn=None, agent_dp=None, tasks=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    # In the single-task / non-libero case, wrap the passed env in a one-entry
    # task registry so collection/eval share a single code path.
    if tasks is None:
        tasks = [{'env': env, 'description': variant.get('task_description'), 'language_emb': None}]

    total_env_steps = 0
    traj_idx = 0
    i = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
    
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            # Round-robin over tasks so the multitask critic sees every task evenly.
            task = tasks[traj_idx % len(tasks)]
            traj_idx += 1
            traj = collect_traj(variant, agent, task, i, agent_dp)
            traj_id = online_replay_buffer._traj_counter
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            print('total env steps:', total_env_steps)
            
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = len(traj["rewards"])*variant.multi_grad_step

            if len(online_replay_buffer) > variant.start_online_updates:
                for _ in range(num_gradsteps):
                    # perform first visualization before updating
                    if i == 0:
                        print('performing evaluation for initial checkpoint')
                        if perform_control_evals:
                            perform_control_eval(agent, tasks, i, variant, wandb_logger, agent_dp)
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
                        # wandb_logger.log({'replay_buffer_size': len(online_replay_buffer)}, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval(agent, tasks, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

            
def add_online_data_to_buffer(variant, traj, online_replay_buffer):

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

def perform_control_eval(agent, tasks, i, variant, wandb_logger, agent_dp=None):
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

def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger):
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(variant, trajs)
    wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)
  
