import numpy as np
import torch
import zarr
import json
from pathlib import Path


def _load_trajectory_language_records(path):
    text = Path(path).read_text(encoding='utf-8').strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records

def obs_wrapper(obs, dof=0):
    pcs = obs['pc']
    device = pcs.device
    if dof == 0:
        dof_state = torch.tensor([]).to(device)
    elif dof == 1:
        dof_state = obs['dof_state'][1:]  # do not reserve the dof state of the prismatic joint
    elif dof == 2:
        dof_state = obs['dof_state']
    env_state = torch.cat([obs['proprioception'], dof_state, obs['prev_action']], dim=-1)
    return pcs, env_state

class Episode_Buffer:
    def __init__(self):
        self.pcs = []
        self.env_state = []
        self.action = []

    def add(self, pc, env_state, action):
        self.pcs.append(pc.cpu().numpy())
        self.env_state.append(env_state.cpu().numpy())
        self.action.append(action.cpu().numpy())

def nested_dict_save(zarr_group, data, path):
    for key,value in data.items():
        if isinstance(value, dict):
            nested_dict_save(zarr_group, value, path+'/'+key)
        else:
            zarr_group[path+'/'+key] = value

class Experience:
    def __init__(self, sample_pcs_num=1000):
        self.sample_pcs_num = sample_pcs_num
        self.data = {"pcs":[], "env_state": [], "action": []}
        self.meta = {"episode_ends": []}
    
    def append(self, episode:Episode_Buffer):
        if len(episode.pcs) == 0:
            # print("skip empty eps")
            return        
        if self.meta["episode_ends"] == []:
            self.data["pcs"] = np.array(episode.pcs)
            self.data["env_state"] = np.array(episode.env_state)
            self.data["action"] = np.array(episode.action)
        else:
            self.data["pcs"] = np.concatenate([self.data["pcs"], np.array(episode.pcs)])
            self.data["env_state"] = np.concatenate([self.data["env_state"], np.array(episode.env_state)])
            self.data["action"] = np.concatenate([self.data["action"], np.array(episode.action)])
        new_end = self.data["pcs"].shape[0]
        self.meta["episode_ends"].append(new_end)

    def save(self, path):
        # save data and meta in one file as npy
        self.meta["episode_ends"] = np.array(self.meta["episode_ends"])
        zarr_group = zarr.open(path, 'w')
        nested_dict_save(zarr_group, self.data, 'data')
        nested_dict_save(zarr_group, self.meta, 'meta')

def create_sample_indices(episode_ends:np.ndarray, obs_length:int, action_length: int, pad_after: int):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        pad_before = obs_length
        
        # action length is the pred_horizon, obs length is the obs_horizon, pad_after is the action_horizon
        for idx in range(episode_length-action_length+pad_after):
            action_start_idx = idx + start_idx
            action_end_idx = min(idx+action_length,episode_length) + start_idx
            obs_start_idx = max(idx-obs_length+1,0) + start_idx
            obs_end_idx = idx + 1 + start_idx

            indices.append([action_start_idx, action_end_idx, obs_start_idx, obs_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, obs_length, action_length,
                    action_start_idx, action_end_idx, obs_start_idx, obs_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        if key != "action":
            sample = input_arr[obs_start_idx:obs_end_idx]
            if obs_end_idx - obs_start_idx < obs_length:
                data = np.zeros(shape=(obs_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
                data[:obs_length-obs_end_idx+obs_start_idx] = sample[0]
                data[obs_length-obs_end_idx+obs_start_idx:] = sample
            else:
                data = sample
        else:
            sample = input_arr[action_start_idx:action_end_idx]
            if action_end_idx - action_start_idx < action_length:
                data = np.zeros(shape=(action_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
                data[:action_end_idx-action_start_idx] = sample
                data[action_end_idx-action_start_idx:] = sample[-1]
            else:
                data = sample

        result[key] = data
    return result


# dataset
class ManipDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: list,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):
        
        # save obs and action
        # obs: global point clouds, joint pose, gripper pose
        # action: joint pose, joint velocity, gripper force
        # read from npy dataset
        print("using new Manip Dataset")
        print(f"loading data from {dataset_path}")
        pcs_list = []
        action_data_list = []
        env_state_list = []
        episode_end_list = []

        if isinstance(dataset_path, (str, Path)):
            dataset_path = [dataset_path]

        dataset_paths = [Path(p) for p in dataset_path]

        local_episode_counts = []

        init = 0
        for data_path in dataset_paths:
            data = zarr.open(data_path, 'a')
            ends = data['meta']['episode_ends'][:]
            episode_end_list.append(ends + init)
            init += ends[-1]
            pcs_list.append(data['data']['pcs'])
            action_data_list.append(data['data']['action'])
            env_state_list.append(data['data']['env_state'])
            local_episode_counts.append(len(ends))

        '''
        seg pointcloud don not need moveaxis
        '''
        pcs = np.concatenate(pcs_list, axis=0)
        action_data = np.concatenate(action_data_list, axis=0)
        env_state = np.concatenate(env_state_list, axis=0)
        action_train = action_data
        train_data = {
            'pcs': pcs,
            'env_state': env_state,
            'action': action_train#dataset_root['data']['action'][:]
        }
        episode_ends = np.concatenate(episode_end_list, axis=0)
        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            obs_length= obs_horizon,
            action_length= pred_horizon,
            pad_after= pred_horizon #action_horizon-1
        )
        print("pad after is", pred_horizon)
        self.indices = indices
        self.train_data = train_data
        self.episode_ends = episode_ends
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

        self.language_enabled = False
        self._episode_lang_index = {}
        self._language_chain_embeddings = []

        traj_paths = [p.parent / 'trajectory_language.jsonl' for p in dataset_paths]
        emb_paths = [p.parent / 'language_embedding_dict.json' for p in dataset_paths]

        has_any_language = any(tp.exists() and ep.exists() for tp, ep in zip(traj_paths, emb_paths))
        has_all_language = all(tp.exists() and ep.exists() for tp, ep in zip(traj_paths, emb_paths))

        if has_any_language and not has_all_language:
            raise FileNotFoundError(
                'Language files are incomplete across dataset paths. '
                'Expected both trajectory_language.jsonl and language_embedding_dict.json for each dataset.'
            )

        if has_all_language:
            episode_offset = 0
            for source_idx, (traj_path, emb_path, local_episode_count) in enumerate(
                zip(traj_paths, emb_paths, local_episode_counts)
            ):
                traj_records = _load_trajectory_language_records(traj_path)
                traj_map = {int(item['episode_id']): item for item in traj_records}

                with emb_path.open('r', encoding='utf-8') as f:
                    emb_dict = json.load(f)
                chain_embeddings = np.asarray(emb_dict['expanded_minimal_chains'], dtype=np.float32)
                self._language_chain_embeddings.append(chain_embeddings)

                for local_episode_id in range(local_episode_count):
                    if local_episode_id not in traj_map:
                        raise KeyError(
                            f'Missing episode_id={local_episode_id} in {traj_path}'
                        )
                    chain_ids = traj_map[local_episode_id].get('command_chain_ids', None)
                    if not chain_ids:
                        raise ValueError(
                            f'Empty command_chain_ids for episode_id={local_episode_id} in {traj_path}'
                        )

                    chain_ids = [int(cid) for cid in chain_ids]
                    max_chain_id = chain_embeddings.shape[0] - 1
                    if max(chain_ids) > max_chain_id or min(chain_ids) < 0:
                        raise IndexError(
                            f'command_chain_ids out of range in {traj_path}: {chain_ids}, '
                            f'valid range [0, {max_chain_id}]'
                        )

                    global_episode_id = episode_offset + local_episode_id
                    self._episode_lang_index[global_episode_id] = (source_idx, chain_ids)

                episode_offset += local_episode_count

            self.language_enabled = True
            print('language conditioning enabled in ManipDataset')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        action_start_idx, action_end_idx, obs_start_idx, obs_end_idx = self.indices[idx]
        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.train_data,
            obs_length=self.obs_horizon,
            action_length=self.pred_horizon,
            action_start_idx=action_start_idx,
            action_end_idx=action_end_idx,
            obs_start_idx=obs_start_idx,
            obs_end_idx=obs_end_idx
        )

        if self.language_enabled:
            frame_idx = int(action_start_idx)
            episode_id = int(np.searchsorted(self.episode_ends, frame_idx, side='right'))
            source_idx, chain_ids = self._episode_lang_index[episode_id]
            chain_id = int(np.random.choice(chain_ids))
            lang_embedding = self._language_chain_embeddings[source_idx][chain_id].astype(np.float32)

            nsample['language_embedding'] = lang_embedding
            nsample['language_chain_id'] = np.int64(chain_id)

        return nsample

def merge_dataset(path_list, to_path):
    for path in path_list:
        dataset_root = zarr.open(path, 'r')
        if path == path_list[0]:
            pcs_data = dataset_root['data']['pcs'][:]
            pose_data = dataset_root['data']['env_state'][:]
            action_data = dataset_root['data']['action'][:]
            meta = dataset_root['meta']['episode_ends'][:]
            cur_len = dataset_root['meta']['episode_ends'][-1]
        else:
            pcs_data = np.concatenate([pcs_data, dataset_root['data']['pcs'][:]])
            pose_data = np.concatenate([pose_data, dataset_root['data']['env_state'][:]])
            action_data = np.concatenate([action_data, dataset_root['data']['action'][:]])
            new_meta = dataset_root['meta']['episode_ends'][:] + cur_len
            cur_len += dataset_root['meta']['episode_ends'][-1]
            meta = np.concatenate([meta, new_meta])
        print(pcs_data.shape, pose_data.shape, action_data.shape, meta.shape)

    # save data and meta in one file using zarr
    all_data = {"pcs":pcs_data, "env_state": pose_data, "action": action_data}
    all_meta = {"episode_ends": meta}
    zarr_group = zarr.open(to_path, 'w')
    nested_dict_save(zarr_group, all_data, 'data')
    nested_dict_save(zarr_group, all_meta, 'meta')