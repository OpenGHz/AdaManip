# AdaManip Training Data Loading Notes

Please use `docs/data_collection.md` as the source of truth for collection-side storage and writing semantics.

## 1. Multi-File Dataset Merge Logic

`ManipDataset` accepts `dataset_path` as a list.

When multiple zarr files are provided:

- all `pcs`, `env_state`, and `action` arrays are concatenated;
- each file's `episode_ends` is shifted by the cumulative frame count before concatenation.

So multiple files behave like one large dataset, but episode boundaries remain correct.

## 2. How Multi-Frame Windows Are Built

Training uses `ManipDataset` in `dataset/dataset.py`.

The two important horizons are:

- `obs_horizon`: number of observation frames used as input
- `pred_horizon`: number of future action frames predicted as output

For each episode, `create_sample_indices()` generates one training sample per timestep.

For a local timestep `idx` inside one episode, it computes:

- `action_start_idx = idx + start_idx`
- `action_end_idx = min(idx + pred_horizon, episode_length) + start_idx`
- `obs_start_idx = max(idx - obs_horizon + 1, 0) + start_idx`
- `obs_end_idx = idx + 1 + start_idx`

So each sample means:

- input: the latest `obs_horizon` observation frames up to the current frame
- target: the next `pred_horizon` action frames starting from the current frame

Because `pad_after` is currently set to `pred_horizon`, every frame in an episode can become one training sample.

## 3. Padding Rules at Episode Boundaries

Since both input and output are multi-frame, the code pads at episode boundaries instead of dropping boundary samples.

### 3.1 Observation Padding at Episode Start

If there are not enough past frames to fill `obs_horizon`, the code repeats the first available observation frame.

Example with `obs_horizon = 2` at the first frame of an episode:

- observation window becomes `[frame_0, frame_0]`

So observation padding is left-padding by repeating the first frame.

### 3.2 Action Padding at Episode End

If there are not enough future actions to fill `pred_horizon`, the code repeats the last available action.

Example with `pred_horizon = 4` and only one future action left:

- action window becomes `[last_action, last_action, last_action, last_action]`

So action padding is right-padding by repeating the last action.

### 3.3 Practical Meaning

This gives the following behavior:

- early timesteps see repeated initial observations;
- late timesteps predict repeated terminal actions.

The implementation keeps boundary frames trainable instead of discarding them.

## 4. Why Windows Never Cross Different Episodes

A training sample cannot cross episode boundaries.

Reason:

- sample indices are generated episode by episode;
- each episode has its own `start_idx` and `end_idx`;
- observation indices are clipped to the episode start;
- action indices are clipped to the episode end.

So even when the model needs history before the first frame or future actions after the last frame, it uses padding inside the same episode instead of reading data from another episode.

This prevents the jump/discontinuity issue that would happen if one window mixed frames from different episodes.

## 5. Training Order: Whole Episode or Random Switching?

Training does not finish one episode before switching to another.

Instead, the logic is:

1. Build all valid windows from all episodes.
2. Flatten them into one large sample index list.
3. Pass that list into `DataLoader(..., shuffle=True)`.

Because `shuffle=True`, minibatches are random mixtures of samples from different episodes.

So during training:

- consecutive minibatches may come from different episodes;
- one minibatch can already contain windows from many different episodes;
- training is supervised over shuffled windows, not sequential episode-by-episode rollout.

## 6. Direct Answers to the Main Questions

### Q1. Is there padding because the model input and output are multi-frame?

Yes.

- observation at episode start is padded by repeating the first frame;
- action at episode end is padded by repeating the last action.

### Q2. Can one prediction range cross different episodes?

No.

The index generation is episode-aware and clips every window to the current episode boundaries.

### Q3. Is one episode trained completely before switching to another?

No.

Training windows from all episodes are shuffled together, so the training loop randomly switches between episodes throughout optimization.

## 7. Notes About the Current Implementation

A few code details are worth keeping in mind:

- `pad_before = obs_length` is assigned in `create_sample_indices()` but is not actually used.
- The comment mentions `action_horizon-1`, but the code currently uses `pred_horizon` directly as `pad_after`.
- The comment says "data normalized in dataset", but `ManipDataset` itself does not do explicit normalization.

These do not affect episode isolation, but they are useful details if the training pipeline is modified later.

## 8. Language Sampling During Training

This section describes the language-conditioning sampling path for each training sample.

Prerequisite files:

- `trajectory_language.jsonl`
- `language_embedding_dict.json`

### 8.1 Step 1: map sample to episode_id

For one dataset sample index `sample_idx`, `ManipDataset.indices[sample_idx]` gives:

- `action_start_idx`
- `action_end_idx`
- `obs_start_idx`
- `obs_end_idx`

Use `action_start_idx` (or equivalently `obs_end_idx - 1`) as the anchor frame index `frame_idx`.

Given global `episode_ends` (strictly increasing), compute:

- `episode_id = searchsorted(episode_ends, frame_idx, side="right")`

This works because each episode is a half-open range:

- episode 0: `[0, episode_ends[0])`
- episode 1: `[episode_ends[0], episode_ends[1])`
- ...

### 8.2 Step 2: fetch trajectory language entry

From `trajectory_language.jsonl`, fetch the record whose `episode_id` equals the computed `episode_id`.

Required field in that record:

- `command_chain_ids` (a non-empty list of candidate chain ids)

### 8.3 Step 3: uniformly sample one chain_id

Sample one id uniformly from `command_chain_ids`:

- `chain_id = random.choice(command_chain_ids)`

Uniform means each candidate id has probability `1 / len(command_chain_ids)`.

### 8.4 Step 4: lookup embedding from dictionary

Load `language_embedding_dict.json` once at dataset initialization.

Then fetch chain embedding by id:

- `chain_embedding = embedding_dict["expanded_minimal_chains"][chain_id]`

Optionally, if operation-level conditioning is needed for another training target:

- `op_embedding = embedding_dict["operation_set"][operation_str]`

### 8.5 Pseudocode

```python
# preloaded at dataset init
episode_ends: np.ndarray
traj_lang: Dict[int, dict]   # key = episode_id
emb_dict: dict

def sample_language_embedding(sample_idx):
	action_start_idx, action_end_idx, obs_start_idx, obs_end_idx = indices[sample_idx]
	frame_idx = action_start_idx

	episode_id = int(np.searchsorted(episode_ends, frame_idx, side="right"))
	traj_item = traj_lang[episode_id]

	chain_ids = traj_item["command_chain_ids"]
	chain_id = int(np.random.choice(chain_ids))

	chain_embedding = np.asarray(
		emb_dict["expanded_minimal_chains"][chain_id],
		dtype=np.float32,
	)
	return chain_id, chain_embedding
```

### 8.6 Integrity checks (recommended)

At dataset startup, validate:

- every `episode_id` in zarr has one trajectory-language record;
- every `command_chain_ids` is non-empty;
- every sampled `chain_id` is in `[0, len(expanded_minimal_chains) - 1]`.
