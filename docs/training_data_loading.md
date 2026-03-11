# AdaManip Training Data Loading Notes

## 1. Data Storage Format

AdaManip stores training data in a zarr file, usually `demo_data.zip`.

The important fields are:

- `data/pcs`: point cloud for each saved frame.
- `data/env_state`: low-dimensional observation for each saved frame.
- `data/action`: action for each saved frame.
- `meta/episode_ends`: cumulative frame offsets marking the end of each episode.

`episode_ends` is the key to episode isolation. For example, if `episode_ends = [10, 25, 42]`, then:

- episode 0 uses frames `[0, 10)`
- episode 1 uses frames `[10, 25)`
- episode 2 uses frames `[25, 42)`

So all frames are concatenated into one long array, but episode boundaries are preserved explicitly.

## 2. How Episodes Are Written During Collection

During microwave data collection, each environment instance keeps its own `Episode_Buffer`.

The collection flow is:

1. Create one `Episode_Buffer` per environment.
2. Every call to `process_data()` appends one `(pc, env_state, action)` frame to that environment's buffer.
3. After the rollout, each successful environment buffer is appended into the global `Experience` buffer.
4. At the end, `Experience.save()` writes everything into one zarr file.

This means one saved episode corresponds to one successful environment rollout.

For microwave collection specifically, one outer rollout loop may generate multiple saved episodes, because `num_envs` environments run in parallel and each successful env is appended separately.

## 3. Multi-File Dataset Merge Logic

`ManipDataset` accepts `dataset_path` as a list.

When multiple zarr files are provided:

- all `pcs`, `env_state`, and `action` arrays are concatenated;
- each file's `episode_ends` is shifted by the cumulative frame count before concatenation.

So multiple files behave like one large dataset, but episode boundaries remain correct.

## 4. How Multi-Frame Windows Are Built

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

## 5. Padding Rules at Episode Boundaries

Since both input and output are multi-frame, the code pads at episode boundaries instead of dropping boundary samples.

### 5.1 Observation Padding at Episode Start

If there are not enough past frames to fill `obs_horizon`, the code repeats the first available observation frame.

Example with `obs_horizon = 2` at the first frame of an episode:

- observation window becomes `[frame_0, frame_0]`

So observation padding is left-padding by repeating the first frame.

### 5.2 Action Padding at Episode End

If there are not enough future actions to fill `pred_horizon`, the code repeats the last available action.

Example with `pred_horizon = 4` and only one future action left:

- action window becomes `[last_action, last_action, last_action, last_action]`

So action padding is right-padding by repeating the last action.

### 5.3 Practical Meaning

This gives the following behavior:

- early timesteps see repeated initial observations;
- late timesteps predict repeated terminal actions.

The implementation keeps boundary frames trainable instead of discarding them.

## 6. Why Windows Never Cross Different Episodes

A training sample cannot cross episode boundaries.

Reason:

- sample indices are generated episode by episode;
- each episode has its own `start_idx` and `end_idx`;
- observation indices are clipped to the episode start;
- action indices are clipped to the episode end.

So even when the model needs history before the first frame or future actions after the last frame, it uses padding inside the same episode instead of reading data from another episode.

This prevents the jump/discontinuity issue that would happen if one window mixed frames from different episodes.

## 7. Training Order: Whole Episode or Random Switching?

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

## 8. Direct Answers to the Main Questions

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

## 9. Notes About the Current Implementation

A few code details are worth keeping in mind:

- `pad_before = obs_length` is assigned in `create_sample_indices()` but is not actually used.
- The comment mentions `action_horizon-1`, but the code currently uses `pred_horizon` directly as `pad_after`.
- The comment says "data normalized in dataset", but `ManipDataset` itself does not do explicit normalization.

These do not affect episode isolation, but they are useful details if the training pipeline is modified later.
