import dataclasses
import gc
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from data_types import Episode, MiniBatch, TensorMiniBatch


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cpu":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    batch: MiniBatch,
    tokenizer: Any,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    prefix_token_ids = batch.prefix_token_ids
    bsz = len(batch.prefix) * num_answer_per_question
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        with _autocast(device, dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = next_token.reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        # if an rollout is finished, we fill the rest of the tokens with pad_token_id
        next_token = torch.where(is_finished, pad_token_id, next_token)
        tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        if is_finished.all():
            break
    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    # prepare the output episodes
    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]) :]
            # remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            generated_text = tokenizer.detokenize(generated_token_ids)
            rewards = reward_function(
                response=generated_text,
                numbers=batch.numbers[i],
                target=batch.target[i],
                end_token=end_token,
            )
            episode = Episode(
                group_id=batch.prefix[i],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished_list[idx],
                metadata={
                    "question_index": i,
                    "sample_index": j,
                    "response_text": generated_text,
                },
            )
            episodes.append(episode)
    # clear the output line
    print("\r", end=" " * 100, flush=True)
    return episodes


@torch.no_grad()
def rollout_tensor_with_mc_dropout(
    model: torch.nn.Module,
    batch: TensorMiniBatch,
    num_samples: int,
    reward_function: Callable[..., Dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
    policy_metadata_fn: Optional[
        Callable[[torch.Tensor, Any, Optional[Dict[str, Any]]], Dict[str, Any]]
    ] = None,
) -> List[Episode]:
    """Rollout trajectories for tensor-only models using Monte Carlo Dropout.

    Args:
        model: Vision/backbone model whose final tensor output is optimised.
        batch: Mini-batch of tensor inputs with optional targets and metadata.
        num_samples: Number of Monte Carlo samples per input.
        reward_function: Computes reward directly from tensor outputs.
        device / dtype: Execution context for the forward passes.
        policy_metadata_fn: Optional callable to populate policy-specific
            metadata (e.g. sampled action index) required by the policy
            adapter during the optimisation step.
    """

    was_training = model.training
    # Keep dropout layers active while disabling gradients.
    model.train()
    inputs = batch.inputs
    if batch.targets is None:
        targets: List[Any] = [None] * len(inputs)
    elif isinstance(batch.targets, torch.Tensor):
        targets = [t.detach().cpu() for t in batch.targets]
    else:
        targets = list(batch.targets)
    metadata_list = (
        batch.metadata if batch.metadata is not None else [{} for _ in range(len(inputs))]
    )
    group_ids = (
        batch.group_ids if batch.group_ids is not None else list(range(len(inputs)))
    )

    batch_size = len(inputs)
    if batch_size == 0:
        return []

    # Vectorise Monte Carlo sampling by repeating the full batch ``num_samples``
    # times so that a single forward pass covers all dropout realisations.
    inputs_device = inputs.to(device, non_blocking=True)
    repeated_inputs = inputs_device.repeat_interleave(num_samples, dim=0)
    with _autocast(device, dtype):
        repeated_outputs = model(repeated_inputs)
    del repeated_inputs
    repeated_outputs_cpu = repeated_outputs.detach().cpu()

    episodes: List[Episode] = []
    for idx, (group_id, sample_metadata) in enumerate(zip(group_ids, metadata_list)):
        target = targets[idx]
        start = idx * num_samples
        end = start + num_samples
        sample_outputs = repeated_outputs_cpu[start:end]
        model_input_cpu = inputs[idx].unsqueeze(0).detach().cpu()

        for sample_idx, output_cpu in enumerate(sample_outputs):
            print(
                f"\r* Generating MC dropout samples: {sample_idx + 1:>2d}/{num_samples:>2d}"
                f" (image {idx + 1:>2d}/{batch_size:>2d})",
                flush=True,
                end="",
            )
            output_cpu_expanded = output_cpu.unsqueeze(0)
            reward_payload = reward_function(
                response=output_cpu_expanded,
                target=target,
                sample_metadata=sample_metadata,
            )
            episode_metadata: Dict[str, Any] = {
                "model_inputs": model_input_cpu,
                "target": target,
                "sample_metadata": sample_metadata,
                "mc_sample_index": sample_idx,
            }
            policy_payload = reward_payload.get("policy_metadata")
            if policy_payload is None and policy_metadata_fn is not None:
                policy_payload = policy_metadata_fn(
                    output_cpu_expanded, target, sample_metadata
                )
            if policy_payload is not None:
                episode_metadata["policy_metadata"] = policy_payload

            episode = Episode(
                group_id=group_id,
                reward=reward_payload["reward"],
                reward_info=reward_payload.get("reward_info", {}),
                response_tensor=output_cpu_expanded,
                metadata=episode_metadata,
            )
            episodes.append(episode)

    if not was_training:
        model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    print("\r", end=" " * 100, flush=True)
    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """Normalize rewards per group.

    The grouping key corresponds to the originating input (``Episode.group_id``)
    so that all Monte Carlo samples drawn from the same stimulus share the same
    normalisation statistics.
    """
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode.group_id].append(episode)
    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def update_policy(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: Optional[int],
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    policy_adapter: Optional[Callable[[torch.Tensor, Sequence[Episode]], Dict[str, torch.Tensor]]] = None,
):
    """Update the policy using the GRPO algorithm.

    When ``Episode.response_tensor`` is populated, the function switches to the
    tensor-oriented optimisation path that works with MC Dropout samples and
    policy adapters.  Otherwise it falls back to the legacy language-model
    routine.
    """

    has_tensor_responses = any(episode.response_tensor is not None for episode in episodes)
    has_textual_responses = any(
        episode.generated_token_ids is not None for episode in episodes
    )

    if has_tensor_responses and has_textual_responses:
        raise ValueError("Mixed response types are not supported within the same batch")

    if has_tensor_responses:
        return _update_policy_tensor(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=micro_batch_size,
            max_grad_norm=max_grad_norm,
            device=device,
            dtype=dtype,
            policy_adapter=policy_adapter,
        )

    return _update_policy_text(
        model=model,
        optimizer=optimizer,
        episodes=episodes,
        micro_batch_size=micro_batch_size,
        pad_token_id=pad_token_id,
        max_grad_norm=max_grad_norm,
        device=device,
        dtype=dtype,
    )


def _update_policy_text(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
):
    episodes = normalize_rewards_per_group(episodes)
    # sort episodes by token length for efficient (micro-)batching
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = 0.0

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with _autocast(device, dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            logits = model.forward(input_token_ids).float()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens

        obj = log_probs * batch_advantages[:, None]
        # per-token objective
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }


def _update_policy_tensor(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    policy_adapter: Optional[Callable[[torch.Tensor, Sequence[Episode]], Dict[str, torch.Tensor]]],
):
    if policy_adapter is None:
        raise ValueError(
            "A policy_adapter must be supplied when optimising tensor responses"
        )

    episodes = normalize_rewards_per_group(episodes)
    episodes.sort(
        key=lambda episode: (
            episode.metadata.get("question_index", 0),
            episode.metadata.get("mc_sample_index", 0),
        )
    )

    entropy_accum = 0.0
    samples_seen = 0
    loss = torch.tensor(0.0, device=device)

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing tensor policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_inputs = torch.cat(
            [episode.metadata["model_inputs"] for episode in batch_episodes], dim=0
        ).to(device)
        batch_advantages = torch.tensor(
            [episode.reward for episode in batch_episodes],
            device=device,
            dtype=torch.float32,
        )

        with _autocast(device, dtype):
            outputs = model(batch_inputs).float()

        adapter_output = policy_adapter(outputs, batch_episodes)
        log_probs = adapter_output["log_prob"]
        if log_probs.ndim > 1:
            log_probs = log_probs.squeeze(-1)
        obj = (log_probs * batch_advantages).mean()
        loss = -obj
        loss.backward()

        entropy_values = adapter_output.get("entropy")
        if entropy_values is None:
            with torch.no_grad():
                try:
                    entropy_values = compute_entropy(outputs)
                except RuntimeError:
                    entropy_values = torch.zeros_like(log_probs)
        if entropy_values.ndim > 1:
            entropy_values = entropy_values.squeeze(-1)
        entropy_accum += entropy_values.detach().sum().item()
        samples_seen += len(batch_episodes)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    mean_entropy = entropy_accum / max(samples_seen, 1)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": mean_entropy,
    }


def classification_policy_adapter(
    logits: torch.Tensor,
    episodes: Sequence[Episode],
) -> Dict[str, torch.Tensor]:
    """Default adapter for classification-style tensor outputs.

    The adapter interprets the logits as a categorical distribution and returns
    the log probability of the action that was implicitly taken during rollout.
    The action index can be supplied via ``Episode.metadata['policy_metadata']``;
    otherwise the adapter falls back to the argmax of the stored response
    tensor.
    """

    if logits.ndim < 2:
        raise ValueError("Classification adapter expects batched logits")

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    action_indices: List[int] = []
    for episode in episodes:
        policy_metadata = episode.metadata.get("policy_metadata", {})
        action = policy_metadata.get("action_index")
        if action is None:
            response_tensor = episode.response_tensor
            if response_tensor is None:
                raise ValueError(
                    "Response tensor missing – provide action_index via policy_metadata"
                )
            flattened = response_tensor.reshape(-1)
            action = int(torch.argmax(flattened).item())
        action_indices.append(int(action))

    action_tensor = torch.tensor(action_indices, device=log_probs.device, dtype=torch.long)
    chosen_log_probs = log_probs.gather(-1, action_tensor.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return {"log_prob": chosen_log_probs, "entropy": entropy}
