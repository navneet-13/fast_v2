Sugegstion for batch stuck with multi-request:

Based on the `BatchBucket` logic you shared and your `batch_sample_dynamo` loop, the "infinite loop" with Batch Size 8 happens because the **global completion check** doesn't account for the **per-slot status**.

In a batch of 1, "all" is "one." In a batch of 8, if Sequence A finishes in 2 steps but Sequence B takes 10, the code keeps seeing `mask_id` tokens in Sequence B's slots and refuses to exit the block loop for the entire batch. Worse, if your `pad_finished_slots` isn't called inside the inner `while` loop, the finished slots might still contain `mask_id`, trapping the whole batch in an infinite loop.

### The Fix

You need to apply the `active_mask` to your `mask_id` checks so that finished sequences are ignored during the "is this block done?" calculation.

#### 1. Update the Inner Diffusion Exit Condition

In your `while True` (diffusion) loop, you check if the current sub-block is denoised. You must modify this to only look at active slots.

```python
# --- Inside the inner diffusion while loop ---
while True:
    current_block_slice = x_t[:, -block_size:]
    
    # OLD: mask_idx = (x_t[:, -block_size:] == mask_id)
    # NEW: Only consider mask_id in slots that are still ACTIVE
    actual_masks = (current_block_slice == mask_id) & bucket.active_mask[:, None]
    
    # Check if the specific small_block range for ACTIVE sequences is clear
    if actual_masks[:, start:end].sum() == 0 or bucket.all_finished():
        break
    
    # ... (forward_fn and sampling logic) ...

    # IMPORTANT: After sampling, ensure finished slots don't have mask_id
    bucket.pad_finished_slots(x_t, block_size)

```

#### 2. Update the Block Completion Check

Similarly, the check that determines if the whole 32-token block is done needs the same logic:

```python
# --- Inside the main loop, before the bridging token generation ---
while True:
    # ... (the sub-block loop above runs first) ...

    # OLD: mask_idx = (x_t[:, -block_size:] == mask_id)
    # NEW: Check completion only for slots that actually need to finish
    current_block_slice = x_t[:, -block_size:]
    active_masks_remaining = (current_block_slice == mask_id) & bucket.active_mask[:, None]

    if active_masks_remaining.sum() == 0:
        # Generate bridge token, update KV cache, etc.
        # ...
        break

```

---

### Why this fixes the Batch Size 8 hang:

1. **Transparency:** By using `& bucket.active_mask[:, None]`, you make finished sequences "transparent" to the exit condition. The sum will hit 0 as soon as the last *active* sequence finishes its tokens.
2. **Prevents Mask Poisoning:** If `pad_finished_slots` isn't called frequently, a sequence that finished early might still have `mask_id` tokens from its initialization. The logic above ignores those "ghost" masks.
3. **Correct Termination:** `bucket.all_finished()` handles the case where the entire batch hits a `stop_token` before the block is full.

### Suggested Optimization: `unmask_idx`

In your sampling section, ensure you are only unmasking for active slots to prevent overwriting padding:

```python
# Ensure we only apply newly sampled tokens to active, masked positions
unmask_idx = unmask_idx & mask_idx[:, start:end] & bucket.active_mask[:, None]
x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

```



