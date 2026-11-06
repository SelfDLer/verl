# Step5 LoRA with FSDP-Turbo

Step5 uses PEFT `target_modules`; no separate attention/MoE mode flag is needed.
Set `actor_rollout_ref.model.lora_rank` and `lora_alpha`, and keep
`actor_rollout_ref.model.lora.merge=true` for rollout synchronization.

Targets (regular expressions passed as quoted Hydra strings):

```bash
# Attention only: the existing baseline
'actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj]'

# Routed experts only
'actor_rollout_ref.model.target_modules=".*\.moe\.experts\.(gate_up_proj|down_proj)"'

# Attention and routed experts
'actor_rollout_ref.model.target_modules=".*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|moe\.experts\.(gate_up_proj|down_proj))"'
```

`gate_up_proj` selects independent gate/up A/B pairs while keeping base FC1 fused.
Select `.*\.moe\.experts\.down_proj` for expert FC2 only. Bare `down_proj` also
matches shared/dense projections and is rejected by Step5's scope check.
`all-linear`, router, vision and indexer LoRA are outside this integration's scope.

The engine prepares expert modules before PEFT injection, including when loading
`lora_adapter_path`. Custom GroupedLinear adapters are registered on the saved
PEFT config before reload. Their three-dimensional factors are a custom adapter
format: reload through this engine rather than an unconfigured vanilla PEFT call.
Rollout receives merged ordinary HF weights, not vLLM-native MoE adapters.

Supported expert execution is native eager EP (including fused FC1), with training
TP/CP size 1. MC2 expert LoRA is rejected. Classic zero-B initialization without
dropout or LoRA variants is supported. Attention-only uses standard PEFT Linear.
