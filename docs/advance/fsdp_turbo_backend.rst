FSDP-Turbo backend
==================

Last updated: 08/31/2026.

FSDP-Turbo (``fsdp_turbo``) is a high-performance FSDP training backend built
on top of the `fsdp-turbo <https://gitcode.com/Ascend/FSDPTurbo.git>`_ library.
It extends verl's FSDP engine family (``fsdp``, ``fsdp2``) with native support
for hybrid parallelism that combines fully-sharded data parallelism with expert
parallelism (EP) and context parallelism (CP) on a single ``DeviceMesh``.

The backend works on both Ascend NPU and NVIDIA GPUs. It is worth noting that
the ``fsdp_turbo`` python path should be exported in related shell scripts,
i.e., export PYTHONPATH=/your_path/FSDPTurbo:$PYTHONPATH

Configuration
-------------

Select the backend by setting ``strategy=fsdp_turbo``:

.. code-block:: bash

   actor_rollout_ref.actor.strategy=fsdp_turbo
   actor_rollout_ref.ref.strategy=fsdp_turbo

The turbo-specific parallelism and memory options live under
``fsdp_config.turbo_config``.  The full schema is defined in
``verl/trainer/config/engine/fsdp.yaml``.

Distributed plan
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   turbo_config:
     distributed:
       fully_shard_parallel_size: 16   # FSDP group size
       tensor_parallel_size: 1         # reserved, keep 1
       expert_parallel_size: 8         # EP group size (MoE only)
       expert_fully_shard_parallel_size: 1
       ulysses_parallel_size: 1        # CP / Ulysses SP size
       fsdp_plan:
         param_dtype: bf16
         reduce_dtype: fp32
         output_dtype: bf16
         fsdp_implementation: native
         num_to_forward_prefetch: 1
         num_to_backward_prefetch: 1

Key fields:

* ``fully_shard_parallel_size`` — number of ranks that share a single set of
  parameter shards (the FSDP mesh).
* ``expert_parallel_size`` — number of ranks across which MoE experts are
  distributed.  Set to ``1`` for dense models.
* ``ulysses_parallel_size`` — context-parallel size.  When greater than 1,
  sequences are split along the head dimension and gathered with all-to-all.

Module-level sharding plans
~~~~~~~~~~~~~~~~~~~~~~~~~~~

FSDP-Turbo accepts explicit module-glob patterns to control which submodules
are sharded, hooked, or re-computed.  Patterns use ``{*}`` for integer indices:

.. code-block:: bash

   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.apply_modules='{model.language_model.layers.{*}:{},lm_head:{}}'
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.hook_modules="['model.language_model.layers.{*}']"
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.apply_modules="['model.language_model.layers.{*}.mlp.experts']"
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.dispatcher=fused

* ``fsdp_plan.apply_modules`` — modules to wrap with FSDP.
* ``fsdp_plan.hook_modules`` — modules whose forward hooks are managed by
  Turbo (needed for recompute overlap).
* ``ep_plan.apply_modules`` — modules whose parameters are split across the EP
  group (typically the MoE expert container).
* ``ep_plan.dispatcher`` — ``eager`` (default) or ``fused``.  Use ``fused``
  on NPU for better all-to-all overlap.

Memory plan
~~~~~~~~~~~

.. code-block:: bash

   +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute=True
   +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan="['model.language_model.layers.{*}','model.visual.blocks.{*}']"

When ``recompute=True``, activation checkpointing is applied to every module
listed in ``recompute_plan``.

Module patches and CP function patches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some Hugging Face model implementations need targeted patches so that
FSDP-Turbo's CP path can intercept attention and loss computation.  These
patches are only required when context parallelism is enabled
(``ulysses_parallel_size > 1``); the bundled example scripts add them
conditionally via ``if [ "${SP_SIZE}" -gt 1 ]``.

Three patch lists are involved (all under ``turbo_config``):

.. code-block:: bash

   # cp_plan.ulysses_function_patches — attention / GDN forwards split by CP
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.cp_plan.ulysses_function_patches="[{target_functions:['transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.eager_attention_forward'],type:full_attention},{target_functions:['transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet.forward'],type:gated_delta_net}]"

   # cp_plan.loss_function_patches — loss recomputation across CP ranks
   +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.cp_plan.loss_function_patches="[{target_functions:['transformers.loss.loss_utils.ForCausalLMLoss'],type:causal_lm_loss}]"

   # module_patches — replace a HF forward with a Turbo-compatible one
   +actor_rollout_ref.actor.fsdp_config.turbo_config.module_patches="[{target:transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeModel.forward,replacement:fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5_moe_model_forward}]"

Patch entry fields:

* ``ulysses_function_patches`` / ``loss_function_patches`` — each entry has a
  ``target_functions`` list (fully-qualified callables) and a ``type`` that
  tells Turbo which CP-aware implementation to substitute.  Attention patches
  use ``full_attention``; gated-delta-net layers use ``gated_delta_net``; the causal-LM
  loss uses ``causal_lm_loss``.
* ``module_patches`` — each entry has ``target`` (the original function) and
  ``replacement`` (the Turbo replacement, shipped under the ``fsdp_turbo``
  package).  The replacement is applied at engine initialization time.

The dense Qwen3.5 scripts point at ``transformers.models.qwen3_5.modeling_qwen3_5``
instead of the ``qwen3_5_moe`` module, and use
``fsdp_turbo.models.qwen.qwen3_5.qwen3_5_model_forward`` as the replacement.
The patch structure is otherwise identical.

Offload policy
~~~~~~~~~~~~~~

FSDP-Turbo reuses verl's existing offload flags:

.. code-block:: bash

   actor_rollout_ref.actor.fsdp_config.offload_policy=True
   actor_rollout_ref.actor.fsdp_config.param_offload=True
   actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

When ``offload_policy`` is ``True``, Turbo applies ``CPUOffloadPolicy``
internally and disables verl's separate param/optimizer offload paths to avoid
double offloading.

Context parallelism constraints
-------------------------------

FSDP-Turbo CP has one hard requirement: verl's own Ulysses sequence
parallelism (``ulysses_sequence_parallel_size > 1``) must **not** be
enabled simultaneously.  Use ``turbo_config.distributed.ulysses_parallel_size``
instead.

Violating this constraint raises ``ValueError`` at engine initialization.

All bundled scripts set ``actor_rollout_ref.model.use_remove_padding=True``
for correct packed-sequence handling, but it is no longer enforced as a
Turbo-specific CP requirement.

Gradient clipping with mixed meshes
-----------------------------------

When EP is enabled, dense and expert parameters may reside on different
``DeviceMesh`` objects.  PyTorch's stock ``clip_grad_norm_`` stacks per-gradient
DTensor scalars and fails when the meshes disagree.

FSDP-Turbo patches ``torch.nn.utils.clip_grad._get_total_norm`` and
``_clip_grads_with_norm_`` (see
``verl/workers/engine/fsdp/utils.py:apply_clip_grad_norm_patch``) to:

* Group norm scalars by ``(mesh, placement, device, dtype)`` before
  materialization, reducing the number of mesh-wide collectives from one per
  parameter to one per compatible group.
* Materialize each group's scalar via ``full_tensor()`` so that private
  ``_NormPartial`` placements complete their mesh reduction.
* Scale local gradient shards directly using a Python-float clip coefficient,
  avoiding ``foreach`` operations across incompatible meshes.

The patch is applied automatically when ``expert_parallel_size > 1`` and is
idempotent.  It delegates to the original PyTorch implementation when no
DTensor gradients are present, so non-Turbo code paths are unaffected.

Example scripts
---------------

Two GRPO example scripts are provided:

.. code-block:: bash

   # Qwen3.5-27B (dense)
   bash examples/grpo_trainer/run_qwen3_5_27b_fsdp_turbo.sh

   # Qwen3.5-35B-A3B (MoE with EP)
   bash examples/grpo_trainer/run_qwen3_5_35b_fsdp_turbo.sh

Both scripts auto-detect NPU vs. GPU, configure HCCL environment variables on
NPU, and pass the full Turbo plan via Hydra command-line overrides.  On NPU
they default to ``fully_shard_parallel_size=16`` and (for the MoE script)
``ep_plan.dispatcher=fused``; on GPU they default to ``8`` and ``eager``
respectively.  The CP patches described above are appended only when
``SP_SIZE > 1``.

For CI, the following smoke-test scripts are available:

.. code-block:: bash

   # NPU CI smoke test for Qwen3.5-2B (dense)
   bash tests/special_npu/run_qwen3_5_2b_fsdp_turbo.sh

   # NPU nightly CI smoke test for Qwen3.5-2B (dense)
   bash tests/special_npu/nightly_ci_ascend/run_grpo_qwen3_5_2b_fsdp_turbo_npu.sh

Both scripts target Qwen3.5-2B (``FSDP_SIZE=4``, ``SP_SIZE=2``) on a single
8-NPU node with context parallelism enabled by default.  The CI smoke test
runs for 1 training step; the nightly script runs for 5.  Both hardcode the
Turbo plan rather than auto-detecting the device.

Step5 actor MFU
---------------

``verl/utils/step5_flops.py`` estimates useful matrix/convolution FLOPs rather
than counting every executed kernel. A multiply-add counts as two FLOPs.
For an actor mini-batch, let ``T`` be the sum of valid prompt and response
lengths, ``N`` the number of actor devices and ``t`` the maximum training time
across actor ranks. The reported fraction is:

.. math::

   \mathrm{MFU} = \frac{F_{\mathrm{LLM}} + F_{\mathrm{indexer}} + F_{\mathrm{vision}}}
   {t\,N\,P_{\mathrm{peak}}}.

The timer covers ``engine.train_batch`` and device synchronization. It excludes
the surrounding parameter/optimizer load and offload, rollout generation,
old-logprob recomputation and actor-to-rollout weight synchronization. Therefore
``t`` differs from the outer ``timing_s/update_actor`` and whole-step timers.
The current device table uses ``432e12`` FLOP/s per ``Ascend950DT``; this is a
configured peak reference, not a measured device throughput.

Text model
~~~~~~~~~~

For text hidden size ``H``, vocabulary ``V``, query/KV head counts ``nq``/``nkv``
and head dimension ``d``, the active linear-weight element count is:

.. math::

   P_{\mathrm{linear}} = HV + \sum_l\left[
     H(2n_q+2n_{kv})d + P_{\mathrm{gate},l} + P_{\mathrm{FFN},l}\right].

The optional head-wise attention gate contributes ``H*nq``.
A dense FFN contributes ``3*H*I_dense``. A MoE FFN contributes
``H*E + 3*H*(k*I_expert + I_shared)``, where ``E`` is the router's expert count
and ``k`` is its per-token top-k. Thus all router columns are counted, but only
activated expert matrices are counted. The LM head is counted once; the
embedding lookup is not a dense matrix multiply. MTP layers not executed by
the actor are excluded.

.. math::

   F_{\mathrm{LLM}} = 6T P_{\mathrm{linear}}
      + 12n_qd\sum_l\sum_i A_l(L_i).

The factors 6 and 12 include the forward pass and both matrix gradients.
``A_l(L)`` counts useful query/key pairs. Full causal attention uses
``L*(L+1)/2``; sliding attention uses ``sum(min(q+1, window))``. For DSA region
size ``b`` and selected-region limit ``K``:

.. math::

   A_{\mathrm{DSA}}(L) = \sum_{q=0}^{L-1}
       \left[b\min(\lfloor q/b\rfloor,K)+(q\bmod b)+1\right].

DSA indexer and auxiliary loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current PPO loss consumes logits rather than ``aux_loss``. Indexer
projections and region scoring therefore contribute forward FLOPs only; the
discrete selection does not backpropagate into the indexer. Let ``a`` be the
indexer head count and ``p`` its proxy dimension. For each sparse layer:

.. math::

   F_{\mathrm{projection}} = 2TH[(a+2)p+a], \qquad
   R(L)=\sum_{q=0}^{L-1}\lfloor q/b\rfloor.

When ``enable_dsa_aux_loss=false``, teacher QK contributes zero. With the
current ``use_gbsa_full_sequence=true`` path, all-selected prefix indices are
constructed directly, so only the genuinely sparse suffix needs scoring:

.. math::

   F_{\mathrm{score}} = 2ap\sum_i
      [R(L_i)-R(\min(L_i,(K+1)b))].

For ``b=8, K=512``, lengths up to 4104 have no region-score FLOPs. Indexer
linear projections still execute over the whole sequence and remain counted.
The generic eager path still scores the prefix and uses ``2ap*sum(R(L_i))``.
With auxiliary loss enabled, the estimate retains full region scoring and
adds forward-only teacher QK FLOPs ``2*nq*d*b*sum(R(L_i))``. This is not an
estimator for a future training objective that actually backpropagates the
indexer auxiliary loss.

Frozen vision modules
~~~~~~~~~~~~~~~~~~~~~

Freeze flags come from the effective model config after overrides:
``freeze_vision_encoder`` and ``freeze_vision_projector`` default to false.
Both spatial downsampling convolutions belong to ``vision_model``. Only the
last linear layer belongs to ``vit_large_projector``.

Let ``e=1`` when the encoder is trainable, otherwise 0; define ``p`` similarly
for the projector. Each component's forward FLOPs are multiplied by:

* Patch projection: ``1+e`` (pixels do not need an input gradient).
* ViT blocks and both downsamplers: ``1+2e``.
* Final projector: ``1+e+p`` (input gradient depends on the encoder, weight
  gradient on the projector).

Consequently, freezing both modules counts their forward passes once, with no
vision backward FLOPs. Freezing only the projector still counts its input
gradient when the encoder is trainable. Real overview/crop shapes from each
PPO mini-batch determine token counts; precomputed image embeddings and dummy
vision calls contribute no vision FLOPs.

This remains an analytical MFU: recomputation, padding, optimizer work,
normalization, activation functions, routing/top-k and communication are not
added to the FLOPs numerator. Their runtime can still lower MFU. It is not
hardware FLOP utilization or whole-RL-step utilization.

Source of truth
---------------

* ``verl/workers/config/engine.py``: ``FSDPEngineConfig`` with the
  ``turbo_config`` field and ``strategy`` validation.
* ``verl/workers/engine/fsdp/fsdp_turbo_impl.py``:
  ``FSDPTurboEngineWithLMHead`` — mesh initialization, module building, and
  CP validation.
* ``verl/workers/engine/fsdp/utils.py``:
  ``apply_clip_grad_norm_patch`` — mixed-mesh gradient clipping.
* ``verl/trainer/config/engine/fsdp.yaml``: default Turbo config schema.
* ``examples/grpo_trainer/run_qwen3_5_*_fsdp_turbo.sh``: runnable examples.
* ``tests/special_npu/run_qwen3_5_2b_fsdp_turbo.sh``: NPU e2e CI smoke test.
* ``tests/special_npu/nightly_ci_ascend/run_grpo_qwen3_5_2b_fsdp_turbo_npu.sh``:
  NPU nightly CI smoke test.
* ``.github/workflows/e2e_ascend.yml``: NPU e2e CI workflow (turbo steps under
  ``vlm_rl_job``, 8x Ascend 910B, clones FSDPTurbo to ``/FSDPTurbo`` and
  exports ``PYTHONPATH``).
* ``.github/workflows/nightly_ascend.yml``: NPU nightly CI workflow (turbo job
  ``nightlyCI_grpo_qwen3_5_2b_fsdp_turbo_vllm_ascend``, scheduled daily).
