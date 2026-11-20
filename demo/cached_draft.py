"""Inference-only Draft state reuse; the sealed Task64 reference is unchanged.

Import alongside latency_smoke.py. Only dense, full-attention Qwen3 and batch=1
are supported. This is output-side repair: applying it must not modify layer KV.
Training, Target verification, the wire protocol and position-dependent repair
are deliberately delegated to the reference implementation.
"""
from collections import Counter
from contextlib import contextmanager

import torch
from transformers import DynamicCache

from latency_smoke import NativeDraft, require
from feedback_codec.fixed_probe import ProtocolView
from h_vq64_head import MessageState


class _PreparedBoundary:
    """Narrow adapter: no arbitrary prefix or continuation can use this hidden."""
    def __init__(self, backend, prefix, hidden):
        self.backend, self.prefix, self.hidden = backend, tuple(prefix), hidden

    def __getattr__(self, key):
        return getattr(self.backend, key)

    def teacher_hidden(self, prefix, continuation):
        require(tuple(prefix) == self.prefix and continuation == [0],
                "prepared hidden used at a different causal boundary")
        return self.hidden.detach().clone()


class CachedDraft(NativeDraft):
    def __init__(self, backend, head, *, mode="kv", cache_control=True):
        super().__init__(backend, head)
        require(mode in ("boundary", "kv"), "unknown Draft cache mode")
        cfg = backend.model.config
        require(cfg.model_type == "qwen3" and not getattr(cfg, "use_sliding_window", False)
                and all(x == "full_attention" for x in getattr(cfg, "layer_types", [])),
                "only dense full-attention Qwen3 supports this cache contract")
        require(not backend.model.training and not head.training,
                "cache is inference-only; training must use the original head")
        require(not any(p.requires_grad for p in backend.model.parameters())
                and not any(p.requires_grad for p in head.parameters()), "weights must be frozen")
        self.mode, self.cache_control = mode, cache_control

    def reset(self, meter):
        super().reset(meter)
        self.prefix = ()
        self.prompt = None
        self.committed = ()
        self.cache = None
        self.hidden_by_length = {}
        self.last_hidden = self.control = None
        self.stats = Counter()

    def snapshot(self):
        """Committed tokens and tokens actually consumed by the decoder differ."""
        return {"mode": self.mode, "committed_length": len(self.committed),
                "decoder_consumed_length": len(self.prefix),
                "kv_length": None if self.cache is None else self.cache.get_seq_length(),
                "pending_committed_tokens": len(self.committed) - len(self.prefix)
                if self.committed[:len(self.prefix)] == self.prefix else None,
                "cache_is_prefix_of_committed": self.committed[:len(self.prefix)] == self.prefix}

    def _check_cache_length(self, expected):
        if self.mode == "kv":
            if self.cache is None:
                require(expected == 0, "cache length missing for consumed prefix")
            else:
                layers = self.backend.model.config.num_hidden_layers
                require(len(self.cache.layers) == layers and
                        all(self.cache.get_seq_length(i) == expected for i in range(layers)),
                        "cache length does not match consumed prefix in every layer")

    @torch.inference_mode()
    def commit(self, confirmed, reply):
        """Discard rejected speculative KV even on terminal rounds, without a new forward."""
        wanted = tuple(confirmed)
        self._check_cache_length(len(self.prefix))
        require(self.prompt is not None, "commit before prompt")
        require(wanted[:len(self.committed)] == self.committed, "committed history changed")
        added = wanted[len(self.committed):]
        accepted, answer = reply["accepted"], reply["answer"]
        require(type(accepted) is int and 0 <= accepted <= 8 and added and
                all(type(t) is int and 0 <= t < self.backend.vocab_size for t in added),
                "invalid committed tokens")
        require(answer is None or (type(answer) is int and added[-1] == answer),
                "committed suffix differs from verifier answer")
        require(len(wanted) - len(self.committed)
                == reply["accepted"] + int(reply["answer"] is not None), "commit count differs")
        known = self.prefix[len(self.committed):len(self.committed) + accepted]
        require(added[:len(known)] == known, "accepted tokens differ from speculative history")
        common = 0
        for a, b in zip(self.prefix, wanted):
            if a != b:
                break
            common += 1
        if common < len(self.prefix):
            if self.mode == "kv":
                require(common >= len(self.prompt), "cannot crop committed prompt")
                with self._timed("draft_kv_crop"):
                    self.cache.crop(common)
                self._check_cache_length(common)
                self.stats["kv_crops"] += 1
                self.stats["discarded_speculative_tokens"] += len(self.prefix) - common
                self.hidden_by_length = {n: h for n, h in self.hidden_by_length.items() if n <= common}
                self.prefix, self.last_hidden = self.prefix[:common], self.hidden_by_length[common]
            else:
                self.prefix, self.last_hidden = (), None
        self.committed = wanted
        self.state = self.control = None
        self.stats["verification_commits"] += 1
        require(self.prefix == wanted[:len(self.prefix)], "rejected KV survived commit")

    @contextmanager
    def _timed(self, name):
        with self.meter.timed(name):
            yield

    @torch.inference_mode()
    def boundary(self, prefix):
        """Return post-norm hidden after processing exactly prefix, never a future token."""
        require(isinstance(prefix, list) and prefix, "nonempty token list required")
        require(all(type(t) is int and 0 <= t < self.backend.vocab_size for t in prefix),
                "token outside Draft vocabulary")
        maximum = getattr(self.backend.model.config, "max_position_embeddings", None)
        require(maximum is None or len(prefix) <= maximum, "context limit exceeded")
        wanted = tuple(prefix)
        self._check_cache_length(len(self.prefix))
        if self.prompt is None:
            self.prompt = wanted
            self.committed = wanted
        require(wanted[:len(self.prompt)] == self.prompt, "prompt changed without reset")
        require(wanted[:len(self.committed)] == self.committed, "draft left committed history")
        if wanted == self.prefix:
            self.stats["boundary_hidden_hits"] += 1
            return self.last_hidden

        if self.mode == "boundary":
            # Isolates boundary/control reuse while retaining a full-prefix decoder.
            with self._timed("draft_decoder_forward"):
                _, final = self.backend._forward_with_final_hidden(
                    self.backend._ids(prefix), decoder_only=True)
            self.stats["decoder_input_tokens"] += len(prefix)
            self.stats["decoder_calls"] += 1
            self.prefix, self.last_hidden = wanted, final[:, -1].detach().clone()
            return self.last_hidden

        common = 0
        for a, b in zip(self.prefix, wanted):
            if a != b:
                break
            common += 1
        if self.cache is None:
            self.cache = DynamicCache(config=self.backend.model.config)
        elif common < len(self.prefix):
            require(common >= len(self.prompt), "cannot discard the prompt cache")
            with self._timed("draft_kv_crop"):
                self.cache.crop(common)
            self.stats["kv_crops"] += 1
            self.stats["discarded_speculative_tokens"] += len(self.prefix) - common
            self.hidden_by_length = {n: h for n, h in self.hidden_by_length.items() if n <= common}
        if common < len(self.prefix):
            self._check_cache_length(common)
        suffix = prefix[common:]
        if suffix:
            with self._timed("draft_decoder_forward"):
                ids = self.backend._ids(suffix)
                positions = torch.arange(common, len(prefix), device=ids.device)
                result = self.backend.model.model(
                    input_ids=ids, past_key_values=self.cache, use_cache=True,
                    attention_mask=torch.ones((1, len(prefix)), dtype=torch.long, device=ids.device),
                    cache_position=positions, position_ids=positions.unsqueeze(0),
                    output_hidden_states=False, return_dict=True)
                self.cache = result.past_key_values
                final = result.last_hidden_state
            self.stats["decoder_calls"] += 1
            self.stats["decoder_input_tokens"] += len(suffix)
            if common == 0:
                self.stats["prompt_prefills"] += 1
                self.hidden_by_length[len(prefix)] = final[:, -1].detach().clone()
            else:
                for j in range(len(suffix)):
                    self.hidden_by_length[common + j + 1] = final[:, j].detach().clone()
            self._check_cache_length(len(prefix))
        require(len(prefix) in self.hidden_by_length, "missing saved boundary hidden")
        self.prefix, self.last_hidden = wanted, self.hidden_by_length[len(prefix)]
        return self.last_hidden

    @torch.inference_mode()
    def build_control(self, state):
        """Exactly the eval branch of HTaskHead.forward_logits, once per round."""
        require(not self.head.training and type(state) is MessageState and state.base.mode == "C",
                "control cache requires a frozen eval MessageState")
        base, head = state.base, self.head
        local = base.local_hidden
        require(local.shape == (1, head.config.draft_hidden_size), "local boundary shape differs")
        codes = head.base.core.local_encode(local, base.rejected_embedding,
                                            base.correction_embedding, base.accepted_count)
        control = (head.base.core.repair._lookup(codes[:, 0], 1)
                   + head.base.core.repair._lookup(codes[:, 1], 1))
        surrogate = head.base.local_hidden_lift(local.to(
            device=head.base.local_hidden_lift[0].weight.device,
            dtype=head.base.local_hidden_lift[0].weight.dtype))
        control = control + head.base.hidden_projection(surrogate)
        semantic = head.base.local_top32_surrogate(local.to(
            device=head.base.local_top32_surrogate.weight.device,
            dtype=head.base.local_top32_surrogate.weight.dtype))
        control = control + head.base.top32_projection(semantic)
        message_control = head.message.decode_payload(state.payload)
        require(message_control.shape == (1, head.config.groups * head.config.code_dim),
                "decoded message shape differs")
        self.stats["control_builds"] += 1
        self.stats["payload_decodes"] += 1
        return (control + message_control.to(device=control.device, dtype=control.dtype)).detach()

    @torch.inference_mode()
    def repaired_logits(self, hidden, position):
        require(self.state is not None, "no round feedback")
        with self._timed("edge_head_including_projection"):
            if self.cache_control:
                logits = self.head.base.core.repair.repair_from_control(hidden, self.control, position)
            else:
                logits = self.head.forward_logits(hidden, self.state, position)
                self.stats["payload_decodes"] += 1
        self.stats["repair_projections"] += 1
        require(bool(torch.isfinite(logits).all()), "nonfinite repaired logits")
        return logits

    @torch.inference_mode()
    def token(self, prefix, position, arm):
        require(arm in ("raw", "shadow", "active") and type(position) is int and 1 <= position <= 8,
                "invalid arm or within-round position")
        hidden = self.boundary(prefix)
        if self.state is not None:
            require(tuple(prefix[:len(self.state.corrected_prefix)]) == self.state.corrected_prefix,
                    "feedback belongs to a different prefix")
        # ACTIVE needs only the repaired projection; SHADOW computes both by design.
        if self.state is None or arm != "active":
            with self._timed("draft_native_projection"):
                logits = self.backend.model.get_output_embeddings()(hidden)[0]
            self.stats["native_projections"] += 1
        if self.state is not None:
            repaired = self.repaired_logits(hidden, position)[0]
            if arm == "active":
                logits = repaired
        with self._timed("draft_decision"):
            return int(logits.argmax().item())

    @torch.inference_mode()
    def prepare(self, confirmed, reply, payload):
        require(type(payload) is bytes and len(payload) == 8, "expected eight feedback bytes")
        require(self.prompt is not None and tuple(confirmed) == self.committed,
                "feedback must use the committed boundary")
        with self._timed("edge_feedback_prepare"):
            # Includes the necessary correction/bonus suffix update, no full replay in KV mode.
            hidden = self.boundary(confirmed)
            view = ProtocolView(tuple(confirmed), 8, reply["accepted"], reply["rejected"], reply["answer"])
            base = self.head.base.prepare_view(view, _PreparedBoundary(self.backend, confirmed, hidden))
            state = MessageState(base, (), payload, tuple(confirmed))
            control = None
            if self.cache_control:
                with self._timed("edge_control_prepare_including_decode"):
                    control = self.build_control(state)
            self.state, self.control = state, control
        self.stats["feedback_prepares"] += 1

