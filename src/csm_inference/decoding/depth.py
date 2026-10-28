"""Specialized frame-local CSM depth generation with reference-equivalent sampling."""

import copy

import torch
from transformers.generation.logits_process import TopPLogitsWarper


def direct_depth_generate(self, input_ids, backbone_last_hidden_state, **kwargs):
    """Same frame-local KV cache and sampling, without 31 generic generate steps.

    Index heads with a Python integer: tensor advanced-indexing copies a whole
    head matrix on every codebook step in the upstream implementation.
    """
    if kwargs:
        raise ValueError(f"Unsupported direct depth options: {sorted(kwargs)}")
    settings = copy.copy(self.generation_config)
    for name, value in settings._get_default_generation_params().items():
        if getattr(settings, name, None) is None:
            setattr(settings, name, value)
    if settings.repetition_penalty != 1 or settings.num_beams != 1:
        raise ValueError("Direct depth supports unpenalized single-beam decoding only")
    sequence = [input_ids]
    cache = None
    token = None
    top_p = TopPLogitsWarper(settings.top_p) if settings.top_p < 1 else None
    for book in range(self.config.num_codebooks - 1):
        if book == 0:
            embeddings = self.model.embed_tokens(input_ids)
            embeddings[:, 0] = backbone_last_hidden_state
        else:
            embeddings = self.model.embed_tokens(token + book * self.config.vocab_size)
        output = self.model(inputs_embeds=embeddings, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = torch.nn.functional.linear(
            output.last_hidden_state[:, -1], self.codebooks_head.weight[book].T
        ).float()
        if settings.do_sample:
            logits = logits / settings.temperature
            if settings.top_k:
                threshold = torch.topk(logits, min(settings.top_k, logits.shape[-1]))[0][:, -1:]
                logits = logits.masked_fill(logits < threshold, -float("inf"))
            if top_p is not None:
                logits = top_p(None, logits)
            token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        else:
            token = logits.argmax(dim=-1, keepdim=True)
        sequence.append(token)
    return torch.cat(sequence, dim=1)
