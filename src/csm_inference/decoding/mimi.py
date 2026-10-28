"""Request-local causal Mimi convolution history plus transformer KV state."""

import types

import torch
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConvTranspose1d


class StatefulMimi:
    """Temporarily adapt a *single owned* codec; always restore in close()."""

    def __init__(self, codec):
        self.codec = codec
        self.originals = []
        self.cache = None
        try:
            for module in codec.decoder.modules():
                if isinstance(module, MimiConv1d):
                    if not module.causal or module.conv.stride != (1,):
                        raise ValueError(
                            "Streaming decoder requires causal stride-one convolutions"
                        )
                    self._patch(module, self._conv(module))
                elif isinstance(module, MimiConvTranspose1d):
                    if not module.causal or module.padding_left != 0:
                        raise ValueError(
                            "Streaming decoder requires right-trimmed causal upsampling"
                        )
                    self._patch(module, self._transpose(module))
        except (ValueError, RuntimeError):
            self.close()
            raise

    def _patch(self, module, forward):
        self.originals.append((module, module.forward))
        module.forward = types.MethodType(forward, module)

    @staticmethod
    def _conv(module):
        previous = None
        original = module.forward
        padding = int(module.padding_total)

        def forward(self, values, padding_cache=None):
            nonlocal previous
            if padding_cache is not None:
                raise ValueError("External convolution padding cache is unsupported")
            if previous is None:
                result = original(values)
                history = self._pad1d(values, (padding, 0), mode=self.pad_mode)
            else:
                history = torch.cat((previous, values), dim=-1)
                result = self.conv(history)
            if padding:
                previous = history[..., -padding:].clone()
            return result

        return forward

    @staticmethod
    def _transpose(module):
        previous = None
        padding = module.padding_right

        def forward(self, values):
            nonlocal previous
            result = self.conv(values)
            if previous is not None:
                result[..., :padding] += previous
            end = result.shape[-1] - padding
            if padding:
                previous = result[..., end:].clone()
                # The next chunk's convolution supplies its own bias once.
                if self.conv.bias is not None:
                    previous -= self.conv.bias[None, :, None]
            return result[..., :end]

        return forward

    @torch.inference_mode()
    def decode(self, codes):
        output = self.codec.decode(
            codes.transpose(1, 2), decoder_past_key_values=self.cache, use_cache=True
        )
        self.cache = output.decoder_past_key_values
        return output.audio_values[0, 0].detach().float().cpu().numpy()

    def close(self):
        for module, forward in reversed(self.originals):
            module.forward = forward
        self.originals.clear()
        self.cache = None
