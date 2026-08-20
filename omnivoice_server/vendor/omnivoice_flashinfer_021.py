"""OmniVoice 0.2.1 FlashInfer integration.

The packed decoder implementation is retained in the legacy-named module to
avoid duplicating a large CUDA patch.  It now imports OmniVoice 0.2.1 helpers
and validates the upstream layout before any model mutation.
"""

from .omnivoice_flashinfer_012 import apply_flashinfer

__all__ = ["apply_flashinfer"]
