"""Adapter around Sage's shared token store, with a standalone-package fallback.

In the full Sage repository, `utils.text_token_embedding_store` is the canonical
implementation merged with the token-level extraction change. The fallback exists
so this research-framework package can be unit-tested or reused standalone.
"""
try:
    from utils.text_token_embedding_store import TokenEmbeddingStore as TokenEmbeddingStore
except ImportError:  # standalone checkout/zip without Sage's existing utils package
    from .token_embedding_store_fallback import TokenEmbeddingStore as TokenEmbeddingStore

__all__ = ["TokenEmbeddingStore"]
