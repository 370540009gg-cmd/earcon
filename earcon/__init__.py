"""Earcon: a self-evolving memory proxy for OpenAI-compatible clients.

Any LLM application gets experience-driven learning by changing one
baseURL. Model weights stay frozen; experience is accumulated,
adjudicated, and injected as context. See docs/how-it-works.md.

Based on the mechanisms of JitRL (arXiv:2601.18510).
"""

__version__ = "0.1.0"
