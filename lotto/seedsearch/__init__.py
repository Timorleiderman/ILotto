"""A side quest: recovering the seed of an explicitly artificial generator.

The real lottery uses physical ball machines, so nothing here is evidence
about it. The question is an engineering one: if draws *were* produced by a
seeded hash, how fast could the seed be found? Start with ``generator.py``;
the Rust and CUDA engines are optional accelerators of the same search.
"""
