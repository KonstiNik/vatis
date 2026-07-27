"""Test utilities that are *writers* rather than assertions.

``bundle_fixture`` writes a checkpoint bundle in the frozen schema by running a
real (tiny) AdamW training loop, so the golden acceptance gate is self-contained
and does not depend on any bundle produced outside this repository.
``llama_factory`` exposes the project's Llama architecture as a bundle model
factory when it happens to be available on disk.
"""
