"""Evaluation harness: golden question sets, retrieval and generation metrics,
the faithfulness check against human labels, and the CI regression gate.

core/evaluation.py stays as the legacy 10 question smoke test. Nothing here
writes real (MIMIC-IV-Note) derived text outside outputs/ or *.local.* files.
"""
