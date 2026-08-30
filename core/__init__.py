"""Real-data pipeline: preprocessing, chunking, embedding, retrieval,
generation, and evaluation over MIMIC-IV-Note.

Modules here are imported lazily by app.py so the public Vercel function,
which installs only requirements.txt, never pulls in torch or faiss.
"""
