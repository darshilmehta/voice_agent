"""Metadata database: SQLAlchemy models, engine helpers and Alembic migrations (docs/DESIGN.md §3.9, §5).

The same models run on SQLite (local) and Postgres (cloud); only the URL changes. Business logic reaches the
database through the ``MetadataDB`` provider interface, never by creating engines itself.
"""
