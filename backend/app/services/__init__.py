"""Business logic. Services depend on provider interfaces (``MetadataDB``), never on concrete providers
(docs/DESIGN.md §6.3), and return domain objects (``app.domain``), never ORM rows."""
