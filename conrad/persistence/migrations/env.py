"""Alembic environment. Migrations are explicit revisions; metadata is used only for drift checks."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from conrad.persistence.db import metadata

target_metadata = metadata


def run_migrations_online() -> None:
    url = context.config.get_main_option("sqlalchemy.url")
    assert url is not None
    engine = create_engine(url, future=True)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()
