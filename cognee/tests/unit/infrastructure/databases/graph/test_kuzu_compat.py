"""Tests for legacy Kuzu compatibility shims."""


def test_kuzu_import_alias_points_to_ladybug():
    from cognee.infrastructure.databases.graph.kuzu.kuzu_migrate import kuzu_migration
    import kuzu
    import ladybug

    assert kuzu is ladybug
    assert kuzu.__version__ == ladybug.__version__
    assert callable(kuzu_migration)
