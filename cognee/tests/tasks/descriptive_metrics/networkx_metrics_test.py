from cognee.tests.tasks.descriptive_metrics.metrics_test_utils import assert_metrics
import asyncio


if __name__ == "__main__":
    asyncio.run(assert_metrics(provider="networkx", include_optional=False))
    asyncio.run(assert_metrics(provider="networkx", include_optional=True))
