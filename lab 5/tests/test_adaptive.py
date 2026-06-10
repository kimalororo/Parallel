from gateway.stats import AdaptiveLoadBalancer, endpoint_key


def test_endpoint_key_uses_host_and_path() -> None:
    assert endpoint_key("http://api.example.com:8080/data?id=1") == "api.example.com:8080/data"


def test_fast_successes_increase_concurrency() -> None:
    balancer = AdaptiveLoadBalancer(
        window_size=10,
        min_samples=3,
        fast_threshold_ms=500,
        slow_threshold_ms=2000,
        absolute_limit=6,
    )
    url = "http://fast.example.com/data"

    for _ in range(6):
        balancer.record(url, True, 120, initial_concurrency=2, hard_limit=5)

    stats = balancer.snapshot_for_urls([url])[endpoint_key(url)]
    assert stats["success_rate"] == 1.0
    assert stats["adjusted_concurrency"] == 5


def test_errors_reduce_concurrency() -> None:
    balancer = AdaptiveLoadBalancer(
        window_size=10,
        min_samples=3,
        error_threshold=0.30,
        absolute_limit=6,
    )
    url = "http://flaky.example.com/data"

    for _ in range(5):
        balancer.record(url, False, 180, status=503, initial_concurrency=3, hard_limit=6)

    stats = balancer.snapshot_for_urls([url])[endpoint_key(url)]
    assert stats["success_rate"] == 0.0
    assert stats["adjusted_concurrency"] == 1


def test_slow_responses_reduce_concurrency_even_without_errors() -> None:
    balancer = AdaptiveLoadBalancer(
        window_size=10,
        min_samples=3,
        slow_threshold_ms=2000,
        absolute_limit=6,
    )
    url = "http://slow.example.com/data"

    for _ in range(4):
        balancer.record(url, True, 2500, initial_concurrency=4, hard_limit=6)

    stats = balancer.snapshot_for_urls([url])[endpoint_key(url)]
    assert stats["success_rate"] == 1.0
    assert stats["adjusted_concurrency"] == 2

