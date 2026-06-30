"""Tests for API utilities (rate limiting, retry logic)."""

from __future__ import annotations

import didactic.api as dx
import pytest
from pytest_mock import MockerFixture

from bead.items.adapters.api_utils import (
    RateLimiter,
    rate_limit,
    retry_with_backoff,
)


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_successful_call_no_retry(self, mocker: MockerFixture) -> None:
        """Test that successful call requires no retry."""
        mock_func = mocker.Mock(return_value="success")

        @retry_with_backoff(max_retries=3)
        def test_func() -> str:
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_on_exception(self, mocker: MockerFixture) -> None:
        """Test retry behavior when exceptions occur."""
        mock_func = mocker.Mock(
            side_effect=[
                ValueError("error 1"),
                ValueError("error 2"),
                "success",
            ]
        )

        @retry_with_backoff(
            max_retries=3,
            initial_delay=0.01,
            backoff_factor=1.5,
            exceptions=(ValueError,),
        )
        def test_func() -> str:
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_exhausted(self, mocker: MockerFixture) -> None:
        """Test that exception is raised when retries are exhausted."""
        mock_func = mocker.Mock(side_effect=ValueError("persistent error"))

        @retry_with_backoff(max_retries=2, initial_delay=0.01, exceptions=(ValueError,))
        def test_func() -> None:
            mock_func()

        with pytest.raises((ValueError, dx.ValidationError), match="persistent error"):
            test_func()

        # Should try 3 times (initial + 2 retries)
        assert mock_func.call_count == 3

    def test_exponential_backoff(self, mocker: MockerFixture) -> None:
        """Test that delays follow exponential backoff pattern."""
        sleep_mock = mocker.patch("time.sleep")
        mock_func = mocker.Mock(
            side_effect=[
                ValueError("error 1"),
                ValueError("error 2"),
                "success",
            ]
        )

        @retry_with_backoff(
            max_retries=3,
            initial_delay=1.0,
            backoff_factor=2.0,
            exceptions=(ValueError,),
        )
        def test_func() -> str:
            return mock_func()

        result = test_func()

        assert result == "success"
        # Should sleep with exponential backoff: 1.0, 2.0
        assert sleep_mock.call_count == 2
        sleep_mock.assert_any_call(1.0)  # First retry
        sleep_mock.assert_any_call(2.0)  # Second retry

    def test_specific_exception_only(self, mocker: MockerFixture) -> None:
        """Test that only specified exceptions are retried."""
        mock_func = mocker.Mock(side_effect=RuntimeError("unexpected"))

        @retry_with_backoff(max_retries=3, initial_delay=0.01, exceptions=(ValueError,))
        def test_func() -> None:
            mock_func()

        # RuntimeError should not be retried
        with pytest.raises(RuntimeError, match="unexpected"):
            test_func()

        # Should only try once (no retries)
        assert mock_func.call_count == 1

    def test_multiple_exception_types(self, mocker: MockerFixture) -> None:
        """Test retry with multiple exception types."""
        mock_func = mocker.Mock(
            side_effect=[
                ValueError("error 1"),
                RuntimeError("error 2"),
                "success",
            ]
        )

        @retry_with_backoff(
            max_retries=3,
            initial_delay=0.01,
            exceptions=(ValueError, RuntimeError),
        )
        def test_func() -> str:
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 3


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_initialization(self) -> None:
        """Test rate limiter initialization."""
        limiter = RateLimiter(calls_per_minute=60)
        assert limiter.calls_per_minute == 60
        assert limiter.call_times == []

    def test_no_wait_under_limit(self, mocker: MockerFixture) -> None:
        """Test that no wait occurs when under rate limit."""
        sleep_mock = mocker.patch("time.sleep")
        limiter = RateLimiter(calls_per_minute=60)

        # Make 10 calls (well under limit)
        for _ in range(10):
            limiter.wait_if_needed()

        # Should not sleep
        sleep_mock.assert_not_called()

    def test_wait_at_limit(self, mocker: MockerFixture) -> None:
        """Test that wait occurs when rate limit is reached."""
        # Mock time to control progression
        mock_time = mocker.patch("time.time")
        sleep_mock = mocker.patch("time.sleep")

        # Start at time 0
        current_time = 0.0
        mock_time.return_value = current_time

        limiter = RateLimiter(calls_per_minute=3)  # Low limit for testing

        # Make 3 calls (at limit)
        for i in range(3):
            mock_time.return_value = current_time + i * 0.1
            limiter.wait_if_needed()

        # Should not have slept yet
        sleep_mock.assert_not_called()

        # 4th call should trigger wait
        mock_time.return_value = current_time + 0.5  # Still within 1 minute
        limiter.wait_if_needed()

        # Should have slept
        assert sleep_mock.call_count >= 1

    def test_old_calls_expire(self, mocker: MockerFixture) -> None:
        """Test that old calls (>60s) are removed from tracking."""
        mock_time = mocker.patch("time.time")
        sleep_mock = mocker.patch("time.sleep")

        limiter = RateLimiter(calls_per_minute=3)

        # Make 3 calls at time 0
        mock_time.return_value = 0.0
        for _ in range(3):
            limiter.wait_if_needed()

        assert len(limiter.call_times) == 3

        # Make another call at time 61 (after 60-second window)
        mock_time.return_value = 61.0
        limiter.wait_if_needed()

        # Old calls should be expired, no sleep needed
        sleep_mock.assert_not_called()
        # Should have 1 call in tracking (the new one)
        assert len(limiter.call_times) == 1


class TestRateLimitDecorator:
    """Tests for rate_limit decorator."""

    def test_decorator_basic(self, mocker: MockerFixture) -> None:
        """Test basic rate limit decorator functionality."""
        mock_func = mocker.Mock(return_value="result")
        sleep_mock = mocker.patch("time.sleep")

        @rate_limit(calls_per_minute=60)
        def test_func() -> str:
            return mock_func()

        # Make a few calls (under limit)
        for _ in range(5):
            result = test_func()
            assert result == "result"

        assert mock_func.call_count == 5
        # Should not sleep if under limit
        sleep_mock.assert_not_called()

    def test_decorator_preserves_function_metadata(self) -> None:
        """Test that decorator preserves function metadata."""

        @rate_limit(calls_per_minute=60)
        def test_func() -> str:
            """Test docstring."""
            return "result"

        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test docstring."

    def test_decorator_with_arguments(self, mocker: MockerFixture) -> None:
        """Test decorator with function arguments."""
        mock_func = mocker.Mock(return_value="result")

        @rate_limit(calls_per_minute=60)
        def test_func(arg1: str, arg2: int) -> str:
            return mock_func(arg1, arg2)

        result = test_func("test", 42)

        assert result == "result"
        mock_func.assert_called_once_with("test", 42)


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_retry_and_rate_limit_together(self, mocker: MockerFixture) -> None:
        """Test using both retry and rate limit decorators together."""
        mock_func = mocker.Mock(side_effect=[ValueError("error"), "success"])
        sleep_mock = mocker.patch("time.sleep")

        @retry_with_backoff(max_retries=3, initial_delay=0.01, exceptions=(ValueError,))
        @rate_limit(calls_per_minute=60)
        def test_func() -> str:
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 2
        # Should sleep from retry (not rate limit in this case)
        assert sleep_mock.call_count >= 1
