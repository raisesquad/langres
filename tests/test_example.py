"""Example test to verify test infrastructure works."""

from langres import hello


def test_example() -> None:
    """Test that hello function returns expected string."""
    result = hello()
    assert result == "Hello from langres!"
    assert isinstance(result, str)
