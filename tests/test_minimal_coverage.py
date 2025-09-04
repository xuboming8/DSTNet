"""
Minimal test with some basic coverage to test the infrastructure.
"""
import pytest


def simple_function(a, b):
    """A simple function for testing coverage."""
    if a > b:
        return a
    else:
        return b


def another_function(x):
    """Another simple function."""
    return x * 2


class TestMinimalCoverage:
    """Test class for minimal coverage."""
    
    def test_simple_function(self):
        """Test the simple function."""
        assert simple_function(5, 3) == 5
        assert simple_function(2, 7) == 7
    
    def test_another_function(self):
        """Test another function."""
        assert another_function(4) == 8
        assert another_function(0) == 0
        
    @pytest.mark.unit
    def test_with_unit_marker(self):
        """Test with unit marker."""
        assert True
        
    @pytest.mark.integration
    def test_with_integration_marker(self):
        """Test with integration marker."""
        result = simple_function(10, 5)
        processed = another_function(result)
        assert processed == 20