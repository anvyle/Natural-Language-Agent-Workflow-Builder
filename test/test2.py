import unittest
import pytest
from typing import List, Dict, Any
import math
import json


class Calculator:
    """Simple calculator class for testing purposes."""
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent."""
        return base ** exponent


class StringProcessor:
    """String processing utilities for testing."""
    
    def reverse(self, text: str) -> str:
        """Reverse a string."""
        return text[::-1]
    
    def count_vowels(self, text: str) -> int:
        """Count vowels in a string."""
        vowels = 'aeiouAEIOU'
        return sum(1 for char in text if char in vowels)
    
    def is_palindrome(self, text: str) -> bool:
        """Check if string is a palindrome."""
        cleaned = ''.join(char.lower() for char in text if char.isalnum())
        return cleaned == cleaned[::-1]
    
    def word_count(self, text: str) -> int:
        """Count words in a string."""
        return len(text.split())


# Unit Tests using unittest
class TestCalculator(unittest.TestCase):
    """Test cases for Calculator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.calc = Calculator()
    
    def test_add_positive_numbers(self):
        """Test addition of positive numbers."""
        result = self.calc.add(5, 3)
        self.assertEqual(result, 8)
    
    def test_add_negative_numbers(self):
        """Test addition of negative numbers."""
        result = self.calc.add(-5, -3)
        self.assertEqual(result, -8)
    
    def test_subtract_numbers(self):
        """Test subtraction."""
        result = self.calc.subtract(10, 4)
        self.assertEqual(result, 6)
    
    def test_multiply_numbers(self):
        """Test multiplication."""
        result = self.calc.multiply(6, 7)
        self.assertEqual(result, 42)
    
    def test_divide_numbers(self):
        """Test division."""
        result = self.calc.divide(15, 3)
        self.assertEqual(result, 5)
    
    def test_divide_by_zero(self):
        """Test division by zero raises ValueError."""
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)
    
    def test_power_positive_exponent(self):
        """Test power with positive exponent."""
        result = self.calc.power(2, 3)
        self.assertEqual(result, 8)
    
    def test_power_zero_exponent(self):
        """Test power with zero exponent."""
        result = self.calc.power(5, 0)
        self.assertEqual(result, 1)


class TestStringProcessor(unittest.TestCase):
    """Test cases for StringProcessor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = StringProcessor()
    
    def test_reverse_string(self):
        """Test string reversal."""
        result = self.processor.reverse("hello")
        self.assertEqual(result, "olleh")
    
    def test_reverse_empty_string(self):
        """Test reversing empty string."""
        result = self.processor.reverse("")
        self.assertEqual(result, "")
    
    def test_count_vowels(self):
        """Test vowel counting."""
        result = self.processor.count_vowels("hello world")
        self.assertEqual(result, 3)
    
    def test_count_vowels_no_vowels(self):
        """Test vowel counting with no vowels."""
        result = self.processor.count_vowels("rhythm")
        self.assertEqual(result, 0)
    
    def test_is_palindrome_true(self):
        """Test palindrome detection with true case."""
        result = self.processor.is_palindrome("A man a plan a canal Panama")
        self.assertTrue(result)
    
    def test_is_palindrome_false(self):
        """Test palindrome detection with false case."""
        result = self.processor.is_palindrome("hello world")
        self.assertFalse(result)
    
    def test_word_count(self):
        """Test word counting."""
        result = self.processor.word_count("hello world test")
        self.assertEqual(result, 3)
    
    def test_word_count_empty_string(self):
        """Test word counting with empty string."""
        result = self.processor.word_count("")
        self.assertEqual(result, 0)


# Pytest style tests
class TestPytestStyle:
    """Pytest style test class."""
    
    @pytest.fixture
    def calculator(self):
        """Fixture to provide calculator instance."""
        return Calculator()
    
    @pytest.fixture
    def string_processor(self):
        """Fixture to provide string processor instance."""
        return StringProcessor()
    
    def test_calculator_fixture(self, calculator):
        """Test using calculator fixture."""
        assert calculator.add(2, 3) == 5
        assert calculator.multiply(4, 5) == 20
    
    @pytest.mark.parametrize("a,b,expected", [
        (1, 2, 3),
        (0, 0, 0),
        (-1, 1, 0),
        (10, -5, 5)
    ])
    def test_add_parametrized(self, calculator, a, b, expected):
        """Parametrized test for addition."""
        assert calculator.add(a, b) == expected
    
    @pytest.mark.parametrize("text,expected", [
        ("hello", 2),
        ("world", 1),
        ("aeiou", 5),
        ("rhythm", 0),
        ("", 0)
    ])
    def test_count_vowels_parametrized(self, string_processor, text, expected):
        """Parametrized test for vowel counting."""
        assert string_processor.count_vowels(text) == expected
    
    def test_divide_by_zero_pytest(self, calculator):
        """Test division by zero using pytest."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calculator.divide(10, 0)


# Integration test example
class TestIntegration:
    """Integration test examples."""
    
    def test_calculator_and_string_processor_integration(self):
        """Test integration between calculator and string processor."""
        calc = Calculator()
        processor = StringProcessor()
        
        # Use calculator result in string processing
        result = calc.add(5, 3)
        result_str = str(result)
        reversed_result = processor.reverse(result_str)
        
        assert result == 8
        assert result_str == "8"
        assert reversed_result == "8"
    
    def test_complex_calculation_workflow(self):
        """Test a more complex calculation workflow."""
        calc = Calculator()
        
        # Multi-step calculation
        step1 = calc.add(10, 5)  # 15
        step2 = calc.multiply(step1, 2)  # 30
        step3 = calc.subtract(step2, 10)  # 20
        step4 = calc.divide(step3, 4)  # 5
        
        assert step4 == 5


# Performance test example
class TestPerformance:
    """Performance test examples."""
    
    def test_large_number_calculation(self):
        """Test performance with large numbers."""
        calc = Calculator()
        
        # Test with large numbers
        result = calc.multiply(999999, 999999)
        expected = 999999 * 999999
        assert result == expected
    
    def test_string_processing_performance(self):
        """Test string processing with large text."""
        processor = StringProcessor()
        
        # Create a large text
        large_text = "hello world " * 1000
        
        # Test performance
        word_count = processor.word_count(large_text)
        vowel_count = processor.count_vowels(large_text)
        
        assert word_count == 2000  # 1000 * 2 words
        assert vowel_count > 0


# Mock test example
class TestWithMocks:
    """Test examples using mocks."""
    
    def test_mock_external_dependency(self, monkeypatch):
        """Test with mocked external dependency."""
        import random
        
        def mock_random():
            return 0.5
        
        monkeypatch.setattr(random, 'random', mock_random)
        
        # Now random.random() will always return 0.5
        import random as random_module
        assert random_module.random() == 0.5


# Data-driven test example
class TestDataDriven:
    """Data-driven test examples."""
    
    @pytest.fixture
    def test_data(self):
        """Provide test data."""
        return [
            {"input": "hello", "expected_reverse": "olleh", "expected_vowels": 2},
            {"input": "world", "expected_reverse": "dlrow", "expected_vowels": 1},
            {"input": "python", "expected_reverse": "nohtyp", "expected_vowels": 1},
            {"input": "testing", "expected_reverse": "gnitset", "expected_vowels": 2},
        ]
    
    def test_string_processing_with_data(self, string_processor, test_data):
        """Test string processing with various data sets."""
        for data in test_data:
            input_text = data["input"]
            expected_reverse = data["expected_reverse"]
            expected_vowels = data["expected_vowels"]
            
            assert string_processor.reverse(input_text) == expected_reverse
            assert string_processor.count_vowels(input_text) == expected_vowels


# Edge case tests
class TestEdgeCases:
    """Edge case test examples."""
    
    def test_calculator_edge_cases(self):
        """Test calculator with edge cases."""
        calc = Calculator()
        
        # Test with zero
        assert calc.add(0, 0) == 0
        assert calc.multiply(0, 5) == 0
        assert calc.power(0, 5) == 0
        
        # Test with very large numbers
        large_num = 1e15
        assert calc.add(large_num, 1) == large_num + 1
        
        # Test with very small numbers
        small_num = 1e-15
        assert calc.multiply(small_num, 2) == 2e-15
    
    def test_string_processor_edge_cases(self):
        """Test string processor with edge cases."""
        processor = StringProcessor()
        
        # Test with special characters
        assert processor.reverse("!@#$%") == "%$#@!"
        assert processor.count_vowels("!@#$%") == 0
        
        # Test with numbers
        assert processor.reverse("12345") == "54321"
        assert processor.count_vowels("12345") == 0
        
        # Test with whitespace
        assert processor.word_count("   hello   world   ") == 2


if __name__ == "__main__":
    # Run unittest tests
    unittest.main(verbosity=2)
