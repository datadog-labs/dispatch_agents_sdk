"""Unit tests for resource parsing functions."""

import pytest

from dispatch_agents.resources import _parse_cpu, _parse_memory


class TestParseCpu:
    """Tests for _parse_cpu function."""

    # Valid inputs - millicores format
    @pytest.mark.parametrize(
        "input_value,expected",
        {
            # Millicores format
            "250m": 256,
            "500m": 512,
            "1000m": 1024,
            "2000m": 2048,
            "4000m": 4096,
            "100m": 102,  # 100 * 1024 / 1000 = 102.4 -> 102
            "1m": 1,  # 1 * 1024 / 1000 = 1.024 -> 1
        }.items(),
    )
    def test_millicores_format(self, input_value: str, expected: int):
        """Should parse millicores format correctly."""
        assert _parse_cpu(input_value) == expected

    # Valid inputs - cores as string
    @pytest.mark.parametrize(
        "input_value,expected",
        {
            "0.25": 256,
            "0.5": 512,
            "1": 1024,
            "1.0": 1024,
            "2": 2048,
            "2.0": 2048,
            "4": 4096,
            "8": 8192,
            "16": 16384,
            "0.1": 102,  # 0.1 * 1024 = 102.4 -> 102
        }.items(),
    )
    def test_cores_as_string(self, input_value: str, expected: int):
        """Should parse cores as string correctly."""
        assert _parse_cpu(input_value) == expected

    # Valid inputs - numeric types (int, float)
    @pytest.mark.parametrize(
        "input_value,expected",
        {
            0.25: 256,
            0.5: 512,
            1: 1024,
            2: 2048,
            4: 4096,
        }.items(),
    )
    def test_numeric_types(self, input_value: int | float, expected: int):
        """Should parse numeric values (int, float) as cores."""
        assert _parse_cpu(input_value) == expected

    # Whitespace handling
    @pytest.mark.parametrize(
        "input_value,expected",
        {
            "  500m  ": 512,
            " 1 ": 1024,
            "\t2\n": 2048,
            "  0.5  ": 512,
        }.items(),
    )
    def test_whitespace_handling(self, input_value: str, expected: int):
        """Should handle leading/trailing whitespace."""
        assert _parse_cpu(input_value) == expected

    # Case insensitivity
    @pytest.mark.parametrize(
        "input_value,expected",
        {
            "500M": 512,
            "1000M": 1024,
            "250M": 256,
        }.items(),
    )
    def test_case_insensitivity(self, input_value: str, expected: int):
        """Should handle uppercase M for millicores."""
        assert _parse_cpu(input_value) == expected

    # Edge cases
    def test_zero_millicores(self):
        """Should handle zero millicores."""
        assert _parse_cpu("0m") == 0

    def test_zero_cores(self):
        """Should handle zero cores."""
        assert _parse_cpu("0") == 0
        assert _parse_cpu(0) == 0
        assert _parse_cpu(0.0) == 0

    # Malformed inputs
    def test_empty_string_raises(self):
        """Should raise ValueError for empty string."""
        with pytest.raises(ValueError):
            _parse_cpu("")

    def test_whitespace_only_raises(self):
        """Should raise ValueError for whitespace-only string."""
        with pytest.raises(ValueError):
            _parse_cpu("   ")

    def test_invalid_suffix_raises(self):
        """Should raise ValueError for invalid suffix."""
        with pytest.raises(ValueError):
            _parse_cpu("500x")

    def test_non_numeric_raises(self):
        """Should raise ValueError for non-numeric string."""
        with pytest.raises(ValueError):
            _parse_cpu("abc")

    def test_negative_millicores_parses(self):
        """Negative millicores parse but result in negative units."""
        # This is technically valid parsing, validation happens elsewhere
        result = _parse_cpu("-500m")
        assert result == -512

    def test_negative_cores_parses(self):
        """Negative cores parse but result in negative units."""
        result = _parse_cpu("-1")
        assert result == -1024


class TestParseMemory:
    """Tests for _parse_memory function."""

    # Valid inputs - Mi (mebibytes) format
    @pytest.mark.parametrize(
        "input_value,expected",
        {
            "512Mi": 512,
            "1024Mi": 1024,
            "2048Mi": 2048,
            "4096Mi": 4096,
            "256Mi": 256,
            "8192Mi": 8192,
        }.items(),
    )
    def test_mebibytes_format(self, input_value: str, expected: int):
        """Should parse mebibytes format correctly."""
        assert _parse_memory(input_value) == expected

    # Valid inputs - Gi (gibibytes) format
    @pytest.mark.parametrize(
        "input_value,expected",
        {
            "1Gi": 1024,
            "2Gi": 2048,
            "4Gi": 4096,
            "8Gi": 8192,
            "0.5Gi": 512,
            "1.5Gi": 1536,
            "16Gi": 16384,
        }.items(),
    )
    def test_gibibytes_format(self, input_value: str, expected: int):
        """Should parse gibibytes format correctly."""
        assert _parse_memory(input_value) == expected

    # Valid inputs - M (decimal megabytes) format
    @pytest.mark.parametrize(
        "input_value,expected",
        {
            "512M": 512,
            "1000M": 1000,
            "2000M": 2000,
        }.items(),
    )
    def test_decimal_megabytes_format(self, input_value: str, expected: int):
        """Should parse decimal megabytes format correctly."""
        assert _parse_memory(input_value) == expected

    # Valid inputs - G (decimal gigabytes) format
    @pytest.mark.parametrize(
        "input_value,expected",
        {
            "1G": 1000,
            "2G": 2000,
            "4G": 4000,
            "0.5G": 500,
        }.items(),
    )
    def test_decimal_gigabytes_format(self, input_value: str, expected: int):
        """Should parse decimal gigabytes format correctly."""
        assert _parse_memory(input_value) == expected

    # Valid inputs - plain integer (MB for backwards compatibility)
    @pytest.mark.parametrize(
        "input_value,expected",
        {
            512: 512,
            1024: 1024,
            2048: 2048,
            4096: 4096,
            0: 0,
        }.items(),
    )
    def test_plain_integer(self, input_value: int, expected: int):
        """Should treat plain integers as MB."""
        assert _parse_memory(input_value) == expected

    # Valid inputs - plain number as string
    @pytest.mark.parametrize(
        "input_value,expected",
        {
            "512": 512,
            "1024": 1024,
            "2048.0": 2048,
            "4096.5": 4096,  # Truncates to int
        }.items(),
    )
    def test_plain_number_as_string(self, input_value: str, expected: int):
        """Should parse plain numbers as MB."""
        assert _parse_memory(input_value) == expected

    # Whitespace handling
    @pytest.mark.parametrize(
        "input_value,expected",
        {
            "  512Mi  ": 512,
            " 1Gi ": 1024,
            "\t2048\n": 2048,
            "  1024M  ": 1024,
        }.items(),
    )
    def test_whitespace_handling(self, input_value: str, expected: int):
        """Should handle leading/trailing whitespace."""
        assert _parse_memory(input_value) == expected

    # Edge cases
    def test_zero_mebibytes(self):
        """Should handle zero mebibytes."""
        assert _parse_memory("0Mi") == 0

    def test_zero_gibibytes(self):
        """Should handle zero gibibytes."""
        assert _parse_memory("0Gi") == 0

    def test_fractional_gibibytes(self):
        """Should handle fractional gibibytes."""
        assert _parse_memory("0.25Gi") == 256
        assert _parse_memory("1.75Gi") == 1792

    def test_fractional_mebibytes(self):
        """Should handle fractional mebibytes (truncated)."""
        assert _parse_memory("512.5Mi") == 512

    # Malformed inputs
    def test_empty_string_raises(self):
        """Should raise ValueError for empty string."""
        with pytest.raises(ValueError):
            _parse_memory("")

    def test_whitespace_only_raises(self):
        """Should raise ValueError for whitespace-only string."""
        with pytest.raises(ValueError):
            _parse_memory("   ")

    def test_invalid_suffix_raises(self):
        """Should raise ValueError for invalid suffix."""
        with pytest.raises(ValueError):
            _parse_memory("512Ki")  # Ki is not supported

    def test_non_numeric_raises(self):
        """Should raise ValueError for non-numeric string."""
        with pytest.raises(ValueError):
            _parse_memory("abc")

    def test_lowercase_suffix_raises(self):
        """Should raise ValueError for lowercase suffix (case sensitive)."""
        with pytest.raises(ValueError):
            _parse_memory("512mi")  # Should be Mi, not mi

    def test_lowercase_gi_raises(self):
        """Should raise ValueError for lowercase Gi."""
        with pytest.raises(ValueError):
            _parse_memory("1gi")  # Should be Gi, not gi

    def test_negative_mebibytes_parses(self):
        """Negative mebibytes parse but result in negative value."""
        result = _parse_memory("-512Mi")
        assert result == -512

    def test_only_suffix_raises(self):
        """Should raise ValueError for suffix without number."""
        with pytest.raises(ValueError):
            _parse_memory("Mi")
        with pytest.raises(ValueError):
            _parse_memory("Gi")
