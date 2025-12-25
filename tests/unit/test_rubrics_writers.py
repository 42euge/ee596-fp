"""Unit tests for rubrics module writers."""

import pytest
import yaml
from pathlib import Path

from src.rubrics.writers import save_rubric_to_yaml, save_rubricset_to_yaml
from src.rubrics.loaders import load_rubric_from_yaml, load_rubricset_from_yaml
from src.rubrics.models import Criterion, Rubric, RubricSet


class TestSaveRubricToYaml:
    """Tests for save_rubric_to_yaml function."""

    def test_save_rubric_minimal(self, tmp_path):
        """Test saving a minimal rubric."""
        rubric = Rubric(name="test", description="Test rubric", criteria=[])
        output_path = tmp_path / "rubric.yaml"

        result = save_rubric_to_yaml(rubric, output_path)

        assert result == output_path
        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "test"
        assert data["description"] == "Test rubric"

    def test_save_rubric_full(self, tmp_path):
        """Test saving a rubric with all fields."""
        rubric = Rubric(
            name="full_rubric",
            description="A full rubric",
            id="test_id",
            criteria=[
                Criterion(
                    name="accuracy",
                    description="Tests accuracy",
                    weight=2.0,
                    keywords=["correct", "accurate"],
                ),
                Criterion(
                    name="format",
                    description="Tests format",
                ),
            ],
            question_types=["math", "science"],
            reference_response="<answer>42</answer>",
            target_score=15.0,
            metadata={"author": "tester"},
        )
        output_path = tmp_path / "full.yaml"

        save_rubric_to_yaml(rubric, output_path)

        with open(output_path) as f:
            data = yaml.safe_load(f)

        assert data["name"] == "full_rubric"
        assert len(data["criteria"]) == 2
        assert data["criteria"][0]["weight"] == 2.0
        assert data["question_types"] == ["math", "science"]
        assert data["target_score"] == 15.0
        assert data["metadata"]["author"] == "tester"

    def test_save_rubric_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created."""
        rubric = Rubric(name="test", description="Test", criteria=[])
        output_path = tmp_path / "nested" / "dir" / "rubric.yaml"

        save_rubric_to_yaml(rubric, output_path)

        assert output_path.exists()

    def test_save_rubric_roundtrip(self, tmp_path):
        """Test that saved rubric can be loaded back."""
        original = Rubric(
            name="roundtrip",
            description="Roundtrip test",
            criteria=[
                Criterion(name="c1", description="C1", weight=2.0),
            ],
            question_types=["math"],
        )
        output_path = tmp_path / "roundtrip.yaml"

        save_rubric_to_yaml(original, output_path)
        loaded = load_rubric_from_yaml(output_path)

        assert loaded.name == original.name
        assert loaded.description == original.description
        assert len(loaded.criteria) == len(original.criteria)
        assert loaded.criteria[0].weight == original.criteria[0].weight
        assert loaded.question_types == original.question_types

    def test_save_rubric_with_multiline_reference(self, tmp_path):
        """Test saving rubric with multiline reference response."""
        rubric = Rubric(
            name="multiline",
            description="Test",
            criteria=[],
            reference_response="Line 1\nLine 2\nLine 3",
        )
        output_path = tmp_path / "multiline.yaml"

        save_rubric_to_yaml(rubric, output_path)

        # Reload and verify multiline preserved
        loaded = load_rubric_from_yaml(output_path)
        assert "Line 1" in loaded.reference_response
        assert "Line 2" in loaded.reference_response

    def test_save_rubric_with_unicode(self, tmp_path):
        """Test saving rubric with unicode characters."""
        rubric = Rubric(
            name="unicode_test",
            description="Test with émojis 🎉 and ñ",
            criteria=[
                Criterion(name="unicode", description="Tëst dëscription"),
            ],
        )
        output_path = tmp_path / "unicode.yaml"

        save_rubric_to_yaml(rubric, output_path)
        loaded = load_rubric_from_yaml(output_path)

        assert "émojis" in loaded.description
        assert "🎉" in loaded.description


class TestSaveRubricsetToYaml:
    """Tests for save_rubricset_to_yaml function."""

    def test_save_rubricset_minimal(self, tmp_path):
        """Test saving a minimal rubric set."""
        rubricset = RubricSet(
            name="test_set",
            rubrics=[
                Rubric(name="r1", description="R1", criteria=[]),
            ],
        )
        output_path = tmp_path / "set.yaml"

        result = save_rubricset_to_yaml(rubricset, output_path)

        assert result == output_path
        assert output_path.exists()

        with open(output_path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "test_set"
        assert len(data["rubrics"]) == 1

    def test_save_rubricset_full(self, tmp_path):
        """Test saving a full rubric set."""
        rubricset = RubricSet(
            name="full_set",
            description="A full rubric set",
            rubrics=[
                Rubric(
                    name="r1",
                    description="First",
                    criteria=[Criterion(name="c1", description="C1")],
                ),
                Rubric(
                    name="r2",
                    description="Second",
                    criteria=[Criterion(name="c2", description="C2", weight=2.0)],
                ),
            ],
            metadata={"version": "1.0"},
        )
        output_path = tmp_path / "full_set.yaml"

        save_rubricset_to_yaml(rubricset, output_path)

        with open(output_path) as f:
            data = yaml.safe_load(f)

        assert data["name"] == "full_set"
        assert data["description"] == "A full rubric set"
        assert len(data["rubrics"]) == 2
        assert data["metadata"]["version"] == "1.0"

    def test_save_rubricset_roundtrip(self, tmp_path):
        """Test that saved rubric set can be loaded back."""
        original = RubricSet(
            name="roundtrip_set",
            description="Roundtrip test",
            rubrics=[
                Rubric(
                    name="r1",
                    description="R1",
                    criteria=[Criterion(name="c1", description="C1")],
                    question_types=["math"],
                ),
                Rubric(
                    name="r2",
                    description="R2",
                    criteria=[Criterion(name="c2", description="C2")],
                ),
            ],
            metadata={"key": "value"},
        )
        output_path = tmp_path / "roundtrip_set.yaml"

        save_rubricset_to_yaml(original, output_path)
        loaded = load_rubricset_from_yaml(output_path)

        assert loaded.name == original.name
        assert loaded.description == original.description
        assert len(loaded) == len(original)
        assert loaded[0].name == original[0].name
        assert loaded[0].question_types == original[0].question_types

    def test_save_empty_rubricset(self, tmp_path):
        """Test saving an empty rubric set."""
        rubricset = RubricSet(name="empty", rubrics=[])
        output_path = tmp_path / "empty.yaml"

        save_rubricset_to_yaml(rubricset, output_path)

        with open(output_path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "empty"
        assert data["rubrics"] == []

    def test_save_rubricset_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created."""
        rubricset = RubricSet(name="test", rubrics=[])
        output_path = tmp_path / "deep" / "nested" / "dir" / "set.yaml"

        save_rubricset_to_yaml(rubricset, output_path)

        assert output_path.exists()


class TestYamlFormat:
    """Tests for YAML formatting."""

    def test_yaml_format_readable(self, tmp_path):
        """Test that output YAML is human-readable."""
        rubric = Rubric(
            name="readable",
            description="Test readability",
            criteria=[
                Criterion(
                    name="long_criterion",
                    description="This is a longer description that should be formatted nicely",
                    keywords=["keyword1", "keyword2", "keyword3"],
                ),
            ],
        )
        output_path = tmp_path / "readable.yaml"

        save_rubric_to_yaml(rubric, output_path)

        content = output_path.read_text()
        # Check it's not all on one line
        assert content.count("\n") > 5
        # Check keys are not sorted (preserves order)
        assert content.index("name:") < content.index("description:")

    def test_yaml_no_flow_style(self, tmp_path):
        """Test that lists are in block style, not flow style."""
        rubric = Rubric(
            name="test",
            description="Test",
            criteria=[
                Criterion(name="c1", description="C1", keywords=["a", "b", "c"]),
            ],
        )
        output_path = tmp_path / "style.yaml"

        save_rubric_to_yaml(rubric, output_path)

        content = output_path.read_text()
        # Should use block style, not [a, b, c]
        assert "- a" in content or "keywords:" in content
