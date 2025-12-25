"""Unit tests for rubrics module loaders."""

import pytest
import tempfile
from pathlib import Path

from src.rubrics.loaders import (
    load_rubric_from_yaml,
    load_rubricset_from_yaml,
    load_rubrics_from_directory,
    _parse_criterion,
    _parse_rubric,
    _parse_rubricset,
    _extract_keywords,
)
from src.rubrics.models import Criterion, Rubric, RubricSet


class TestLoadRubricFromYaml:
    """Tests for load_rubric_from_yaml function."""

    def test_load_single_rubric(self, tmp_path):
        """Test loading a single rubric from YAML."""
        yaml_content = """
name: test_rubric
description: A test rubric
criteria:
  - name: accuracy
    description: Tests accuracy
    weight: 2.0
question_types:
  - math
"""
        yaml_file = tmp_path / "rubric.yaml"
        yaml_file.write_text(yaml_content)

        rubric = load_rubric_from_yaml(yaml_file)

        assert rubric.name == "test_rubric"
        assert rubric.description == "A test rubric"
        assert len(rubric.criteria) == 1
        assert rubric.criteria[0].name == "accuracy"
        assert rubric.criteria[0].weight == 2.0
        assert "math" in rubric.question_types

    def test_load_rubric_with_reference(self, tmp_path):
        """Test loading a rubric with reference response."""
        yaml_content = """
name: with_ref
description: Has reference
criteria:
  - name: format
    description: Format check
reference_response: |
  <reasoning>Step by step</reasoning>
  <answer>42</answer>
target_score: 15.0
"""
        yaml_file = tmp_path / "rubric.yaml"
        yaml_file.write_text(yaml_content)

        rubric = load_rubric_from_yaml(yaml_file)

        assert rubric.reference_response is not None
        assert "<reasoning>" in rubric.reference_response
        assert rubric.target_score == 15.0

    def test_load_rubric_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_rubric_from_yaml("/nonexistent/path.yaml")


class TestLoadRubricsetFromYaml:
    """Tests for load_rubricset_from_yaml function."""

    def test_load_rubricset(self, tmp_path):
        """Test loading a rubric set from YAML."""
        yaml_content = """
name: test_set
description: A test rubric set
rubrics:
  - name: rubric1
    description: First rubric
    criteria:
      - name: c1
        description: Criterion 1
  - name: rubric2
    description: Second rubric
    criteria:
      - name: c2
        description: Criterion 2
"""
        yaml_file = tmp_path / "rubricset.yaml"
        yaml_file.write_text(yaml_content)

        rubricset = load_rubricset_from_yaml(yaml_file)

        assert rubricset.name == "test_set"
        assert rubricset.description == "A test rubric set"
        assert len(rubricset) == 2
        assert rubricset[0].name == "rubric1"
        assert rubricset[1].name == "rubric2"

    def test_load_single_rubric_as_rubricset(self, tmp_path):
        """Test loading a single rubric file as RubricSet."""
        yaml_content = """
name: single_rubric
description: A single rubric
criteria:
  - name: test
    description: Test criterion
"""
        yaml_file = tmp_path / "single.yaml"
        yaml_file.write_text(yaml_content)

        rubricset = load_rubricset_from_yaml(yaml_file)

        assert isinstance(rubricset, RubricSet)
        assert len(rubricset) == 1
        assert rubricset[0].name == "single_rubric"

    def test_load_rubricset_with_metadata(self, tmp_path):
        """Test loading rubricset with metadata."""
        yaml_content = """
name: with_meta
description: Has metadata
metadata:
  version: "1.0"
  author: test
rubrics:
  - name: r1
    description: R1
    criteria: []
"""
        yaml_file = tmp_path / "meta.yaml"
        yaml_file.write_text(yaml_content)

        rubricset = load_rubricset_from_yaml(yaml_file)

        assert rubricset.metadata.get("version") == "1.0"
        assert rubricset.metadata.get("author") == "test"
        assert "source_file" in rubricset.metadata


class TestLoadRubricsFromDirectory:
    """Tests for load_rubrics_from_directory function."""

    def test_load_from_directory(self, tmp_path):
        """Test loading multiple rubrics from a directory."""
        # Create multiple YAML files
        yaml1 = """
name: rubric1
description: First
criteria:
  - name: c1
    description: C1
"""
        yaml2 = """
name: rubric2
description: Second
criteria:
  - name: c2
    description: C2
"""
        (tmp_path / "r1.yaml").write_text(yaml1)
        (tmp_path / "r2.yaml").write_text(yaml2)

        rubricset = load_rubrics_from_directory(tmp_path)

        assert len(rubricset) == 2
        names = [r.name for r in rubricset]
        assert "rubric1" in names
        assert "rubric2" in names

    def test_load_from_directory_with_rubricset_files(self, tmp_path):
        """Test loading from directory containing rubricset files."""
        yaml_content = """
name: set1
rubrics:
  - name: r1
    description: R1
    criteria: []
  - name: r2
    description: R2
    criteria: []
"""
        (tmp_path / "set.yaml").write_text(yaml_content)

        rubricset = load_rubrics_from_directory(tmp_path)

        assert len(rubricset) == 2

    def test_load_from_directory_custom_pattern(self, tmp_path):
        """Test loading with custom glob pattern."""
        (tmp_path / "rubric.yaml").write_text("name: yaml\ndescription: YAML\ncriteria: []")
        (tmp_path / "rubric.yml").write_text("name: yml\ndescription: YML\ncriteria: []")

        yaml_only = load_rubrics_from_directory(tmp_path, pattern="*.yaml")
        all_files = load_rubrics_from_directory(tmp_path, pattern="*.y*ml")

        assert len(yaml_only) == 1
        assert len(all_files) == 2


class TestParseFunctions:
    """Tests for internal parsing functions."""

    def test_parse_criterion_minimal(self):
        """Test parsing criterion with minimal data."""
        data = {"name": "test", "description": "Test desc"}
        criterion = _parse_criterion(data)

        assert criterion.name == "test"
        assert criterion.description == "Test desc"
        assert criterion.weight == 1.0

    def test_parse_criterion_full(self):
        """Test parsing criterion with all fields."""
        data = {
            "name": "full",
            "description": "Full desc",
            "weight": 2.5,
            "score_range": [0, 5],
            "keywords": ["key1", "key2"],
            "examples": {0: "Bad", 5: "Good"},
        }
        criterion = _parse_criterion(data)

        assert criterion.weight == 2.5
        assert criterion.score_range == (0.0, 5.0)
        assert criterion.keywords == ["key1", "key2"]

    def test_parse_rubric(self):
        """Test parsing rubric from dict."""
        data = {
            "name": "test",
            "description": "Test",
            "criteria": [{"name": "c1", "description": "C1"}],
            "question_types": ["math"],
        }
        rubric = _parse_rubric(data)

        assert rubric.name == "test"
        assert len(rubric.criteria) == 1
        assert rubric.question_types == ["math"]

    def test_parse_rubricset(self):
        """Test parsing rubricset from dict."""
        data = {
            "name": "set",
            "description": "A set",
            "rubrics": [
                {"name": "r1", "description": "R1", "criteria": []},
                {"name": "r2", "description": "R2", "criteria": []},
            ],
        }
        rubricset = _parse_rubricset(data)

        assert rubricset.name == "set"
        assert len(rubricset) == 2


class TestExtractKeywords:
    """Tests for _extract_keywords function."""

    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        text = "The student should demonstrate clear reasoning and accuracy."
        keywords = _extract_keywords(text)

        assert "student" in keywords
        assert "demonstrate" in keywords
        assert "clear" in keywords
        # Stopwords should be filtered
        assert "the" not in keywords
        assert "and" not in keywords

    def test_extract_keywords_min_length(self):
        """Test minimum length filtering."""
        text = "A is B or C"
        keywords = _extract_keywords(text, min_length=3)

        # All words are too short
        assert len(keywords) == 0

    def test_extract_keywords_max_keywords(self):
        """Test max keywords limit."""
        text = " ".join([f"word{i}" for i in range(50)])
        keywords = _extract_keywords(text, max_keywords=10)

        assert len(keywords) == 10

    def test_extract_keywords_removes_duplicates(self):
        """Test duplicate removal."""
        text = "accuracy accuracy accuracy testing"
        keywords = _extract_keywords(text)

        assert keywords.count("accuracy") == 1

    def test_extract_keywords_handles_punctuation(self):
        """Test punctuation handling."""
        text = "Test: accuracy, precision; and recall!"
        keywords = _extract_keywords(text)

        assert "accuracy" in keywords
        assert "precision" in keywords
        assert "recall" in keywords


class TestLoadExampleFile:
    """Test loading the actual example file."""

    def test_load_example_math_yaml(self):
        """Test loading the example_math.yaml file."""
        example_path = Path(__file__).parent.parent.parent / "rubrics" / "example_math.yaml"

        if example_path.exists():
            rubricset = load_rubricset_from_yaml(example_path)

            assert rubricset.name == "Math Reasoning Rubric Set"
            assert len(rubricset) >= 1
            # Check first rubric has criteria
            assert len(rubricset[0].criteria) >= 1
        else:
            pytest.skip("example_math.yaml not found")
