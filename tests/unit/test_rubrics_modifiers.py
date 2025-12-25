"""Unit tests for rubrics module modifiers."""

import pytest

from src.rubrics.modifiers import (
    RubricBuilder,
    clone_rubric,
    merge_rubrics,
    apply_transform,
    adjust_weights,
    add_keywords_to_criterion,
    scale_weights,
    normalize_weights,
)
from src.rubrics.models import Criterion, Rubric, RubricSet


class TestRubricBuilder:
    """Tests for RubricBuilder class."""

    def test_builder_minimal(self):
        """Test building a rubric with minimal fields."""
        rubric = RubricBuilder("test", "A test rubric").build()

        assert rubric.name == "test"
        assert rubric.description == "A test rubric"
        assert rubric.criteria == []

    def test_builder_with_criterion(self):
        """Test adding criteria via builder."""
        rubric = (
            RubricBuilder("test", "Test")
            .with_criterion("accuracy", "Tests accuracy")
            .with_criterion("format", "Tests format", weight=2.0)
            .build()
        )

        assert len(rubric.criteria) == 2
        assert rubric.criteria[0].name == "accuracy"
        assert rubric.criteria[0].weight == 1.0
        assert rubric.criteria[1].name == "format"
        assert rubric.criteria[1].weight == 2.0

    def test_builder_with_criterion_full(self):
        """Test adding criterion with all options."""
        rubric = (
            RubricBuilder("test", "Test")
            .with_criterion(
                name="full",
                description="Full criterion",
                weight=3.0,
                keywords=["key1", "key2"],
                score_range=(0.0, 5.0),
                examples={0: "Bad", 5: "Good"},
            )
            .build()
        )

        criterion = rubric.criteria[0]
        assert criterion.name == "full"
        assert criterion.weight == 3.0
        assert criterion.keywords == ["key1", "key2"]
        assert criterion.score_range == (0.0, 5.0)
        assert criterion.examples[5] == "Good"

    def test_builder_with_question_types(self):
        """Test setting question types."""
        rubric = (
            RubricBuilder("test", "Test")
            .with_question_types("math", "science", "general")
            .build()
        )

        assert rubric.question_types == ["math", "science", "general"]

    def test_builder_with_reference(self):
        """Test setting reference response."""
        rubric = (
            RubricBuilder("test", "Test")
            .with_reference("<answer>42</answer>", target_score=10.0)
            .build()
        )

        assert rubric.reference_response == "<answer>42</answer>"
        assert rubric.target_score == 10.0

    def test_builder_with_metadata(self):
        """Test adding metadata."""
        rubric = (
            RubricBuilder("test", "Test")
            .with_metadata(author="tester", version="1.0")
            .build()
        )

        assert rubric.metadata["author"] == "tester"
        assert rubric.metadata["version"] == "1.0"

    def test_builder_with_id(self):
        """Test setting custom ID."""
        rubric = (
            RubricBuilder("test", "Test")
            .with_id("custom_id")
            .build()
        )

        assert rubric.id == "custom_id"

    def test_builder_chaining(self):
        """Test fluent interface chaining."""
        rubric = (
            RubricBuilder("complete", "Complete rubric")
            .with_id("r1")
            .with_criterion("c1", "Criterion 1", weight=2.0)
            .with_criterion("c2", "Criterion 2")
            .with_question_types("math")
            .with_reference("Example", target_score=15.0)
            .with_metadata(author="test")
            .build()
        )

        assert rubric.name == "complete"
        assert rubric.id == "r1"
        assert len(rubric.criteria) == 2
        assert rubric.question_types == ["math"]
        assert rubric.reference_response == "Example"
        assert rubric.target_score == 15.0
        assert rubric.metadata["author"] == "test"


class TestCloneRubric:
    """Tests for clone_rubric function."""

    @pytest.fixture
    def sample_rubric(self):
        """Create a sample rubric for testing."""
        return Rubric(
            name="original",
            description="Original rubric",
            id="orig_id",
            criteria=[
                Criterion(name="c1", description="C1", weight=2.0),
            ],
            question_types=["math"],
            metadata={"key": "value"},
        )

    def test_clone_creates_copy(self, sample_rubric):
        """Test that clone creates an independent copy."""
        cloned = clone_rubric(sample_rubric)

        # Verify it's a copy
        assert cloned is not sample_rubric
        assert cloned.name == sample_rubric.name
        assert cloned.description == sample_rubric.description

        # Modify clone, original should be unchanged
        cloned.criteria[0].weight = 5.0
        assert sample_rubric.criteria[0].weight == 2.0

    def test_clone_with_new_name(self, sample_rubric):
        """Test cloning with a new name."""
        cloned = clone_rubric(sample_rubric, new_name="cloned_rubric")

        assert cloned.name == "cloned_rubric"
        assert cloned.id != sample_rubric.id  # Should get new ID
        assert len(cloned.id) == 8

    def test_clone_preserves_all_fields(self, sample_rubric):
        """Test that clone preserves all fields."""
        sample_rubric.reference_response = "ref"
        sample_rubric.target_score = 10.0

        cloned = clone_rubric(sample_rubric)

        assert cloned.criteria[0].name == "c1"
        assert cloned.question_types == ["math"]
        assert cloned.metadata["key"] == "value"
        assert cloned.reference_response == "ref"
        assert cloned.target_score == 10.0


class TestMergeRubrics:
    """Tests for merge_rubrics function."""

    def test_merge_rubrics(self):
        """Test merging multiple rubrics."""
        r1 = Rubric(
            name="r1",
            description="First",
            criteria=[Criterion(name="c1", description="C1")],
            question_types=["math"],
        )
        r2 = Rubric(
            name="r2",
            description="Second",
            criteria=[Criterion(name="c2", description="C2")],
            question_types=["science"],
        )

        merged = merge_rubrics([r1, r2], "merged")

        assert merged.name == "merged"
        assert len(merged.criteria) == 2
        assert set(c.name for c in merged.criteria) == {"c1", "c2"}
        assert set(merged.question_types) == {"math", "science"}

    def test_merge_rubrics_skips_duplicates(self):
        """Test that duplicate criteria are skipped."""
        r1 = Rubric(
            name="r1",
            description="First",
            criteria=[Criterion(name="shared", description="Shared criterion")],
        )
        r2 = Rubric(
            name="r2",
            description="Second",
            criteria=[Criterion(name="shared", description="Different desc")],
        )

        merged = merge_rubrics([r1, r2], "merged")

        assert len(merged.criteria) == 1
        # First one wins
        assert merged.criteria[0].description == "Shared criterion"

    def test_merge_rubrics_auto_description(self):
        """Test auto-generated description."""
        r1 = Rubric(name="r1", description="First", criteria=[])
        r2 = Rubric(name="r2", description="Second", criteria=[])

        merged = merge_rubrics([r1, r2], "merged")

        assert "r1" in merged.description
        assert "r2" in merged.description

    def test_merge_rubrics_custom_description(self):
        """Test custom description."""
        r1 = Rubric(name="r1", description="First", criteria=[])

        merged = merge_rubrics([r1], "merged", description="Custom desc")

        assert merged.description == "Custom desc"

    def test_merge_rubrics_metadata(self):
        """Test that merged_from metadata is set."""
        r1 = Rubric(name="r1", description="First", id="id1", criteria=[])
        r2 = Rubric(name="r2", description="Second", id="id2", criteria=[])

        merged = merge_rubrics([r1, r2], "merged")

        assert "merged_from" in merged.metadata
        assert "id1" in merged.metadata["merged_from"]
        assert "id2" in merged.metadata["merged_from"]


class TestApplyTransform:
    """Tests for apply_transform function."""

    def test_apply_transform(self):
        """Test applying a transformation to all rubrics."""
        rubricset = RubricSet(
            name="test",
            rubrics=[
                Rubric(name="r1", description="R1", criteria=[]),
                Rubric(name="r2", description="R2", criteria=[]),
            ],
        )

        def add_prefix(rubric):
            cloned = clone_rubric(rubric)
            cloned.name = "prefix_" + cloned.name
            return cloned

        transformed = apply_transform(rubricset, add_prefix)

        assert transformed.name == "test"
        assert transformed[0].name == "prefix_r1"
        assert transformed[1].name == "prefix_r2"
        # Original unchanged
        assert rubricset[0].name == "r1"


class TestAdjustWeights:
    """Tests for adjust_weights function."""

    def test_adjust_weights(self):
        """Test adjusting criterion weights."""
        rubric = Rubric(
            name="test",
            description="Test",
            criteria=[
                Criterion(name="c1", description="C1", weight=1.0),
                Criterion(name="c2", description="C2", weight=1.0),
            ],
        )

        adjusted = adjust_weights(rubric, {"c1": 3.0, "c2": 2.0})

        assert adjusted.get_criterion("c1").weight == 3.0
        assert adjusted.get_criterion("c2").weight == 2.0
        # Original unchanged
        assert rubric.get_criterion("c1").weight == 1.0

    def test_adjust_weights_partial(self):
        """Test adjusting only some weights."""
        rubric = Rubric(
            name="test",
            description="Test",
            criteria=[
                Criterion(name="c1", description="C1", weight=1.0),
                Criterion(name="c2", description="C2", weight=1.0),
            ],
        )

        adjusted = adjust_weights(rubric, {"c1": 5.0})

        assert adjusted.get_criterion("c1").weight == 5.0
        assert adjusted.get_criterion("c2").weight == 1.0  # Unchanged


class TestAddKeywordsToCriterion:
    """Tests for add_keywords_to_criterion function."""

    def test_add_keywords(self):
        """Test adding keywords to a criterion."""
        rubric = Rubric(
            name="test",
            description="Test",
            criteria=[
                Criterion(name="c1", description="C1", keywords=["existing"]),
            ],
        )

        modified = add_keywords_to_criterion(rubric, "c1", ["new1", "new2"])

        assert "existing" in modified.get_criterion("c1").keywords
        assert "new1" in modified.get_criterion("c1").keywords
        assert "new2" in modified.get_criterion("c1").keywords
        # Original unchanged
        assert "new1" not in rubric.get_criterion("c1").keywords

    def test_add_keywords_no_duplicates(self):
        """Test that duplicate keywords are not added."""
        rubric = Rubric(
            name="test",
            description="Test",
            criteria=[
                Criterion(name="c1", description="C1", keywords=["key1"]),
            ],
        )

        modified = add_keywords_to_criterion(rubric, "c1", ["key1", "key2"])

        keywords = modified.get_criterion("c1").keywords
        assert keywords.count("key1") == 1
        assert "key2" in keywords


class TestScaleWeights:
    """Tests for scale_weights function."""

    def test_scale_weights(self):
        """Test scaling all weights by a factor."""
        rubric = Rubric(
            name="test",
            description="Test",
            criteria=[
                Criterion(name="c1", description="C1", weight=1.0),
                Criterion(name="c2", description="C2", weight=2.0),
            ],
        )

        scaled = scale_weights(rubric, 2.0)

        assert scaled.get_criterion("c1").weight == 2.0
        assert scaled.get_criterion("c2").weight == 4.0
        # Original unchanged
        assert rubric.get_criterion("c1").weight == 1.0


class TestNormalizeWeights:
    """Tests for normalize_weights function."""

    def test_normalize_weights_default(self):
        """Test normalizing weights to sum to 1.0."""
        rubric = Rubric(
            name="test",
            description="Test",
            criteria=[
                Criterion(name="c1", description="C1", weight=2.0),
                Criterion(name="c2", description="C2", weight=3.0),
            ],
        )

        normalized = normalize_weights(rubric)

        total = sum(c.weight for c in normalized.criteria)
        assert abs(total - 1.0) < 0.001
        # Proportions preserved
        assert normalized.get_criterion("c1").weight < normalized.get_criterion("c2").weight

    def test_normalize_weights_custom_sum(self):
        """Test normalizing to a custom target sum."""
        rubric = Rubric(
            name="test",
            description="Test",
            criteria=[
                Criterion(name="c1", description="C1", weight=1.0),
                Criterion(name="c2", description="C2", weight=1.0),
            ],
        )

        normalized = normalize_weights(rubric, target_sum=10.0)

        total = sum(c.weight for c in normalized.criteria)
        assert abs(total - 10.0) < 0.001

    def test_normalize_weights_empty_criteria(self):
        """Test normalizing with no criteria."""
        rubric = Rubric(name="test", description="Test", criteria=[])

        normalized = normalize_weights(rubric)

        assert len(normalized.criteria) == 0
