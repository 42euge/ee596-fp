"""Unit tests for rubrics module models."""

import pytest
from src.rubrics.models import Criterion, Rubric, RubricSet


class TestCriterion:
    """Tests for Criterion dataclass."""

    def test_criterion_creation_minimal(self):
        """Test creating a criterion with minimal required fields."""
        criterion = Criterion(name="accuracy", description="Tests accuracy")
        assert criterion.name == "accuracy"
        assert criterion.description == "Tests accuracy"
        assert criterion.weight == 1.0
        assert criterion.score_range == (0.0, 10.0)
        assert criterion.keywords == []
        assert criterion.examples == {}

    def test_criterion_creation_full(self):
        """Test creating a criterion with all fields."""
        criterion = Criterion(
            name="format",
            description="Uses proper format",
            weight=2.0,
            score_range=(0.0, 5.0),
            keywords=["<reasoning>", "<answer>"],
            examples={0: "Bad", 5: "Good"},
        )
        assert criterion.name == "format"
        assert criterion.weight == 2.0
        assert criterion.score_range == (0.0, 5.0)
        assert "<reasoning>" in criterion.keywords
        assert criterion.examples[5] == "Good"

    def test_criterion_to_text(self):
        """Test converting criterion to text."""
        criterion = Criterion(name="clarity", description="Clear explanation")
        text = criterion.to_text()
        assert "clarity" in text
        assert "Clear explanation" in text

    def test_criterion_to_dict(self):
        """Test converting criterion to dict."""
        criterion = Criterion(
            name="test",
            description="Test desc",
            weight=2.0,
            keywords=["key1"],
        )
        d = criterion.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "Test desc"
        assert d["weight"] == 2.0
        assert d["keywords"] == ["key1"]
        # Default score_range should not be included
        assert "score_range" not in d

    def test_criterion_to_dict_excludes_defaults(self):
        """Test that to_dict excludes default values."""
        criterion = Criterion(name="test", description="Test")
        d = criterion.to_dict()
        assert "weight" not in d  # Default is 1.0
        assert "score_range" not in d
        assert "keywords" not in d
        assert "examples" not in d


class TestRubric:
    """Tests for Rubric dataclass."""

    def test_rubric_creation_minimal(self):
        """Test creating a rubric with minimal required fields."""
        rubric = Rubric(name="test", description="A test rubric")
        assert rubric.name == "test"
        assert rubric.description == "A test rubric"
        assert rubric.criteria == []
        assert rubric.id is not None  # Auto-generated
        assert len(rubric.id) == 8

    def test_rubric_creation_full(self):
        """Test creating a rubric with all fields."""
        criterion = Criterion(name="acc", description="Accuracy")
        rubric = Rubric(
            name="full_rubric",
            description="Full rubric",
            criteria=[criterion],
            id="custom_id",
            question_types=["math", "science"],
            reference_response="Example response",
            target_score=10.0,
            metadata={"author": "test"},
        )
        assert rubric.name == "full_rubric"
        assert len(rubric.criteria) == 1
        assert rubric.id == "custom_id"
        assert "math" in rubric.question_types
        assert rubric.reference_response == "Example response"
        assert rubric.target_score == 10.0
        assert rubric.metadata["author"] == "test"

    def test_rubric_to_text(self):
        """Test converting rubric to text."""
        rubric = Rubric(
            name="test_rubric",
            description="Testing",
            criteria=[Criterion(name="c1", description="Criterion 1")],
        )
        text = rubric.to_text()
        assert "test_rubric" in text
        assert "Testing" in text
        assert "Criterion 1" in text

    def test_rubric_get_keywords(self):
        """Test extracting keywords from all criteria."""
        rubric = Rubric(
            name="test",
            description="Test",
            criteria=[
                Criterion(name="c1", description="C1", keywords=["key1", "key2"]),
                Criterion(name="c2", description="C2", keywords=["key3"]),
            ],
        )
        keywords = rubric.get_keywords()
        assert "key1" in keywords
        assert "key2" in keywords
        assert "key3" in keywords

    def test_rubric_add_criterion(self):
        """Test adding a criterion to a rubric."""
        rubric = Rubric(name="test", description="Test")
        criterion = Criterion(name="new", description="New criterion")

        result = rubric.add_criterion(criterion)

        assert result is rubric  # Returns self for chaining
        assert len(rubric.criteria) == 1
        assert rubric.criteria[0].name == "new"

    def test_rubric_remove_criterion(self):
        """Test removing a criterion from a rubric."""
        rubric = Rubric(
            name="test",
            description="Test",
            criteria=[
                Criterion(name="keep", description="Keep"),
                Criterion(name="remove", description="Remove"),
            ],
        )

        result = rubric.remove_criterion("remove")

        assert result is True
        assert len(rubric.criteria) == 1
        assert rubric.criteria[0].name == "keep"

    def test_rubric_remove_criterion_not_found(self):
        """Test removing a non-existent criterion."""
        rubric = Rubric(name="test", description="Test")
        result = rubric.remove_criterion("nonexistent")
        assert result is False

    def test_rubric_update_criterion(self):
        """Test updating a criterion's attributes."""
        rubric = Rubric(
            name="test",
            description="Test",
            criteria=[Criterion(name="update_me", description="Original", weight=1.0)],
        )

        result = rubric.update_criterion("update_me", weight=2.0, description="Updated")

        assert result is True
        assert rubric.criteria[0].weight == 2.0
        assert rubric.criteria[0].description == "Updated"

    def test_rubric_get_criterion(self):
        """Test getting a criterion by name."""
        criterion = Criterion(name="target", description="Target")
        rubric = Rubric(name="test", description="Test", criteria=[criterion])

        result = rubric.get_criterion("target")

        assert result is criterion
        assert rubric.get_criterion("nonexistent") is None

    def test_rubric_to_dict(self):
        """Test converting rubric to dict."""
        rubric = Rubric(
            name="test",
            description="Test desc",
            criteria=[Criterion(name="c1", description="C1")],
            question_types=["math"],
            target_score=10.0,
        )
        d = rubric.to_dict()

        assert d["name"] == "test"
        assert d["description"] == "Test desc"
        assert len(d["criteria"]) == 1
        assert d["question_types"] == ["math"]
        assert d["target_score"] == 10.0


class TestRubricSet:
    """Tests for RubricSet dataclass."""

    @pytest.fixture
    def sample_rubrics(self):
        """Create sample rubrics for testing."""
        return [
            Rubric(name="rubric1", description="First", id="r1", question_types=["math"]),
            Rubric(name="rubric2", description="Second", id="r2", question_types=["science"]),
            Rubric(name="rubric3", description="Third", id="r3", question_types=["math", "general"]),
        ]

    def test_rubricset_creation(self, sample_rubrics):
        """Test creating a rubric set."""
        rubricset = RubricSet(
            name="test_set",
            rubrics=sample_rubrics,
            description="A test set",
        )
        assert rubricset.name == "test_set"
        assert len(rubricset.rubrics) == 3
        assert rubricset.description == "A test set"

    def test_rubricset_len(self, sample_rubrics):
        """Test __len__ method."""
        rubricset = RubricSet(name="test", rubrics=sample_rubrics)
        assert len(rubricset) == 3

    def test_rubricset_iter(self, sample_rubrics):
        """Test __iter__ method."""
        rubricset = RubricSet(name="test", rubrics=sample_rubrics)
        names = [r.name for r in rubricset]
        assert names == ["rubric1", "rubric2", "rubric3"]

    def test_rubricset_getitem_by_index(self, sample_rubrics):
        """Test __getitem__ with integer index."""
        rubricset = RubricSet(name="test", rubrics=sample_rubrics)
        assert rubricset[0].name == "rubric1"
        assert rubricset[1].name == "rubric2"
        assert rubricset[-1].name == "rubric3"

    def test_rubricset_getitem_by_name(self, sample_rubrics):
        """Test __getitem__ with string name."""
        rubricset = RubricSet(name="test", rubrics=sample_rubrics)
        assert rubricset["rubric2"].description == "Second"

    def test_rubricset_getitem_by_id(self, sample_rubrics):
        """Test __getitem__ with string id."""
        rubricset = RubricSet(name="test", rubrics=sample_rubrics)
        assert rubricset["r1"].name == "rubric1"

    def test_rubricset_getitem_not_found(self, sample_rubrics):
        """Test __getitem__ raises KeyError for missing rubric."""
        rubricset = RubricSet(name="test", rubrics=sample_rubrics)
        with pytest.raises(KeyError):
            _ = rubricset["nonexistent"]

    def test_rubricset_contains(self, sample_rubrics):
        """Test __contains__ method."""
        rubricset = RubricSet(name="test", rubrics=sample_rubrics)
        assert "rubric1" in rubricset
        assert "r2" in rubricset
        assert "nonexistent" not in rubricset

    def test_rubricset_add(self, sample_rubrics):
        """Test adding a rubric to the set."""
        rubricset = RubricSet(name="test", rubrics=sample_rubrics[:2])
        new_rubric = Rubric(name="new", description="New")

        result = rubricset.add(new_rubric)

        assert result is rubricset  # Returns self for chaining
        assert len(rubricset) == 3
        assert rubricset[-1].name == "new"

    def test_rubricset_remove_by_name(self, sample_rubrics):
        """Test removing a rubric by name."""
        rubricset = RubricSet(name="test", rubrics=sample_rubrics)

        result = rubricset.remove("rubric2")

        assert result is True
        assert len(rubricset) == 2
        assert "rubric2" not in rubricset

    def test_rubricset_remove_by_id(self, sample_rubrics):
        """Test removing a rubric by id."""
        rubricset = RubricSet(name="test", rubrics=sample_rubrics)

        result = rubricset.remove("r1")

        assert result is True
        assert "rubric1" not in rubricset

    def test_rubricset_filter(self, sample_rubrics):
        """Test filtering rubrics by question type."""
        rubricset = RubricSet(name="test", rubrics=sample_rubrics)

        math_rubrics = rubricset.filter("math")

        assert len(math_rubrics) == 2
        names = [r.name for r in math_rubrics]
        assert "rubric1" in names
        assert "rubric3" in names

    def test_rubricset_get_for_question_with_detector(self, sample_rubrics):
        """Test get_for_question with type detector."""
        rubricset = RubricSet(name="test", rubrics=sample_rubrics)

        def mock_detector(question):
            return "math" if "math" in question.lower() else "general"

        result = rubricset.get_for_question("A math problem", mock_detector)

        assert result is not None
        assert "math" in result.question_types

    def test_rubricset_get_for_question_fallback(self, sample_rubrics):
        """Test get_for_question falls back to first rubric."""
        rubricset = RubricSet(name="test", rubrics=sample_rubrics)

        result = rubricset.get_for_question("Any question")

        assert result is not None
        assert result.name == "rubric1"  # First rubric

    def test_rubricset_get_for_question_empty(self):
        """Test get_for_question with empty set."""
        rubricset = RubricSet(name="empty", rubrics=[])

        result = rubricset.get_for_question("Any question")

        assert result is None

    def test_rubricset_to_dict(self, sample_rubrics):
        """Test converting rubric set to dict."""
        rubricset = RubricSet(
            name="test",
            rubrics=sample_rubrics,
            description="Test set",
            metadata={"version": "1.0"},
        )
        d = rubricset.to_dict()

        assert d["name"] == "test"
        assert d["description"] == "Test set"
        assert len(d["rubrics"]) == 3
        assert d["metadata"]["version"] == "1.0"
