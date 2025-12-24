"""Integration tests for rubrics scoring functions.

These tests require src.utils to be available and test the integration
between the rubrics module and the existing scoring infrastructure.
"""

import pytest

from src.rubrics.models import Criterion, Rubric, RubricSet
from src.rubrics.modifiers import RubricBuilder
from src.rubrics.scoring import (
    create_rubric_scorer,
    create_rubricset_scorer,
    create_weighted_rubric_scorer,
    create_multi_rubric_scorer,
    rubric_reward_adapter,
)
from src.rubrics.testing import (
    RubricTestConfig,
    RubricTestResult,
    test_rubric_with_dataset,
    quick_test_rubric,
    create_grpo_reward_function,
    compare_rubrics,
)


class TestCreateRubricScorer:
    """Tests for create_rubric_scorer function."""

    @pytest.fixture
    def format_rubric(self):
        """Create a rubric focused on format."""
        return (
            RubricBuilder("format_check", "Checks response format")
            .with_criterion(
                "tags",
                "Uses reasoning and answer tags",
                keywords=["<reasoning>", "</reasoning>", "<answer>", "</answer>"],
            )
            .build()
        )

    def test_scorer_returns_callable(self, format_rubric):
        """Test that create_rubric_scorer returns a callable."""
        scorer = create_rubric_scorer(format_rubric)
        assert callable(scorer)

    def test_scorer_returns_scores(self, format_rubric):
        """Test that scorer returns list of scores."""
        scorer = create_rubric_scorer(format_rubric)

        prompts = ["What is 2+2?"]
        completions = ["<reasoning>2+2=4</reasoning><answer>4</answer>"]

        scores = scorer(prompts, completions)

        assert isinstance(scores, list)
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_scorer_higher_for_matching_content(self, format_rubric):
        """Test that matching content gets higher scores."""
        scorer = create_rubric_scorer(format_rubric)

        prompts = ["test"] * 2
        completions = [
            "<reasoning>Step by step</reasoning><answer>42</answer>",  # Has tags
            "The answer is 42",  # No tags
        ]

        scores = scorer(prompts, completions)

        assert scores[0] > scores[1]

    def test_scorer_handles_batch(self, format_rubric):
        """Test scorer handles batch of completions."""
        scorer = create_rubric_scorer(format_rubric)

        prompts = ["q1", "q2", "q3"]
        completions = ["a1", "a2", "a3"]

        scores = scorer(prompts, completions)

        assert len(scores) == 3


class TestCreateRubricsetScorer:
    """Tests for create_rubricset_scorer function."""

    @pytest.fixture
    def rubricset(self):
        """Create a rubric set with multiple rubrics."""
        math_rubric = (
            RubricBuilder("math", "Math rubric")
            .with_criterion("calculation", "Shows calculations", keywords=["calculate", "equals"])
            .with_question_types("math")
            .build()
        )
        general_rubric = (
            RubricBuilder("general", "General rubric")
            .with_criterion("clarity", "Clear explanation", keywords=["explain", "because"])
            .with_question_types("general")
            .build()
        )
        return RubricSet(name="test", rubrics=[math_rubric, general_rubric])

    def test_rubricset_scorer_returns_callable(self, rubricset):
        """Test that create_rubricset_scorer returns a callable."""
        scorer = create_rubricset_scorer(rubricset)
        assert callable(scorer)

    def test_rubricset_scorer_with_type_detector(self, rubricset):
        """Test scorer with question type detector."""

        def detector(question):
            return "math" if "calculate" in question.lower() else "general"

        scorer = create_rubricset_scorer(rubricset, question_type_detector=detector)

        prompts = ["Calculate 2+2", "Explain why sky is blue"]
        completions = [
            "I calculate 2+2 equals 4",
            "The sky appears blue because of light scattering",
        ]

        scores = scorer(prompts, completions)

        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)


class TestCreateWeightedRubricScorer:
    """Tests for create_weighted_rubric_scorer function."""

    @pytest.fixture
    def weighted_rubric(self):
        """Create a rubric with weighted criteria."""
        return (
            RubricBuilder("weighted", "Weighted rubric")
            .with_criterion("high_weight", "Important", weight=3.0, keywords=["important", "key"])
            .with_criterion("low_weight", "Less important", weight=1.0, keywords=["minor"])
            .build()
        )

    def test_weighted_scorer_returns_callable(self, weighted_rubric):
        """Test that weighted scorer is callable."""
        scorer = create_weighted_rubric_scorer(weighted_rubric)
        assert callable(scorer)

    def test_weighted_scorer_respects_weights(self, weighted_rubric):
        """Test that higher weighted criteria have more impact."""
        scorer = create_weighted_rubric_scorer(weighted_rubric)

        prompts = ["test"] * 2
        completions = [
            "This is important and key",  # Matches high weight
            "This is minor",  # Matches low weight
        ]

        scores = scorer(prompts, completions)

        # High weight match should score higher
        assert scores[0] > scores[1]


class TestCreateMultiRubricScorer:
    """Tests for create_multi_rubric_scorer function."""

    @pytest.fixture
    def multi_rubricset(self):
        """Create a rubric set for multi-scoring."""
        r1 = (
            RubricBuilder("r1", "First")
            .with_criterion("c1", "C1", keywords=["alpha"])
            .build()
        )
        r2 = (
            RubricBuilder("r2", "Second")
            .with_criterion("c2", "C2", keywords=["beta"])
            .build()
        )
        return RubricSet(name="multi", rubrics=[r1, r2])

    def test_multi_scorer_max_aggregation(self, multi_rubricset):
        """Test max aggregation mode."""
        scorer = create_multi_rubric_scorer(multi_rubricset, aggregation="max")

        scores = scorer(["test"], ["alpha and beta"])

        assert len(scores) == 1
        assert scores[0] > 0

    def test_multi_scorer_mean_aggregation(self, multi_rubricset):
        """Test mean aggregation mode."""
        scorer = create_multi_rubric_scorer(multi_rubricset, aggregation="mean")

        scores = scorer(["test"], ["alpha"])

        assert len(scores) == 1

    def test_multi_scorer_sum_aggregation(self, multi_rubricset):
        """Test sum aggregation mode."""
        scorer = create_multi_rubric_scorer(multi_rubricset, aggregation="sum")

        scores = scorer(["test"], ["alpha beta"])

        assert len(scores) == 1


class TestRubricRewardAdapter:
    """Tests for rubric_reward_adapter function."""

    @pytest.fixture
    def rubric_with_reference(self):
        """Create a rubric with reference response."""
        return (
            RubricBuilder("ref_rubric", "Has reference")
            .with_criterion("format", "Format check", keywords=["<answer>"])
            .with_reference("<answer>42</answer>", target_score=10.0)
            .build()
        )

    def test_adapter_returns_callable(self, rubric_with_reference):
        """Test that adapter returns callable."""
        reward_fn = rubric_reward_adapter(rubric_with_reference)
        assert callable(reward_fn)

    def test_adapter_with_single_rubric(self, rubric_with_reference):
        """Test adapter with a single Rubric (not RubricSet)."""
        reward_fn = rubric_reward_adapter(rubric_with_reference)

        scores = reward_fn(
            prompts=["What is the answer?"],
            completions=["<answer>42</answer>"],
        )

        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_adapter_with_rubricset(self):
        """Test adapter with RubricSet."""
        rubricset = RubricSet(
            name="test",
            rubrics=[
                Rubric(
                    name="r1",
                    description="R1",
                    criteria=[Criterion(name="c1", description="C1")],
                )
            ],
        )
        reward_fn = rubric_reward_adapter(rubricset)

        scores = reward_fn(
            prompts=["test"],
            completions=["response"],
        )

        assert len(scores) == 1


class TestQuickTestRubric:
    """Tests for quick_test_rubric function."""

    @pytest.fixture
    def test_rubric(self):
        """Create a test rubric."""
        return (
            RubricBuilder("quick_test", "For quick testing")
            .with_criterion("format", "Has answer tag", keywords=["<answer>", "</answer>"])
            .build()
        )

    def test_quick_test_returns_dict(self, test_rubric):
        """Test that quick_test returns a dict."""
        result = quick_test_rubric(
            test_rubric,
            test_responses=["<answer>test</answer>", "no tags"],
        )

        assert isinstance(result, dict)
        assert "scores" in result
        assert "mean" in result
        assert "rubric_text" in result

    def test_quick_test_with_expected_scores(self, test_rubric):
        """Test quick_test with expected scores."""
        result = quick_test_rubric(
            test_rubric,
            test_responses=["<answer>test</answer>"],
            expected_scores=[5.0],
        )

        assert "expected" in result
        assert "differences" in result
        assert "max_diff" in result
        assert "passed" in result

    def test_quick_test_scores_correct_length(self, test_rubric):
        """Test that scores match input length."""
        responses = ["r1", "r2", "r3"]
        result = quick_test_rubric(test_rubric, test_responses=responses)

        assert len(result["scores"]) == 3


class TestTestRubricWithDataset:
    """Tests for test_rubric_with_dataset function."""

    @pytest.fixture
    def dataset(self):
        """Create a sample dataset."""
        return [
            {"question": "What is 2+2?", "response": "<answer>4</answer>"},
            {"question": "What is 3+3?", "response": "Six"},
            {"question": "What is 5+5?", "response": "<answer>10</answer>"},
        ]

    @pytest.fixture
    def test_rubric(self):
        """Create a test rubric."""
        return (
            RubricBuilder("dataset_test", "For dataset testing")
            .with_criterion("format", "Has tags", keywords=["<answer>"])
            .build()
        )

    def test_returns_result_object(self, test_rubric, dataset):
        """Test that function returns RubricTestResult."""
        result = test_rubric_with_dataset(test_rubric, dataset)

        assert isinstance(result, RubricTestResult)
        assert result.rubric_name == "dataset_test"

    def test_result_has_statistics(self, test_rubric, dataset):
        """Test that result has statistical fields."""
        result = test_rubric_with_dataset(test_rubric, dataset)

        assert hasattr(result, "mean_score")
        assert hasattr(result, "std_score")
        assert hasattr(result, "min_score")
        assert hasattr(result, "max_score")
        assert hasattr(result, "scores")
        assert len(result.scores) == 3

    def test_result_summary(self, test_rubric, dataset):
        """Test result summary method."""
        result = test_rubric_with_dataset(test_rubric, dataset)
        summary = result.summary()

        assert "dataset_test" in summary
        assert "Mean" in summary

    def test_config_limits_examples(self, test_rubric, dataset):
        """Test that config limits number of examples."""
        config = RubricTestConfig(num_examples=2)
        result = test_rubric_with_dataset(test_rubric, dataset, config)

        assert len(result.scores) == 2

    def test_verbose_includes_examples(self, test_rubric, dataset):
        """Test verbose mode includes examples."""
        config = RubricTestConfig(verbose=True, num_verbose_examples=2)
        result = test_rubric_with_dataset(test_rubric, dataset, config)

        assert len(result.examples) <= 2


class TestCreateGrpoRewardFunction:
    """Tests for create_grpo_reward_function."""

    @pytest.fixture
    def rubricset(self):
        """Create a test rubric set."""
        return RubricSet(
            name="grpo_test",
            rubrics=[
                (
                    RubricBuilder("r1", "R1")
                    .with_criterion("c1", "C1", keywords=["keyword"])
                    .build()
                )
            ],
        )

    def test_returns_callable(self, rubricset):
        """Test that function returns callable."""
        reward_fn = create_grpo_reward_function(rubricset)
        assert callable(reward_fn)

    def test_basic_scoring(self, rubricset):
        """Test basic reward scoring."""
        reward_fn = create_grpo_reward_function(rubricset)

        scores = reward_fn(
            prompts=["test"],
            completions=["response with keyword"],
        )

        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_with_weight(self, rubricset):
        """Test weighted scoring."""
        reward_fn = create_grpo_reward_function(rubricset, weight=0.5)

        scores = reward_fn(
            prompts=["test"],
            completions=["keyword"],
        )

        # Score should be scaled by weight
        assert len(scores) == 1

    def test_combine_with_other_functions(self, rubricset):
        """Test combining with other reward functions."""

        def constant_reward(prompts, completions, **kwargs):
            return [1.0] * len(completions)

        reward_fn = create_grpo_reward_function(
            rubricset,
            combine_with=[constant_reward],
            weight=1.0,
        )

        scores = reward_fn(
            prompts=["test"],
            completions=["keyword"],
        )

        # Should include both rubric score and constant reward
        assert scores[0] >= 1.0  # At least the constant reward


class TestCompareRubrics:
    """Tests for compare_rubrics function."""

    @pytest.fixture
    def rubrics_to_compare(self):
        """Create rubrics to compare."""
        return [
            (
                RubricBuilder("r1", "First")
                .with_criterion("c1", "C1", keywords=["first"])
                .build()
            ),
            (
                RubricBuilder("r2", "Second")
                .with_criterion("c2", "C2", keywords=["second"])
                .build()
            ),
        ]

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        return [
            {"question": "Q1", "response": "first response"},
            {"question": "Q2", "response": "second response"},
        ]

    def test_compare_returns_list(self, rubrics_to_compare, dataset):
        """Test that compare returns list of results."""
        results = compare_rubrics(rubrics_to_compare, dataset)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, RubricTestResult) for r in results)

    def test_compare_with_rubricset(self, dataset):
        """Test compare with RubricSet input."""
        rubricset = RubricSet(
            name="test",
            rubrics=[
                RubricBuilder("r1", "R1").with_criterion("c1", "C1").build(),
            ],
        )

        results = compare_rubrics(rubricset, dataset)

        assert len(results) == 1
