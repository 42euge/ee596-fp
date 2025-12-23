"""
Unit tests for rubric testing infrastructure

Tests cover:
- Rubric designer classes and interfaces
- Evaluation engine
- Comparison tools
- Reporter functionality
"""

import pytest
import numpy as np
from pathlib import Path
import json
import tempfile

from src.rubric_testing import (
    BaseRubric,
    RubricScore,
    RubricDesigner,
    FunctionRubric,
    CompositeRubric,
    WeightedRubric,
    create_rubric,
    KeywordMatchRubric,
    LengthRubric,
    FormatComplianceRubric,
    RubricEvaluator,
    EvaluationConfig,
    EvaluationResult,
    RubricComparator,
    ComparisonResult,
    compare_rubrics,
    RubricReporter,
)


class TestRubricScore:
    """Test RubricScore data class"""

    def test_rubric_score_creation(self):
        score = RubricScore(total=8.5, components={"a": 3.0, "b": 5.5})
        assert score.total == 8.5
        assert score.components["a"] == 3.0
        assert score.components["b"] == 5.5

    def test_rubric_score_repr(self):
        score = RubricScore(total=8.5, components={"keyword": 3.0, "format": 5.5})
        repr_str = repr(score)
        assert "8.50" in repr_str
        assert "keyword" in repr_str


class TestBaseRubric:
    """Test BaseRubric abstract class"""

    def test_base_rubric_properties(self):
        class SimpleRubric(BaseRubric):
            def score(self, prompt, completion, rubric, **kwargs):
                return RubricScore(total=5.0)

        rubric = SimpleRubric("test", weight=0.5)
        assert rubric.name == "test"
        assert rubric.weight == 0.5
        assert rubric.score_range == (0.0, 10.0)

    def test_normalize_score(self):
        class SimpleRubric(BaseRubric):
            def score(self, prompt, completion, rubric, **kwargs):
                return RubricScore(total=5.0)

        rubric = SimpleRubric("test")
        assert rubric.normalize_score(0.0) == 0.0
        assert rubric.normalize_score(10.0) == 1.0
        assert rubric.normalize_score(5.0) == 0.5


class TestRubricDesigner:
    """Test RubricDesigner registration system"""

    def test_register_function(self):
        designer = RubricDesigner()

        @designer.register("test_rubric")
        def my_rubric(prompt, completion, rubric, **kwargs):
            return RubricScore(total=7.0)

        rubric = designer.get("test_rubric")
        assert rubric is not None
        assert rubric.name == "test_rubric"

    def test_list_rubrics(self):
        designer = RubricDesigner()

        @designer.register("rubric1")
        def r1(**kwargs):
            return 1.0

        @designer.register("rubric2")
        def r2(**kwargs):
            return 2.0

        rubrics = designer.list_rubrics()
        assert "rubric1" in rubrics
        assert "rubric2" in rubrics


class TestFunctionRubric:
    """Test FunctionRubric wrapper"""

    def test_function_rubric_with_score(self):
        def scorer(prompt, completion, rubric, **kwargs):
            return RubricScore(total=8.0, components={"test": 8.0})

        rubric = FunctionRubric("func", scorer)
        result = rubric.score("prompt", "completion", "rubric")
        assert result.total == 8.0

    def test_function_rubric_with_float(self):
        def scorer(prompt, completion, rubric, **kwargs):
            return 6.5

        rubric = FunctionRubric("func", scorer)
        result = rubric.score("prompt", "completion", "rubric")
        assert result.total == 6.5

    def test_function_rubric_with_dict(self):
        def scorer(prompt, completion, rubric, **kwargs):
            return {"a": 3.0, "b": 4.0}

        rubric = FunctionRubric("func", scorer)
        result = rubric.score("prompt", "completion", "rubric")
        assert result.total == 7.0  # sum of components


class TestCompositeRubric:
    """Test CompositeRubric combination"""

    def test_composite_basic(self):
        rubric1 = FunctionRubric("r1", lambda **kw: 5.0)
        rubric2 = FunctionRubric("r2", lambda **kw: 3.0)

        composite = CompositeRubric("combo", [rubric1, rubric2], normalize=True)
        result = composite.score("p", "c", "r")

        # Normalized: (5/10) + (3/10) = 0.5 + 0.3 = 0.8
        assert abs(result.total - 0.8) < 0.01

    def test_composite_weighted(self):
        rubric1 = FunctionRubric("r1", lambda **kw: 10.0)
        rubric1.weight = 2.0

        rubric2 = FunctionRubric("r2", lambda **kw: 10.0)
        rubric2.weight = 1.0

        composite = CompositeRubric("combo", [rubric1, rubric2], normalize=True)
        result = composite.score("p", "c", "r")

        # r1 normalized: 1.0 * 2.0 = 2.0
        # r2 normalized: 1.0 * 1.0 = 1.0
        # total: 3.0
        assert abs(result.total - 3.0) < 0.01


class TestWeightedRubric:
    """Test WeightedRubric"""

    def test_weighted_rubric_normalization(self):
        r1 = FunctionRubric("r1", lambda **kw: 10.0)
        r2 = FunctionRubric("r2", lambda **kw: 10.0)

        weighted = WeightedRubric("weighted", {r1: 0.7, r2: 0.3})

        # Weights should sum to 1.0
        assert abs(r1.weight + r2.weight - 1.0) < 0.01
        assert abs(r1.weight - 0.7) < 0.01


class TestCreateRubric:
    """Test rubric factory function"""

    def test_create_function_rubric(self):
        rubric = create_rubric("test", score_func=lambda **kw: 5.0)
        assert isinstance(rubric, FunctionRubric)

    def test_create_composite_rubric(self):
        r1 = FunctionRubric("r1", lambda **kw: 5.0)
        r2 = FunctionRubric("r2", lambda **kw: 3.0)

        rubric = create_rubric("combo", components=[r1, r2])
        assert isinstance(rubric, CompositeRubric)

    def test_create_weighted_rubric(self):
        r1 = FunctionRubric("r1", lambda **kw: 5.0)
        r2 = FunctionRubric("r2", lambda **kw: 3.0)

        rubric = create_rubric("weighted", weights={r1: 0.6, r2: 0.4})
        assert isinstance(rubric, WeightedRubric)


class TestKeywordMatchRubric:
    """Test built-in KeywordMatchRubric"""

    def test_keyword_match_basic(self):
        rubric = KeywordMatchRubric()

        rubric_text = "solve equation quadratic formula"
        completion = "To solve this quadratic equation, we use the quadratic formula"

        result = rubric.score("", completion, rubric_text)

        # Should match "solve", "quadratic", "formula", "equation"
        assert result.total > 0
        assert result.components["matches"] > 0

    def test_keyword_match_case_insensitive(self):
        rubric = KeywordMatchRubric(case_sensitive=False)

        rubric_text = "SOLVE EQUATION"
        completion = "solve the equation"

        result = rubric.score("", completion, rubric_text)
        assert result.total > 0


class TestLengthRubric:
    """Test built-in LengthRubric"""

    def test_length_exact_match(self):
        rubric = LengthRubric(target_length=100, tolerance=0.1)

        completion = "x" * 100
        result = rubric.score("", completion, "")

        assert result.total == 10.0  # Perfect match
        assert result.components["length"] == 100

    def test_length_too_short(self):
        rubric = LengthRubric(target_length=100, tolerance=0.1)

        completion = "x" * 50  # 50% deviation
        result = rubric.score("", completion, "")

        assert result.total == 0.0  # Beyond tolerance


class TestFormatComplianceRubric:
    """Test built-in FormatComplianceRubric"""

    def test_format_with_all_tags(self):
        rubric = FormatComplianceRubric()

        completion = (
            "<reasoning>This is my reasoning with over 50 characters of content here</reasoning>"
            "<answer>42</answer>"
        )

        result = rubric.score("", completion, "")

        # Should get points for both tags and content
        assert result.total > 0
        assert result.components["has_reasoning_tags"] == 1.0
        assert result.components["has_answer_tags"] == 1.0

    def test_format_missing_tags(self):
        rubric = FormatComplianceRubric()

        completion = "Just plain text with no tags"
        result = rubric.score("", completion, "")

        assert result.total == 0.0


class TestEvaluationConfig:
    """Test EvaluationConfig dataclass"""

    def test_default_config(self):
        config = EvaluationConfig()

        assert config.num_samples == 100
        assert config.temperature == 0.9
        assert config.model_name == "google/gemma-3-1b-it"

    def test_custom_config(self):
        config = EvaluationConfig(
            num_samples=50,
            temperature=0.7,
            model_name="custom/model"
        )

        assert config.num_samples == 50
        assert config.temperature == 0.7
        assert config.model_name == "custom/model"


class TestEvaluationResult:
    """Test EvaluationResult dataclass"""

    def test_evaluation_result_summary(self):
        config = EvaluationConfig(num_samples=10)
        result = EvaluationResult(
            rubric_name="test",
            config=config,
            mean_score=7.5,
            std_score=1.2,
            num_samples=10,
        )

        summary = result.summary()
        assert summary["rubric_name"] == "test"
        assert summary["mean_score"] == 7.5
        assert summary["num_samples"] == 10

    def test_evaluation_result_save(self):
        config = EvaluationConfig()
        scores = [RubricScore(total=5.0), RubricScore(total=7.0)]
        result = EvaluationResult(
            rubric_name="test",
            config=config,
            scores=scores,
            mean_score=6.0,
        )

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            result.save(temp_path)

            with open(temp_path, 'r') as f:
                data = json.load(f)

            assert data["rubric_name"] == "test"
            assert len(data["scores"]) == 2
        finally:
            Path(temp_path).unlink()


class TestRubricEvaluator:
    """Test RubricEvaluator (without model loading)"""

    def test_evaluator_initialization(self):
        config = EvaluationConfig(num_samples=10)
        evaluator = RubricEvaluator(config)

        assert evaluator.config.num_samples == 10
        assert not evaluator._initialized

    def test_evaluate_with_provided_data(self):
        """Test evaluation with pre-provided data (no model needed)"""
        config = EvaluationConfig(num_samples=5)
        evaluator = RubricEvaluator(config)

        # Skip initialization since we provide all data
        prompts = ["Q1", "Q2", "Q3"]
        completions = ["A1", "A2", "A3"]
        rubrics = ["R1", "R2", "R3"]

        rubric = KeywordMatchRubric()

        # This should work without model/dataset
        result = evaluator.evaluate(
            rubric,
            prompts=prompts,
            completions=completions,
            rubrics=rubrics,
        )

        assert result.num_samples == 3
        assert len(result.scores) == 3
        assert result.mean_score >= 0


class TestRubricComparator:
    """Test RubricComparator"""

    def test_comparator_basic(self):
        # Create mock results
        config = EvaluationConfig()

        result1 = EvaluationResult(
            rubric_name="rubric1",
            config=config,
            scores=[RubricScore(total=i) for i in [5, 6, 7]],
            mean_score=6.0,
            std_score=1.0,
            num_samples=3,
        )

        result2 = EvaluationResult(
            rubric_name="rubric2",
            config=config,
            scores=[RubricScore(total=i) for i in [7, 8, 9]],
            mean_score=8.0,
            std_score=1.0,
            num_samples=3,
        )

        comparator = RubricComparator()
        comparison = comparator.compare([result1, result2])

        assert comparison.best_rubric == "rubric2"
        assert comparison.rankings["rubric2"] == 1
        assert comparison.rankings["rubric1"] == 2

    def test_statistical_tests(self):
        config = EvaluationConfig()

        # Create results with different means
        result1 = EvaluationResult(
            rubric_name="low",
            config=config,
            scores=[RubricScore(total=i) for i in [3, 4, 5]],
            mean_score=4.0,
            std_score=1.0,
            num_samples=3,
        )

        result2 = EvaluationResult(
            rubric_name="high",
            config=config,
            scores=[RubricScore(total=i) for i in [8, 9, 10]],
            mean_score=9.0,
            std_score=1.0,
            num_samples=3,
        )

        comparator = RubricComparator()
        comparison = comparator.compare([result1, result2])

        assert "pairwise_ttests" in comparison.statistical_tests
        assert "low vs high" in comparison.statistical_tests["pairwise_ttests"]


class TestCompareRubrics:
    """Test compare_rubrics convenience function"""

    def test_compare_rubrics_function(self):
        config = EvaluationConfig()

        result1 = EvaluationResult("r1", config, mean_score=5.0, num_samples=10)
        result2 = EvaluationResult("r2", config, mean_score=7.0, num_samples=10)

        comparison = compare_rubrics([result1, result2])

        assert comparison.best_rubric == "r2"


class TestRubricReporter:
    """Test RubricReporter"""

    def test_reporter_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = RubricReporter(output_dir=tmpdir)
            assert reporter.output_dir.exists()

    def test_generate_markdown_report(self):
        config = EvaluationConfig()
        result = EvaluationResult(
            rubric_name="test",
            config=config,
            mean_score=7.5,
            std_score=1.2,
            median_score=7.0,
            min_score=5.0,
            max_score=10.0,
            num_samples=10,
        )

        reporter = RubricReporter()
        report = reporter.generate_report([result], format="markdown")

        assert "# Rubric Evaluation Report" in report
        assert "test" in report
        assert "7.5" in report

    def test_generate_json_report(self):
        config = EvaluationConfig()
        result = EvaluationResult("test", config, mean_score=7.5, num_samples=10)

        reporter = RubricReporter()
        report_str = reporter.generate_report([result], format="json")

        report_data = json.loads(report_str)
        assert "summary" in report_data
        assert "results" in report_data

    def test_save_report(self):
        config = EvaluationConfig()
        result = EvaluationResult("test", config, mean_score=7.5, num_samples=10)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_path = f.name

        try:
            reporter = RubricReporter()
            reporter.generate_report([result], output_path=temp_path)

            assert Path(temp_path).exists()
            content = Path(temp_path).read_text()
            assert "# Rubric Evaluation Report" in content
        finally:
            Path(temp_path).unlink()


class TestIntegration:
    """Integration tests for full workflow"""

    def test_end_to_end_workflow(self):
        """Test complete rubric testing workflow"""
        # 1. Create custom rubrics
        rubric1 = KeywordMatchRubric("keywords")
        rubric2 = FormatComplianceRubric("format")

        # 2. Create evaluator
        config = EvaluationConfig(num_samples=5)
        evaluator = RubricEvaluator(config)

        # 3. Prepare test data
        prompts = ["Question " + str(i) for i in range(5)]
        completions = [
            "<reasoning>analysis</reasoning><answer>result</answer>"
            for _ in range(5)
        ]
        rubrics = ["test rubric analysis" for _ in range(5)]

        # 4. Evaluate rubrics
        results = []
        for rubric in [rubric1, rubric2]:
            result = evaluator.evaluate(
                rubric,
                prompts=prompts,
                completions=completions,
                rubrics=rubrics,
            )
            results.append(result)

        # 5. Compare results
        comparison = compare_rubrics(results)

        # 6. Generate report
        reporter = RubricReporter()
        report = reporter.generate_report(results, comparison)

        # Verify workflow completed
        assert len(results) == 2
        assert comparison.best_rubric in ["keywords", "format"]
        assert "# Rubric Evaluation Report" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
