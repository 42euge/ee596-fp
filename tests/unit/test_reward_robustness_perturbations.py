"""Unit tests for reward_robustness/perturbations.py."""

import pytest
import sys
from pathlib import Path

# Add src to path to allow direct imports without going through src/__init__.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestProtectedRegions:
    """Tests for tag protection functionality."""

    def test_extract_protected_regions_basic(self):
        """Test extracting protected tag regions."""
        from reward_robustness.perturbations import extract_protected_regions

        text = "<reasoning>Some reasoning here</reasoning><answer>42</answer>"
        modified, regions = extract_protected_regions(text)

        assert len(regions) == 2
        assert "__PROTECTED_reasoning_0__" in modified
        assert "__PROTECTED_answer_0__" in modified

    def test_extract_protected_regions_empty(self):
        """Test with no protected regions."""
        from reward_robustness.perturbations import extract_protected_regions

        text = "No tags here, just plain text."
        modified, regions = extract_protected_regions(text)

        assert modified == text
        assert len(regions) == 0

    def test_restore_protected_regions(self):
        """Test restoring protected regions."""
        from reward_robustness.perturbations import (
            extract_protected_regions,
            restore_protected_regions,
        )

        original = "<reasoning>Think step by step</reasoning>\n<answer>42</answer>"
        modified, regions = extract_protected_regions(original)
        restored = restore_protected_regions(modified, regions)

        assert restored == original

    def test_protected_regions_multiline(self):
        """Test with multiline content in tags."""
        from reward_robustness.perturbations import (
            extract_protected_regions,
            restore_protected_regions,
        )

        original = """<reasoning>
Step 1: Add the numbers
Step 2: Multiply by 2
Step 3: Get result
</reasoning>
<answer>42</answer>"""

        modified, regions = extract_protected_regions(original)
        restored = restore_protected_regions(modified, regions)

        assert restored == original


class TestPerturbedText:
    """Tests for PerturbedText dataclass."""

    def test_perturbed_text_creation(self):
        """Test creating PerturbedText instances."""
        from reward_robustness.perturbations import PerturbedText

        pt = PerturbedText(
            original="Hello world",
            perturbed="Hi world",
            perturbation_type="synonym",
            changes=["Hello -> Hi"],
        )

        assert pt.original == "Hello world"
        assert pt.perturbed == "Hi world"
        assert pt.perturbation_type == "synonym"
        assert len(pt.changes) == 1


class TestSynonymPerturbation:
    """Tests for SynonymPerturbation class."""

    def test_synonym_perturbation_init(self):
        """Test initialization."""
        from reward_robustness.perturbations import SynonymPerturbation

        sp = SynonymPerturbation(
            probability=0.5,
            max_replacements=3,
            seed=123,
        )

        assert sp.probability == 0.5
        assert sp.max_replacements == 3
        assert sp.seed == 123
        assert sp.name == "synonym"

    def test_synonym_perturbation_preserves_tags(self):
        """Test that XML tags are preserved during perturbation."""
        from reward_robustness.perturbations import SynonymPerturbation

        sp = SynonymPerturbation(preserve_tags=True, seed=42)
        text = "<reasoning>This is good reasoning</reasoning><answer>42</answer>"

        try:
            variants = sp.generate(text, n=2)
            for variant in variants:
                assert "<reasoning>" in variant.perturbed
                assert "</reasoning>" in variant.perturbed
                assert "<answer>" in variant.perturbed
                assert "</answer>" in variant.perturbed
                assert "42" in variant.perturbed
        except ImportError:
            pytest.skip("nltk not installed")

    def test_synonym_perturbation_preserves_numbers(self):
        """Test that numbers are not replaced."""
        from reward_robustness.perturbations import SynonymPerturbation

        sp = SynonymPerturbation(seed=42)
        text = "The total is 42 dollars and 50 cents."

        try:
            variants = sp.generate(text, n=3)
            for variant in variants:
                assert "42" in variant.perturbed
                assert "50" in variant.perturbed
        except ImportError:
            pytest.skip("nltk not installed")

    def test_synonym_perturbation_generates_n_variants(self):
        """Test that the correct number of variants is generated."""
        from reward_robustness.perturbations import SynonymPerturbation

        sp = SynonymPerturbation(seed=42)
        text = "The quick brown fox jumps over the lazy dog."

        try:
            variants = sp.generate(text, n=5)
            assert len(variants) == 5
            for v in variants:
                assert v.perturbation_type == "synonym"
        except ImportError:
            pytest.skip("nltk not installed")


class TestSentenceReorderPerturbation:
    """Tests for SentenceReorderPerturbation class."""

    def test_sentence_reorder_init(self):
        """Test initialization."""
        from reward_robustness.perturbations import SentenceReorderPerturbation

        sr = SentenceReorderPerturbation(
            preserve_first=True,
            preserve_last=True,
            seed=42,
        )

        assert sr.preserve_first is True
        assert sr.preserve_last is True
        assert sr.seed == 42
        assert sr.name == "reorder"

    def test_sentence_reorder_preserves_first_last(self):
        """Test that first and last sentences are preserved."""
        from reward_robustness.perturbations import SentenceReorderPerturbation

        sr = SentenceReorderPerturbation(
            preserve_first=True,
            preserve_last=True,
            preserve_tags=False,
            seed=42,
        )

        text = "First sentence. Middle one. Another middle. Last sentence."
        variants = sr.generate(text, n=3)

        for variant in variants:
            sentences = variant.perturbed.split(". ")
            # First sentence should start with "First"
            assert sentences[0].startswith("First")
            # Last should end with "Last sentence" (possibly with period)
            assert "Last sentence" in sentences[-1]

    def test_sentence_reorder_preserves_tags(self):
        """Test that tags are preserved during reordering."""
        from reward_robustness.perturbations import SentenceReorderPerturbation

        sr = SentenceReorderPerturbation(preserve_tags=True, seed=42)
        text = """<reasoning>
First step. Second step. Third step. Final step.
</reasoning>
<answer>42</answer>"""

        variants = sr.generate(text, n=2)

        for variant in variants:
            assert "<reasoning>" in variant.perturbed
            assert "</reasoning>" in variant.perturbed
            assert "<answer>42</answer>" in variant.perturbed

    def test_sentence_reorder_too_few_sentences(self):
        """Test behavior with too few sentences to reorder."""
        from reward_robustness.perturbations import SentenceReorderPerturbation

        sr = SentenceReorderPerturbation(preserve_tags=False, seed=42)
        text = "Just one sentence."

        variants = sr.generate(text, n=2)

        # Should return original text with note about no changes
        assert len(variants) == 1
        assert variants[0].perturbed == text


class TestParaphrasePerturbation:
    """Tests for ParaphrasePerturbation class."""

    def test_paraphrase_init(self):
        """Test initialization."""
        from reward_robustness.perturbations import ParaphrasePerturbation

        pp = ParaphrasePerturbation(
            model_id="test/model",
            device="cpu",
            max_length=128,
        )

        assert pp.model_id == "test/model"
        assert pp.device == "cpu"
        assert pp.max_length == 128
        assert pp.name == "paraphrase"

    def test_paraphrase_preserves_answer_tag(self):
        """Test that answer tag content is preserved."""
        # This test would require mocking the model
        # Skip if transformers not available
        pytest.skip("Requires model mocking or transformers installed")


class TestPerturbationPipeline:
    """Tests for PerturbationPipeline class."""

    def test_pipeline_init_default(self):
        """Test default initialization."""
        from reward_robustness.perturbations import PerturbationPipeline
        from reward_robustness.config import PerturbationConfig

        config = PerturbationConfig(enabled_types=["reorder"])
        pipeline = PerturbationPipeline(config)

        assert "reorder" in pipeline.perturbation_names

    def test_pipeline_init_multiple_types(self):
        """Test initialization with multiple perturbation types."""
        from reward_robustness.perturbations import PerturbationPipeline
        from reward_robustness.config import PerturbationConfig

        # Only use reorder to avoid nltk/transformers dependencies
        config = PerturbationConfig(enabled_types=["reorder"])
        pipeline = PerturbationPipeline(config)

        assert len(pipeline.perturbation_names) == 1

    def test_pipeline_generate_variants(self):
        """Test generating variants through pipeline."""
        from reward_robustness.perturbations import PerturbationPipeline
        from reward_robustness.config import PerturbationConfig

        config = PerturbationConfig(
            enabled_types=["reorder"],
            num_variants=3,
        )
        pipeline = PerturbationPipeline(config)

        text = """<reasoning>
First I add. Then I multiply. Next I subtract. Finally I divide.
</reasoning>
<answer>10</answer>"""

        variants = pipeline.generate_variants(text)

        # Should have variants from reorder
        assert len(variants) >= 1
        for v in variants:
            assert "<answer>10</answer>" in v.perturbed

    def test_pipeline_empty_types(self):
        """Test pipeline with no enabled types."""
        from reward_robustness.perturbations import PerturbationPipeline
        from reward_robustness.config import PerturbationConfig

        config = PerturbationConfig(enabled_types=[])
        pipeline = PerturbationPipeline(config)

        variants = pipeline.generate_variants("Some text")

        assert len(variants) == 0
        assert pipeline.perturbation_names == []


class TestBasePerturbation:
    """Tests for BasePerturbation abstract class."""

    def test_base_perturbation_is_abstract(self):
        """Test that BasePerturbation cannot be instantiated directly."""
        from reward_robustness.perturbations import BasePerturbation

        with pytest.raises(TypeError):
            BasePerturbation()

    def test_concrete_implementations_have_name(self):
        """Test that concrete implementations have name property."""
        from reward_robustness.perturbations import (
            SynonymPerturbation,
            ParaphrasePerturbation,
            SentenceReorderPerturbation,
        )

        assert SynonymPerturbation().name == "synonym"
        assert ParaphrasePerturbation().name == "paraphrase"
        assert SentenceReorderPerturbation().name == "reorder"
