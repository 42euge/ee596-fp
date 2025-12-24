"""
Semantic-preserving perturbation generators for robustness evaluation.

Implements perturbations that modify text while preserving meaning:
- Synonym replacement using WordNet
- Paraphrasing using T5 models
- Sentence reordering
"""

import re
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass

from .config import PerturbationConfig


# XML tags to preserve during perturbation
PROTECTED_TAGS = [
    ("<reasoning>", "</reasoning>"),
    ("<answer>", "</answer>"),
]


@dataclass
class PerturbedText:
    """Container for a perturbed text variant."""
    original: str
    perturbed: str
    perturbation_type: str
    changes: List[str]  # Description of changes made


class BasePerturbation(ABC):
    """Abstract base class for perturbation strategies."""

    @abstractmethod
    def generate(self, text: str, n: int = 5) -> List[PerturbedText]:
        """Generate n perturbed variants of the input text.

        Args:
            text: Original text to perturb
            n: Number of variants to generate

        Returns:
            List of PerturbedText objects
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this perturbation type."""
        pass


def extract_protected_regions(text: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    """Extract protected tag regions from text.

    Returns:
        Tuple of (text with placeholders, list of (placeholder, tag_name, content))
    """
    regions = []
    modified_text = text

    for start_tag, end_tag in PROTECTED_TAGS:
        pattern = rf"({re.escape(start_tag)})(.*?)({re.escape(end_tag)})"
        matches = list(re.finditer(pattern, modified_text, re.DOTALL))

        for i, match in enumerate(matches):
            placeholder = f"__PROTECTED_{start_tag[1:-1]}_{i}__"
            full_match = match.group(0)
            regions.append((placeholder, start_tag, match.group(2), end_tag))
            modified_text = modified_text.replace(full_match, placeholder, 1)

    return modified_text, regions


def restore_protected_regions(
    text: str, regions: List[Tuple[str, str, str, str]]
) -> str:
    """Restore protected regions from placeholders."""
    for placeholder, start_tag, content, end_tag in regions:
        text = text.replace(placeholder, f"{start_tag}{content}{end_tag}")
    return text


class SynonymPerturbation(BasePerturbation):
    """Replace words with synonyms using WordNet."""

    def __init__(
        self,
        probability: float = 0.3,
        max_replacements: int = 5,
        seed: int = 42,
        preserve_tags: bool = True,
    ):
        self.probability = probability
        self.max_replacements = max_replacements
        self.seed = seed
        self.preserve_tags = preserve_tags
        self._wordnet = None
        self._stopwords: Set[str] = set()

    def _load_wordnet(self):
        """Lazy load WordNet resources."""
        if self._wordnet is not None:
            return

        try:
            import nltk
            from nltk.corpus import wordnet, stopwords

            # Download required data if not present
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)

            try:
                stopwords.words("english")
            except LookupError:
                nltk.download("stopwords", quiet=True)

            self._wordnet = wordnet
            self._stopwords = set(stopwords.words("english"))

        except ImportError:
            raise ImportError(
                "nltk is required for synonym perturbation. "
                "Install with: pip install nltk"
            )

    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word from WordNet."""
        self._load_wordnet()

        synonyms = set()
        for syn in self._wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)

        return list(synonyms)

    def _is_replaceable(self, word: str) -> bool:
        """Check if a word is eligible for replacement."""
        # Skip short words, numbers, stopwords
        if len(word) < 3:
            return False
        if word.lower() in self._stopwords:
            return False
        if word.isdigit() or re.match(r"^[\d.,]+$", word):
            return False
        if not word.isalpha():
            return False
        return True

    def generate(self, text: str, n: int = 5) -> List[PerturbedText]:
        """Generate synonym-replaced variants."""
        self._load_wordnet()
        random.seed(self.seed)

        results = []

        # Extract protected regions
        if self.preserve_tags:
            working_text, regions = extract_protected_regions(text)
        else:
            working_text = text
            regions = []

        # Tokenize (simple whitespace + punctuation aware)
        words = re.findall(r"\b\w+\b", working_text)
        replaceable_words = [
            (w, self._get_synonyms(w))
            for w in words
            if self._is_replaceable(w) and self._get_synonyms(w)
        ]

        for variant_idx in range(n):
            random.seed(self.seed + variant_idx)
            perturbed = working_text
            changes = []

            # Select words to replace
            num_to_replace = min(
                self.max_replacements,
                int(len(replaceable_words) * self.probability) + 1,
            )
            if replaceable_words:
                selected = random.sample(
                    replaceable_words,
                    min(num_to_replace, len(replaceable_words)),
                )

                for original_word, synonyms in selected:
                    if synonyms:
                        replacement = random.choice(synonyms)
                        # Preserve capitalization
                        if original_word[0].isupper():
                            replacement = replacement.capitalize()
                        if original_word.isupper():
                            replacement = replacement.upper()

                        # Replace only first occurrence to avoid over-replacement
                        pattern = rf"\b{re.escape(original_word)}\b"
                        perturbed = re.sub(pattern, replacement, perturbed, count=1)
                        changes.append(f"{original_word} -> {replacement}")

            # Restore protected regions
            if self.preserve_tags:
                perturbed = restore_protected_regions(perturbed, regions)

            results.append(
                PerturbedText(
                    original=text,
                    perturbed=perturbed,
                    perturbation_type=self.name,
                    changes=changes,
                )
            )

        return results

    @property
    def name(self) -> str:
        return "synonym"


class ParaphrasePerturbation(BasePerturbation):
    """Paraphrase text using T5-based models."""

    def __init__(
        self,
        model_id: str = "humarin/chatgpt_paraphraser_on_T5_base",
        device: str = "auto",
        max_length: int = 256,
        num_beams: int = 5,
        temperature: float = 0.7,
        preserve_tags: bool = True,
    ):
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.num_beams = num_beams
        self.temperature = temperature
        self.preserve_tags = preserve_tags
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load the paraphrase model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for paraphrase perturbation. "
                "Install with: pip install transformers torch"
            )

        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.device

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(device)
        self._device = device

    def _paraphrase(self, text: str, n: int = 1) -> List[str]:
        """Generate n paraphrases of the text."""
        self._load_model()

        # Prepare input
        input_text = f"paraphrase: {text}"
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self._device)

        # Generate
        outputs = self._model.generate(
            **inputs,
            max_length=self.max_length,
            num_beams=self.num_beams,
            num_return_sequences=n,
            temperature=self.temperature,
            do_sample=True,
        )

        paraphrases = [
            self._tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return paraphrases

    def generate(self, text: str, n: int = 5) -> List[PerturbedText]:
        """Generate paraphrased variants."""
        results = []

        if self.preserve_tags:
            # Extract and paraphrase only the content between tags
            working_text, regions = extract_protected_regions(text)

            # If there are protected regions, paraphrase the reasoning content
            if regions:
                # Find reasoning region
                reasoning_region = None
                for i, (placeholder, start_tag, content, end_tag) in enumerate(regions):
                    if "reasoning" in start_tag.lower():
                        reasoning_region = (i, content)
                        break

                if reasoning_region:
                    idx, content = reasoning_region
                    paraphrases = self._paraphrase(content.strip(), n)

                    for paraphrase in paraphrases:
                        # Update the reasoning content
                        new_regions = list(regions)
                        placeholder, start_tag, _, end_tag = new_regions[idx]
                        new_regions[idx] = (placeholder, start_tag, f"\n{paraphrase}\n", end_tag)

                        perturbed = restore_protected_regions(working_text, new_regions)
                        results.append(
                            PerturbedText(
                                original=text,
                                perturbed=perturbed,
                                perturbation_type=self.name,
                                changes=["paraphrased reasoning section"],
                            )
                        )
                    return results

            # Fallback: paraphrase the whole text
            paraphrases = self._paraphrase(working_text, n)
            for paraphrase in paraphrases:
                perturbed = restore_protected_regions(paraphrase, regions)
                results.append(
                    PerturbedText(
                        original=text,
                        perturbed=perturbed,
                        perturbation_type=self.name,
                        changes=["paraphrased full text"],
                    )
                )
        else:
            paraphrases = self._paraphrase(text, n)
            for paraphrase in paraphrases:
                results.append(
                    PerturbedText(
                        original=text,
                        perturbed=paraphrase,
                        perturbation_type=self.name,
                        changes=["paraphrased full text"],
                    )
                )

        return results

    @property
    def name(self) -> str:
        return "paraphrase"


class SentenceReorderPerturbation(BasePerturbation):
    """Reorder sentences while preserving semantic meaning."""

    def __init__(
        self,
        preserve_first: bool = True,
        preserve_last: bool = True,
        preserve_tags: bool = True,
        seed: int = 42,
    ):
        self.preserve_first = preserve_first
        self.preserve_last = preserve_last
        self.preserve_tags = preserve_tags
        self.seed = seed

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on common terminators
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def generate(self, text: str, n: int = 5) -> List[PerturbedText]:
        """Generate sentence-reordered variants."""
        random.seed(self.seed)
        results = []

        if self.preserve_tags:
            working_text, regions = extract_protected_regions(text)

            # Find reasoning content to reorder
            for i, (placeholder, start_tag, content, end_tag) in enumerate(regions):
                if "reasoning" in start_tag.lower():
                    sentences = self._split_sentences(content.strip())

                    if len(sentences) < 3:
                        # Not enough sentences to reorder meaningfully
                        continue

                    for variant_idx in range(n):
                        random.seed(self.seed + variant_idx)

                        # Determine which sentences to shuffle
                        if self.preserve_first and self.preserve_last:
                            first = sentences[0]
                            last = sentences[-1]
                            middle = sentences[1:-1]
                            random.shuffle(middle)
                            reordered = [first] + middle + [last]
                        elif self.preserve_first:
                            first = sentences[0]
                            rest = sentences[1:]
                            random.shuffle(rest)
                            reordered = [first] + rest
                        elif self.preserve_last:
                            last = sentences[-1]
                            rest = sentences[:-1]
                            random.shuffle(rest)
                            reordered = rest + [last]
                        else:
                            reordered = sentences[:]
                            random.shuffle(reordered)

                        new_content = " ".join(reordered)

                        new_regions = list(regions)
                        placeholder, start_tag, _, end_tag = new_regions[i]
                        new_regions[i] = (placeholder, start_tag, f"\n{new_content}\n", end_tag)

                        perturbed = restore_protected_regions(working_text, new_regions)
                        results.append(
                            PerturbedText(
                                original=text,
                                perturbed=perturbed,
                                perturbation_type=self.name,
                                changes=[f"reordered {len(sentences)} sentences"],
                            )
                        )

                    return results

        # Fallback: reorder sentences in the whole text
        sentences = self._split_sentences(text)

        if len(sentences) < 3:
            # Return original if not enough sentences
            return [
                PerturbedText(
                    original=text,
                    perturbed=text,
                    perturbation_type=self.name,
                    changes=["no changes (too few sentences)"],
                )
            ]

        for variant_idx in range(n):
            random.seed(self.seed + variant_idx)

            if self.preserve_first and self.preserve_last:
                first = sentences[0]
                last = sentences[-1]
                middle = sentences[1:-1]
                random.shuffle(middle)
                reordered = [first] + middle + [last]
            else:
                reordered = sentences[:]
                random.shuffle(reordered)

            perturbed = " ".join(reordered)
            results.append(
                PerturbedText(
                    original=text,
                    perturbed=perturbed,
                    perturbation_type=self.name,
                    changes=[f"reordered {len(sentences)} sentences"],
                )
            )

        return results

    @property
    def name(self) -> str:
        return "reorder"


class PerturbationPipeline:
    """Compose multiple perturbation strategies."""

    def __init__(self, config: Optional[PerturbationConfig] = None):
        self.config = config or PerturbationConfig()
        self._perturbations: List[BasePerturbation] = []
        self._setup_perturbations()

    def _setup_perturbations(self):
        """Initialize perturbation strategies based on config."""
        for ptype in self.config.enabled_types:
            if ptype == "synonym":
                self._perturbations.append(
                    SynonymPerturbation(
                        probability=self.config.synonym_probability,
                        max_replacements=self.config.synonym_max_replacements,
                        seed=self.config.seed,
                        preserve_tags=self.config.preserve_tags,
                    )
                )
            elif ptype == "paraphrase":
                self._perturbations.append(
                    ParaphrasePerturbation(
                        model_id=self.config.paraphrase_model,
                        max_length=self.config.paraphrase_max_length,
                        num_beams=self.config.paraphrase_num_beams,
                        temperature=self.config.paraphrase_temperature,
                        preserve_tags=self.config.preserve_tags,
                    )
                )
            elif ptype == "reorder":
                self._perturbations.append(
                    SentenceReorderPerturbation(
                        preserve_first=self.config.reorder_preserve_first,
                        preserve_last=self.config.reorder_preserve_last,
                        preserve_tags=self.config.preserve_tags,
                        seed=self.config.seed,
                    )
                )

    def generate_variants(
        self, text: str, n_per_type: Optional[int] = None
    ) -> List[PerturbedText]:
        """Generate perturbed variants using all enabled strategies.

        Args:
            text: Original text to perturb
            n_per_type: Number of variants per perturbation type.
                        If None, uses config.num_variants.

        Returns:
            List of all perturbed variants from all strategies
        """
        n = n_per_type or self.config.num_variants
        all_variants = []

        for perturbation in self._perturbations:
            try:
                variants = perturbation.generate(text, n)
                all_variants.extend(variants)
            except Exception as e:
                print(f"Warning: {perturbation.name} perturbation failed: {e}")
                continue

        return all_variants

    @property
    def perturbation_names(self) -> List[str]:
        """Get names of all enabled perturbations."""
        return [p.name for p in self._perturbations]
