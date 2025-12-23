# Research to Platform Capabilities: RL Rewards for LLMs

**Date**: December 2025
**Purpose**: Translate latest research on RL rewards for LLMs into actionable platform capabilities

---

## Executive Summary

This document synthesizes cutting-edge research on reward modeling for LLM reinforcement learning (2024-2025) and maps scientific requirements to platform capabilities for our GRPO-based training system.

**Key Finding**: The field is rapidly evolving from simple reward models toward:
- Multi-faceted reward functions combining multiple signals
- Dynamic reward adaptation during training
- Verification-based rewards for factual grounding
- Inference-time reward scaling for better generalization
- Token-level vs response-level reward tradeoffs

---

## 1. Latest Research Findings

### 1.1 Core Methods Evolution

#### DPO vs PPO (2024-2025)
**Finding**: Direct Preference Optimization (DPO) eliminates explicit reward models but shows fundamental limitations compared to PPO in complex tasks.

**Key Results**:
- DPO: 61% win rate, simpler implementation, no RL training loop
- PPO: 57% win rate but achieves SOTA on code competitions
- DPO struggles with: out-of-distribution generalization, complex multi-step reasoning
- PPO excels when: proper hyperparameter tuning, sufficient compute available

**Platform Implication**: Support both DPO and PPO pipelines with easy switching based on task complexity.

#### GRPO Current Status
Your platform uses GRPO (Group Relative Policy Optimization), which:
- Shares PPO's advantages (explicit reward optimization)
- Adds group-based normalization for stability
- Aligns well with recent research emphasizing explicit reward models

---

### 1.2 Advanced Reward Modeling Techniques

#### Response-Level vs Token-Level Rewards (2025)
**Research**: [arXiv:2506.02553] "Response-Level Rewards Are All You Need"

**Finding**: Trajectory Policy Gradient Theorem shows that:
- Token-level rewards can be unbiasedly estimated using only response-level rewards
- No need for expensive per-token reward annotation
- Reduces reward model complexity significantly

**Current Platform**: Uses response-level rewards ✓
**Enhancement Needed**: Add theoretical grounding and variance reduction techniques

#### Reinforcement Learning with Verifiable Rewards (RLVR) (2025)
**Research**: Multiple 2025 papers on verifiable outcomes

**Finding**:
- Rewards should be grounded in verifiable outcomes (code execution, math validation)
- Reduces reward hacking and improves factual accuracy
- Particularly effective for reasoning tasks

**Current Platform**: Uses format compliance + answer matching ✓
**Enhancement Needed**: Add external verification systems (code execution, symbolic math)

---

### 1.3 Dynamic and Adaptive Rewards

#### CARD Framework (2024)
**Research**: "A Large Language Model-Driven Reward Design Framework via Dynamic Feedback"

**Components**:
1. **Coder**: Generates reward function code iteratively
2. **Evaluator**: Provides dynamic feedback on reward quality
3. **Trajectory Preference Evaluation (TPE)**: Evaluates reward functions based on trajectory preferences without full RL training

**Innovation**: LLM-generated reward functions that evolve during training

**Current Platform**: Static reward functions
**Enhancement Needed**: Meta-reward system that adapts during training

#### RLTHF - Targeted Human Feedback (2025)
**Research**: Selective human correction based on reward model uncertainty

**Finding**:
- Identify hard-to-annotate samples using reward distribution
- Achieve 93-94% of full human annotation quality with only 6-7% of effort
- Focus human feedback where model is most uncertain

**Current Platform**: No human feedback loop
**Enhancement Needed**: Active learning system for human-in-the-loop rewards

---

### 1.4 Multi-Objective Reward Design

#### Contrastive Learning for Reward Models (2024)
**Research**: "Secrets of RLHF Part II: Reward Modeling"

**Techniques**:
- Contrastive learning enhances distinction between chosen/rejected responses
- Meta-learning enables generalization to out-of-distribution samples
- Addresses incorrect and ambiguous preference pairs

**Current Platform**: Simple scalar rewards
**Enhancement Needed**: Contrastive reward training pipeline

#### Inference-Time Reward Scaling (2025)
**Research**: DeepSeek-GRM and generalist reward models

**Finding**:
- Reward models improve with more inference compute
- Use ensemble or iterative refinement at inference time
- Better calibration and generalization

**Current Platform**: Single forward pass rewards
**Enhancement Needed**: Ensemble reward evaluation option

---

## 2. Scientific Requirements from Research

Based on the research, modern RL reward systems for LLMs need:

### 2.1 Core Requirements

| Requirement | Research Basis | Priority |
|-------------|----------------|----------|
| **Multi-signal reward composition** | CARD, RLHF surveys | High |
| **Verifiable outcome integration** | RLVR 2025 | High |
| **Response-level reward efficiency** | Response-level theorem | Medium |
| **Dynamic reward adaptation** | CARD framework | Medium |
| **Human-in-the-loop selective feedback** | RLTHF 2025 | Low |
| **Contrastive reward training** | RLHF Part II | Medium |
| **Inference-time reward scaling** | DeepSeek-GRM | Low |

### 2.2 Architectural Requirements

1. **Modular Reward Pipeline**
   - Composable reward functions
   - Easy addition/removal of reward signals
   - Weight tuning per reward component

2. **Verification Systems**
   - Code execution sandbox
   - Symbolic math solver
   - Fact-checking API integration

3. **Feedback Collection**
   - Uncertainty quantification
   - Active learning sample selection
   - Human annotation interface

4. **Monitoring & Debugging**
   - Per-reward-component logging
   - Reward distribution tracking
   - Reward hacking detection

---

## 3. Platform Capability Mapping

### 3.1 Current Platform Capabilities ✓

Your platform already has:

1. **GRPO Training Loop** (`scripts/train_grpo.py`)
   - Group-based policy optimization
   - KL divergence penalty (beta=0.08)
   - LoRA efficient fine-tuning

2. **Basic Reward Functions** (`TunRex/src/tunrex/datasets/rewards.py`)
   - Format compliance checking
   - Answer correctness (exact and approximate)
   - Numerical answer extraction

3. **Response-Level Rewards**
   - Aligns with 2025 research on response-level efficiency
   - Simple, fast evaluation

4. **Modular Architecture**
   - Separate reward module
   - Config-driven hyperparameters
   - Easy reward function swapping

### 3.2 Capability Gaps and Enhancements

#### Priority 1: Multi-Signal Reward Composition

**Current State**: Single reward per function, manually combined
**Research Need**: Weighted combination of multiple reward signals

**Enhancement**:
```python
# New: CompositeReward class
class CompositeReward:
    def __init__(self, reward_functions, weights):
        self.rewards = reward_functions
        self.weights = weights

    def __call__(self, prompts, completions, **kwargs):
        scores = []
        for reward_fn, weight in zip(self.rewards, self.weights):
            component_scores = reward_fn(prompts, completions, **kwargs)
            scores.append([s * weight for s in component_scores])
        # Combine all signals
        return [sum(s) for s in zip(*scores)]
```

**Platform Capability**: `RewardCompositor` with configurable weights

---

#### Priority 2: Verification-Based Rewards

**Current State**: String matching and numerical comparison
**Research Need**: External verification (code execution, symbolic math)

**Enhancement**:
```python
# New: Verifiable reward functions
def verify_code_execution(prompts, completions, test_cases, **kwargs):
    """Execute code and verify against test cases."""
    scores = []
    for completion in completions:
        code = extract_code(completion)
        passed = run_test_cases(code, test_cases)
        scores.append(3.0 * (passed / len(test_cases)))
    return scores

def verify_math_symbolic(prompts, completions, answer, **kwargs):
    """Verify mathematical equivalence symbolically."""
    scores = []
    for completion in completions:
        expr = extract_math_expression(completion)
        is_equivalent = symbolic_equal(expr, answer)
        scores.append(3.0 if is_equivalent else 0.0)
    return scores
```

**Platform Capability**: `VerificationRewards` module with:
- Safe code execution sandbox
- SymPy symbolic math integration
- External API verification (Wolfram Alpha, etc.)

---

#### Priority 3: Dynamic Reward Adaptation

**Current State**: Static reward functions throughout training
**Research Need**: Rewards that adapt based on model progress

**Enhancement**:
```python
# New: Adaptive reward wrapper
class AdaptiveReward:
    def __init__(self, base_reward, adaptation_strategy):
        self.base_reward = base_reward
        self.strategy = adaptation_strategy
        self.step = 0
        self.history = []

    def __call__(self, prompts, completions, **kwargs):
        scores = self.base_reward(prompts, completions, **kwargs)

        # Track distribution
        self.history.append({
            'step': self.step,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'distribution': scores
        })

        # Adapt based on strategy
        adapted_scores = self.strategy.adapt(scores, self.history)
        self.step += 1

        return adapted_scores

# Example strategies
class CurriculumStrategy:
    """Gradually increase difficulty/strictness."""

class TemperatureAnnealing:
    """Adjust reward sharpness over time."""

class DifficultyScaling:
    """Scale rewards based on task difficulty."""
```

**Platform Capability**: `AdaptiveRewardSystem` with logging and visualization

---

#### Priority 4: Reward Component Monitoring

**Current State**: Minimal reward logging
**Research Need**: Comprehensive reward analysis for debugging

**Enhancement**:
```python
# New: Reward monitoring system
class RewardMonitor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics = defaultdict(list)

    def log_reward_components(self, step, component_scores):
        """Log individual reward component scores."""
        for component_name, scores in component_scores.items():
            self.metrics[component_name].append({
                'step': step,
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'histogram': np.histogram(scores)
            })

    def detect_reward_hacking(self, threshold=0.9):
        """Detect potential reward hacking via distribution shifts."""
        # Check for sudden spikes, mode collapse, etc.
        pass

    def export_wandb(self):
        """Export metrics to Weights & Biases."""
        pass
```

**Platform Capability**: Integrated W&B dashboard for reward analysis

---

#### Priority 5: Contrastive Reward Learning

**Current State**: No contrastive training
**Research Need**: Better preference learning through contrastive objectives

**Enhancement**:
```python
# New: Contrastive reward model training
class ContrastiveRewardModel:
    """
    Train reward model to maximize margin between
    chosen and rejected responses.
    """

    def __init__(self, base_model):
        self.model = base_model
        self.margin = 0.5

    def contrastive_loss(self, chosen_rewards, rejected_rewards):
        """
        Maximize: r(chosen) - r(rejected) - margin
        """
        return torch.max(
            torch.zeros_like(chosen_rewards),
            self.margin - (chosen_rewards - rejected_rewards)
        ).mean()

    def train_step(self, chosen_samples, rejected_samples):
        chosen_rewards = self.model(chosen_samples)
        rejected_rewards = self.model(rejected_samples)
        loss = self.contrastive_loss(chosen_rewards, rejected_rewards)
        return loss
```

**Platform Capability**: Optional contrastive pre-training for reward models

---

## 4. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Modular reward infrastructure

- [ ] Implement `CompositeReward` class
- [ ] Create reward function registry
- [ ] Add per-component logging
- [ ] Build reward configuration YAML system

**Deliverable**: Configurable multi-reward system

---

### Phase 2: Verification Systems (Weeks 3-4)
**Goal**: Grounded, verifiable rewards

- [ ] Build safe code execution sandbox
- [ ] Integrate SymPy for symbolic math
- [ ] Add test case generation utilities
- [ ] Create verification reward functions

**Deliverable**: Code and math verification rewards

---

### Phase 3: Monitoring & Analysis (Week 5)
**Goal**: Comprehensive reward observability

- [ ] Implement `RewardMonitor` class
- [ ] Build W&B integration
- [ ] Create reward distribution visualizations
- [ ] Add reward hacking detection

**Deliverable**: Real-time reward analytics dashboard

---

### Phase 4: Advanced Features (Weeks 6-8)
**Goal**: Research-grade capabilities

- [ ] Implement adaptive reward strategies
- [ ] Add contrastive reward pre-training
- [ ] Build human-in-the-loop active learning
- [ ] Create reward ensemble evaluation

**Deliverable**: State-of-the-art reward system

---

## 5. Specific Enhancements for Current Platform

### 5.1 Immediate Wins (Low Effort, High Impact)

#### A. Weighted Reward Composition
**File**: `TunRex/src/tunrex/datasets/rewards.py`

Add to existing code:
```python
def composite_reward(weights=None):
    """
    Factory function for composite rewards.

    Args:
        weights: Dict mapping reward function names to weights
                 Default: {'format': 0.3, 'answer': 0.7}

    Returns:
        Composite reward function
    """
    if weights is None:
        weights = {
            'format_exact': 0.2,
            'format_approx': 0.1,
            'answer': 0.7
        }

    def _composite(prompts, completions, **kwargs):
        scores = np.zeros(len(completions))

        if 'format_exact' in weights:
            format_scores = match_format_exactly(prompts, completions, **kwargs)
            scores += np.array(format_scores) * weights['format_exact']

        if 'format_approx' in weights:
            approx_scores = match_format_approximately(prompts, completions, **kwargs)
            scores += np.array(approx_scores) * weights['format_approx']

        if 'answer' in weights:
            answer_scores = check_answer(prompts, completions, **kwargs)
            scores += np.array(answer_scores) * weights['answer']

        return scores.tolist()

    return _composite
```

**Impact**: Easy A/B testing of reward combinations

---

#### B. Reward Logging Enhancement
**File**: `scripts/train_grpo.py`

Add detailed logging:
```python
import wandb

# During training loop
def log_reward_components(step, rewards_dict):
    """Log individual reward components to W&B."""
    wandb.log({
        f'rewards/{k}_mean': np.mean(v),
        f'rewards/{k}_std': np.std(v),
        f'rewards/{k}_min': np.min(v),
        f'rewards/{k}_max': np.max(v),
    }, step=step)

    # Also log distributions
    wandb.log({
        f'rewards/{k}_hist': wandb.Histogram(v)
    }, step=step)
```

**Impact**: Better understanding of reward dynamics during training

---

### 5.2 Medium-Term Additions

#### C. Math Verification Reward
**New File**: `TunRex/src/tunrex/datasets/rewards_verified.py`

```python
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

def verify_symbolic_math(prompts, completions, answer, **kwargs):
    """
    Verify mathematical expressions symbolically.
    Handles algebraic equivalence, not just string matching.
    """
    scores = []

    for completion, true_answer in zip(completions, answer):
        # Extract answer from completion
        extracted = extract_answer_from_completion(completion)

        try:
            # Parse both as symbolic expressions
            student_expr = parse_expr(extracted)
            correct_expr = parse_expr(true_answer)

            # Check symbolic equality
            if sp.simplify(student_expr - correct_expr) == 0:
                scores.append(3.0)  # Perfect match
            else:
                # Try numerical evaluation for approximate match
                diff = abs(float(student_expr) - float(correct_expr))
                if diff < 0.01:
                    scores.append(1.5)  # Close enough
                else:
                    scores.append(0.0)
        except:
            # Fallback to string matching
            scores.append(1.0 if extracted.strip() == true_answer.strip() else 0.0)

    return scores
```

**Impact**: More accurate math problem evaluation

---

#### D. Uncertainty-Based Active Learning
**New File**: `TunRex/src/tunrex/active_learning.py`

```python
import numpy as np

class UncertaintySelector:
    """
    Select samples for human annotation based on reward model uncertainty.
    Implements RLTHF-style targeted feedback.
    """

    def __init__(self, reward_model, budget=100):
        self.reward_model = reward_model
        self.budget = budget
        self.annotated_samples = []

    def compute_uncertainty(self, completions):
        """
        Compute reward uncertainty for each completion.
        Use ensemble variance or dropout-based uncertainty.
        """
        # Multiple forward passes with dropout
        rewards_samples = []
        for _ in range(10):
            rewards = self.reward_model(completions, training=True)
            rewards_samples.append(rewards)

        # Variance as uncertainty measure
        uncertainty = np.std(rewards_samples, axis=0)
        return uncertainty

    def select_samples(self, completions, k=None):
        """
        Select top-k most uncertain samples for human annotation.
        """
        if k is None:
            k = self.budget - len(self.annotated_samples)

        uncertainty = self.compute_uncertainty(completions)
        top_k_indices = np.argsort(uncertainty)[-k:]

        return top_k_indices
```

**Impact**: 93% quality with 6-7% annotation effort (from research)

---

### 5.3 Configuration Updates

#### E. Enhanced Config for Rewards
**File**: `src/config.py`

Add new section:
```python
@dataclass
class RewardConfig:
    """Reward function configuration."""

    # Composition weights
    format_exact_weight: float = 0.2
    format_approx_weight: float = 0.1
    answer_correctness_weight: float = 0.7

    # Verification options
    use_symbolic_math: bool = False
    use_code_execution: bool = False

    # Adaptive rewards
    use_adaptive_rewards: bool = False
    adaptation_strategy: str = "curriculum"  # "curriculum", "temperature", "difficulty"

    # Monitoring
    log_reward_components: bool = True
    detect_reward_hacking: bool = True
    reward_distribution_check_interval: int = 10

    # Active learning
    use_active_learning: bool = False
    annotation_budget: int = 100
    uncertainty_threshold: float = 0.5

@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)  # NEW
```

**Impact**: Easy experimentation with reward configurations

---

## 6. Research-Validated Best Practices

Based on 2024-2025 research, follow these principles:

### 6.1 Reward Design Principles

1. **Multi-Signal Composition** (CARD Framework)
   - Combine format, correctness, and process rewards
   - Use learned weights or grid search for optimal combination
   - Log each component separately for debugging

2. **Verification Over Heuristics** (RLVR 2025)
   - Prefer executable verification (code, math) over string matching
   - Ground rewards in objective outcomes when possible
   - Use heuristics only as auxiliary signals

3. **Response-Level Efficiency** (2025 Theorem)
   - No need for token-level rewards
   - Response-level rewards are sufficient and more efficient
   - Use variance reduction techniques if needed

4. **Contrastive Training** (RLHF Part II)
   - If pre-training reward models, use contrastive objectives
   - Maximize margin between chosen/rejected pairs
   - Helps generalization to OOD samples

5. **Monitor for Reward Hacking**
   - Track reward distributions over training
   - Look for sudden spikes or mode collapse
   - Validate with held-out human judgments periodically

### 6.2 Training Best Practices

1. **Start Simple, Add Complexity**
   - Begin with basic rewards (format + correctness)
   - Add verification and adaptive components incrementally
   - A/B test each addition

2. **Log Everything**
   - Per-component reward scores
   - Distributions and statistics
   - Sample generations at regular intervals
   - KL divergence from base model

3. **Iterate on Rewards**
   - Rewards are as important as model architecture
   - Expect to iterate 5-10 times on reward design
   - Use small-scale experiments before full training

4. **Balance Exploration vs Exploitation**
   - Don't over-optimize for initial reward
   - Maintain diversity in generations (temperature, top-k)
   - Use KL penalty to prevent reward hacking

---

## 7. Competitive Analysis

### How Our Platform Compares to Research Systems

| Feature | Our Platform | OpenAI RLHF | Anthropic Constitutional AI | DeepSeek-GRM |
|---------|--------------|-------------|----------------------------|--------------|
| **Method** | GRPO | PPO | PPO + RLAIF | PPO |
| **Reward Composition** | Basic (can enhance) | Advanced | Multi-objective | Generalist |
| **Verification** | Limited | Code execution | Constitutional checks | General |
| **Human Feedback** | None | Extensive | RLAIF (AI feedback) | Extensive |
| **Adaptation** | Static | Dynamic | Rule-based | Inference-time scaling |
| **Open Source** | ✓ | ✗ | Partial | Partial |

**Our Advantage**: Fully open, modular, easy to customize for research

**Enhancement Opportunity**: Add verification, adaptation, and feedback loops

---

## 8. Recommended Next Steps

### Immediate (This Week)
1. Implement composite reward with configurable weights
2. Add per-component reward logging to training script
3. Create reward configuration YAML system

### Short-Term (Next Month)
1. Build symbolic math verification reward
2. Implement reward monitoring dashboard (W&B)
3. Add reward distribution tracking and alerts

### Medium-Term (Next Quarter)
1. Create adaptive reward system with curriculum learning
2. Build active learning sample selector
3. Add contrastive reward model pre-training option

### Research Directions
1. Explore LLM-generated reward functions (CARD framework)
2. Investigate inference-time reward scaling
3. Study reward hacking detection methods
4. Benchmark against DPO on reasoning tasks

---

## References & Sources

### Research Papers

1. [Reinforcement Learning Enhanced LLMs: A Survey](https://arxiv.org/abs/2412.10400) - Comprehensive 2025 survey
2. [Response-Level Rewards Are All You Need](https://arxiv.org/abs/2506.02553) - Theoretical foundation for response-level rewards
3. [Secrets of RLHF Part II: Reward Modeling](https://arxiv.org/abs/2401.06080) - Contrastive learning and meta-learning for rewards
4. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - DPO methodology
5. [Is DPO Superior to PPO?](https://arxiv.org/abs/2404.10719) - Comprehensive comparison
6. [LLM-Driven Reward Design Framework (CARD)](https://arxiv.org/abs/2410.14660) - Dynamic reward adaptation

### Resources

- [LLM Research Papers: 2025 List](https://magazine.sebastianraschka.com/p/llm-research-papers-2025-list-one)
- [awesome-RLHF GitHub Repository](https://github.com/opendilab/awesome-RLHF)
- [Direct Preference Optimization Explained](https://cameronrwolfe.substack.com/p/direct-preference-optimization)

### Industry References

- OpenAI RLHF systems (ChatGPT)
- Anthropic Constitutional AI (Claude)
- DeepSeek Generalist Reward Models
- Google Gemma + Tunix ecosystem

---

## Appendix: Code Examples

### A1. Full Composite Reward Implementation

See `enhancements/composite_rewards.py` for complete implementation with:
- Weight configuration
- Component logging
- Visualization tools
- Unit tests

### A2. Verification Reward Examples

See `enhancements/verification_rewards.py` for:
- Symbolic math verification
- Code execution sandbox
- Test case generation
- API-based verification (Wolfram Alpha, etc.)

### A3. Monitoring Dashboard

See `enhancements/reward_monitor.py` for:
- Real-time metrics collection
- W&B integration
- Reward hacking detection
- Distribution analysis

---

**Document Version**: 1.0
**Last Updated**: December 2025
**Author**: Claude Code AI Assistant
**Next Review**: After implementing Phase 1 enhancements
