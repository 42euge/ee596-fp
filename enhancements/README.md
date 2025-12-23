# Reward System Enhancements

This directory contains enhanced reward implementations based on latest research (2024-2025).

## Structure

- `composite_rewards.py` - Multi-signal reward composition with configurable weights
- `verification_rewards.py` - Code execution and symbolic math verification
- `reward_monitor.py` - Real-time monitoring and reward hacking detection
- `adaptive_rewards.py` - Dynamic reward adaptation strategies
- `active_learning.py` - Uncertainty-based sample selection for human feedback

## Usage

These enhancements are designed to be drop-in replacements or additions to the existing reward system in `TunRex/src/tunrex/datasets/rewards.py`.

See `/docs/research-to-platform-mapping.md` for detailed implementation roadmap and research background.

## Implementation Status

- [ ] Composite Rewards (Priority 1)
- [ ] Verification Rewards (Priority 2)
- [ ] Reward Monitor (Priority 3)
- [ ] Adaptive Rewards (Priority 4)
- [ ] Active Learning (Priority 5)

## Quick Start

```python
from enhancements.composite_rewards import CompositeReward
from tunrex.datasets.rewards import match_format_exactly, check_answer

# Create composite reward
reward_fn = CompositeReward(
    components={
        'format': match_format_exactly,
        'answer': check_answer
    },
    weights={
        'format': 0.3,
        'answer': 0.7
    }
)

# Use in training
scores = reward_fn(prompts, completions, answer=answers)
```

## Research References

All enhancements are based on peer-reviewed research from 2024-2025. See the research-to-platform mapping document for full citations.
