# Reward Hack Detection and Behavior Monitoring System

## Overview

This system provides automated detection of reward hacking and problematic behaviors during GRPO (Group Relative Policy Optimization) training. It monitors various aspects of the training process to identify when the model is exploiting reward functions or exhibiting degenerate behaviors.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Detection Mechanisms](#detection-mechanisms)
- [Configuration](#configuration)
- [Integration Guide](#integration-guide)
- [Monitoring Metrics](#monitoring-metrics)
- [Troubleshooting](#troubleshooting)

## Features

### Core Detection Capabilities

1. **Statistical Anomaly Detection**
   - Reward distribution anomalies using z-score analysis
   - Distribution shift detection
   - Variance monitoring

2. **Response Quality Analysis**
   - Length anomalies (too short/long responses)
   - Excessive token repetition detection
   - N-gram repetition analysis
   - Response diversity tracking (mode collapse detection)

3. **Reward Function Exploitation**
   - Format gaming detection
   - Reward component imbalance detection
   - Quality vs format reward analysis

4. **Training Dynamics Monitoring**
   - KL divergence tracking (too high or too low)
   - Gradient norm monitoring (exploding/vanishing gradients)
   - Loss plateau detection

5. **Weights & Biases Integration**
   - Automatic metric logging
   - Real-time alerts for critical issues
   - Comprehensive dashboards

## Architecture

The monitoring system consists of three main components:

```
src/
├── reward_monitoring.py              # Core detection algorithms
├── reward_monitoring_integration.py  # Integration with training loops
└── config.py                        # Configuration (MonitoringConfig)

examples/
└── reward_monitoring_example.py     # Usage examples
```

### Component Descriptions

- **`RewardHackDetector`**: Main detector coordinating all monitoring systems
- **`RewardStatistics`**: Tracks and analyzes reward distributions over time
- **`ResponseAnalyzer`**: Analyzes generated responses for problematic patterns
- **`RewardComponentAnalyzer`**: Monitors individual reward components
- **`TrainingDynamicsMonitor`**: Monitors training stability metrics
- **`RewardFunctionMonitor`**: Integration wrapper for GRPO training

## Quick Start

### Basic Usage

```python
from src.reward_monitoring import RewardHackDetector, DetectionConfig
from TunRex.src.tunrex.datasets.rewards import (
    match_format_exactly,
    check_answer,
)

# Create detector with custom config
config = DetectionConfig(
    reward_zscore_threshold=3.0,
    min_reasoning_length=20,
    max_ngram_repetition_ratio=0.3,
)
detector = RewardHackDetector(config)

# Analyze a training step
detections = detector.analyze_step(
    response=generated_response,
    total_reward=total_reward,
    reward_components={
        'format': format_reward,
        'accuracy': accuracy_reward,
    },
    kl_divergence=kl_div,
    gradient_norm=grad_norm,
    loss=current_loss,
)

# Handle detections
for detection in detections:
    print(f"[{detection.severity}] {detection.detection_type}: {detection.message}")
```

### Integration with GRPO Training

```python
from src.reward_monitoring_integration import (
    RewardFunctionMonitor,
    create_monitoring_callback,
)
from TunRex.src.tunrex.datasets.rewards import (
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
)

# Create monitor
monitor = RewardFunctionMonitor(
    reward_fns=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    reward_fn_names=['exact_format', 'approx_format', 'accuracy', 'numbers'],
    wandb_enabled=True,
)

# Use in training loop
for step, batch in enumerate(train_dataset):
    # ... generate responses ...

    # Compute rewards with monitoring
    results = monitor.compute_rewards(
        responses=generated_responses,
        questions=questions,
        answers=ground_truth_answers,
        step=step,
        kl_divergence=kl_div,
        gradient_norm=grad_norm,
        loss=loss_value,
    )

    # Extract rewards for optimization
    rewards = [result.reward for result in results]

    # ... continue training ...

# Print final summary
print(monitor.get_summary_report())
```

## Detection Mechanisms

### 1. Statistical Anomaly Detection

**Purpose**: Detect abnormal reward values that deviate significantly from expected distribution.

**Method**:
- Maintains rolling statistics (mean, std) of rewards
- Uses z-score to identify outliers
- Detects distribution shifts using recent vs historical comparison

**Triggers**:
- Z-score > `reward_zscore_threshold` (default: 3.0)
- Recent mean differs by > 2σ from historical mean

**Example**:
```
Reward anomaly: 15.20 (mean: 5.30, std: 2.10)
Distribution shift detected: shift score 3.50
```

### 2. Response Length Monitoring

**Purpose**: Detect responses that are too short (minimal effort) or too long (rambling).

**Method**:
- Checks character count against min/max thresholds
- Can use z-score for dynamic thresholding

**Triggers**:
- Length < `min_response_length` (default: 10)
- Length > `max_response_length` (default: 2048)

**Example**:
```
Response too short: 5 chars (min: 10)
```

### 3. Repetition Detection

**Purpose**: Identify excessive repetition of tokens or phrases.

**Method**:
- Token-level repetition: counts most frequent token
- N-gram repetition: identifies repeated phrases
- Computes repetition ratio

**Triggers**:
- Token repetition ratio > `max_token_repetition_ratio` (default: 0.5)
- N-gram repetition ratio > `max_ngram_repetition_ratio` (default: 0.3)

**Example**:
```
Excessive token repetition: "the" appears 45 times (65.2%)
3-gram repetition detected: 40.0% ratio
```

### 4. Format Gaming Detection

**Purpose**: Detect when model optimizes for format rewards without substantive content.

**Method**:
- Extracts reasoning section length
- Compares format vs quality reward balance
- Checks for minimal reasoning with high format reward

**Triggers**:
- High format reward but reasoning < `min_reasoning_length` (default: 20)
- Format reward ratio > `format_quality_ratio_threshold` with zero quality reward

**Example**:
```
Format gaming detected: high format reward (3.0) but minimal reasoning (8 chars)
Reward imbalance: 100.0% from format, 0% from quality
```

### 5. Mode Collapse Detection

**Purpose**: Detect when model generates very similar or identical responses.

**Method**:
- Maintains hash-based fingerprints of recent responses
- Tracks unique response ratio
- Triggers when diversity drops below threshold

**Triggers**:
- Unique response ratio < `min_unique_responses_ratio` (default: 0.5)

**Example**:
```
Potential mode collapse: only 35.0% unique responses in last 100 samples
```

### 6. KL Divergence Monitoring

**Purpose**: Monitor policy deviation from reference model.

**Method**:
- Tracks KL divergence values
- Detects both excessive divergence and insufficient exploration

**Triggers**:
- KL < `kl_divergence_min` (default: 0.001) - too little exploration
- KL > `kl_divergence_max` (default: 5.0) - policy diverging too much

**Example**:
```
KL divergence too low: 0.000500 (min: 0.001000) - insufficient exploration
KL divergence too high: 7.50 (max: 5.00) - policy diverging too much
```

### 7. Gradient Monitoring

**Purpose**: Detect training instabilities through gradient analysis.

**Method**:
- Monitors gradient norm values
- Detects both vanishing and exploding gradients

**Triggers**:
- Gradient norm < `gradient_norm_min` (default: 1e-6) - vanishing
- Gradient norm > `gradient_norm_max` (default: 10.0) - exploding

**Example**:
```
Vanishing gradients detected: 5.23e-08 (min: 1.00e-06)
Exploding gradients detected: 125.30 (max: 10.00)
```

### 8. Loss Plateau Detection

**Purpose**: Identify when training has stagnated.

**Method**:
- Computes coefficient of variation (CV) of recent losses
- Very low CV indicates plateau

**Triggers**:
- CV < 0.01 over last 20 steps

**Example**:
```
Loss plateau detected: CV=0.0045 over last 20 steps
```

### 9. Reward Component Exploitation

**Purpose**: Detect if model exploits a single reward component.

**Method**:
- Tracks contribution of each reward component
- Identifies when one component dominates

**Triggers**:
- Single component > `max_component_imbalance` (default: 90%) of total reward

**Example**:
```
Reward exploitation detected: 95.2% of reward from exact_format component
```

## Configuration

### DetectionConfig Parameters

```python
from src.reward_monitoring import DetectionConfig

config = DetectionConfig(
    # Statistical anomaly detection
    reward_zscore_threshold=3.0,          # Z-score threshold for anomalies
    reward_variance_threshold=5.0,        # Max variance change factor
    min_samples_for_detection=10,         # Min samples before detection

    # Response length monitoring
    min_response_length=10,               # Min acceptable length
    max_response_length=2048,             # Max acceptable length
    length_outlier_threshold=3.0,         # Z-score for length outliers

    # Repetition detection
    max_ngram_repetition_ratio=0.3,       # Max ratio of repeated n-grams
    ngram_size=3,                         # N-gram size to check
    max_token_repetition_ratio=0.5,       # Max ratio of repeated tokens

    # Format gaming detection
    min_reasoning_length=20,              # Min chars in reasoning
    format_quality_ratio_threshold=0.3,   # Min quality/format ratio

    # Diversity monitoring
    min_unique_responses_ratio=0.5,       # Min unique response ratio
    similarity_threshold=0.9,             # Similarity threshold

    # KL divergence monitoring
    kl_divergence_min=0.001,              # Too little exploration
    kl_divergence_max=5.0,                # Too much divergence

    # Gradient monitoring
    gradient_norm_max=10.0,               # Max gradient norm
    gradient_norm_min=1e-6,               # Min gradient norm

    # Reward component balance
    max_component_imbalance=0.9,          # Max single component fraction

    # Window sizes
    short_window_size=20,                 # Short-term statistics
    long_window_size=100,                 # Long-term statistics
)
```

### MonitoringConfig in Config

```python
from src.config import Config

config = Config()
config.monitoring.enabled = True
config.monitoring.reward_zscore_threshold = 3.0
config.monitoring.log_detections_to_wandb = True
config.monitoring.detection_log_interval = 1
```

## Integration Guide

### Step 1: Import Required Modules

```python
from src.reward_monitoring import RewardHackDetector, DetectionConfig
from src.reward_monitoring_integration import (
    RewardFunctionMonitor,
    log_detections_to_wandb,
)
```

### Step 2: Create Monitor Instance

```python
# Option A: Use default config
monitor = RewardFunctionMonitor(
    reward_fns=your_reward_functions,
    reward_fn_names=['format', 'accuracy', 'numbers'],
    wandb_enabled=True,
)

# Option B: Use custom config
custom_config = DetectionConfig(
    reward_zscore_threshold=2.5,  # More sensitive
    min_reasoning_length=50,       # Require longer reasoning
)

monitor = RewardFunctionMonitor(
    reward_fns=your_reward_functions,
    reward_fn_names=['format', 'accuracy', 'numbers'],
    detection_config=custom_config,
    wandb_enabled=True,
)
```

### Step 3: Integrate in Training Loop

```python
for step in range(num_training_steps):
    # Generate responses
    responses = model.generate(prompts)

    # Compute monitored rewards
    results = monitor.compute_rewards(
        responses=responses,
        questions=questions,
        answers=ground_truth,
        step=step,
        kl_divergence=kl_div,      # Optional
        gradient_norm=grad_norm,    # Optional
        loss=current_loss,          # Optional
    )

    # Extract rewards for optimization
    rewards = [r.reward for r in results]

    # Check for critical issues
    if monitor.should_alert():
        logger.warning("Critical training issues detected!")
        # Optionally: reduce learning rate, checkpoint, etc.

    # Log metrics periodically
    if step % 10 == 0:
        metrics = monitor.get_metrics_for_logging()
        wandb.log(metrics, step=step)
```

### Step 4: Generate Summary Report

```python
# At end of training
final_report = monitor.get_summary_report()
print(final_report)

# Save report
with open('monitoring_report.txt', 'w') as f:
    f.write(final_report)
```

## Monitoring Metrics

### Metrics Logged to W&B

The system automatically logs the following metrics:

#### Reward Metrics
- `reward/mean` - Overall mean reward
- `reward/std` - Reward standard deviation
- `reward/recent_mean` - Recent mean (last 20 steps)
- `reward/recent_std` - Recent std
- `reward/min` - Minimum reward seen
- `reward/max` - Maximum reward seen

#### Component Metrics
- `reward_component/{name}/mean` - Mean for each component
- `reward_component/{name}/recent_mean` - Recent mean for component

#### Detection Metrics
- `detections/total` - Total detections
- `detections/critical` - Critical severity count
- `detections/high` - High severity count
- `detections/medium` - Medium severity count
- `detections/low` - Low severity count
- `detections/rate` - Detections per 100 steps

#### Detection Type Breakdown
- `detection_type/reward_anomaly` - Count
- `detection_type/format_gaming` - Count
- `detection_type/mode_collapse` - Count
- `detection_type/token_repetition` - Count
- `detection_type/kl_divergence` - Count
- `detection_type/gradient_norm` - Count
- ... (one for each detection type)

### W&B Alerts

The system automatically creates W&B alerts for:
- **CRITICAL** severity: Error-level alerts
- **HIGH** severity: Warning-level alerts

Example alert:
```
CRITICAL: mode_collapse
Potential mode collapse: only 35.0% unique responses in last 100 samples
```

## Troubleshooting

### Common Issues

#### 1. Too Many False Positives

**Symptom**: Many low-severity detections that aren't problematic

**Solutions**:
- Increase thresholds (e.g., `reward_zscore_threshold=4.0`)
- Increase `min_samples_for_detection` to reduce early noise
- Adjust component-specific thresholds

```python
config = DetectionConfig(
    reward_zscore_threshold=4.0,      # Less sensitive
    min_samples_for_detection=20,     # More samples needed
    max_ngram_repetition_ratio=0.4,   # Allow more repetition
)
```

#### 2. Missing Real Issues

**Symptom**: Known problems not being detected

**Solutions**:
- Decrease thresholds for higher sensitivity
- Enable more detection types
- Check that all metrics are being passed to `analyze_step()`

```python
config = DetectionConfig(
    reward_zscore_threshold=2.0,      # More sensitive
    min_reasoning_length=30,          # Stricter format requirements
)
```

#### 3. W&B Logging Not Working

**Symptom**: Metrics not appearing in W&B

**Checks**:
- Verify `wandb.init()` was called
- Check `wandb_enabled=True` in monitor
- Ensure W&B API key is set
- Check internet connectivity

```python
import wandb

# Verify W&B is initialized
assert wandb.run is not None, "W&B not initialized"

# Check monitor config
assert monitor.wandb_enabled == True
```

#### 4. Performance Impact

**Symptom**: Training slowed down significantly

**Solutions**:
- Increase `detection_log_interval` to log less frequently
- Disable some detection mechanisms
- Use smaller window sizes

```python
config = DetectionConfig(
    short_window_size=10,    # Smaller windows
    long_window_size=50,
    detection_log_interval=20,  # Log less often
)
```

### Interpreting Detection Severity

- **CRITICAL**: Immediate action required (mode collapse, exploding gradients)
- **HIGH**: Serious issue that needs attention (format gaming, vanishing gradients)
- **MEDIUM**: Potential issue to monitor (reward anomalies, distribution shifts)
- **LOW**: Informational, may be normal variation

### Best Practices

1. **Start with Default Config**: Begin with defaults and adjust based on results
2. **Monitor Early Training**: Expect more anomalies in early steps
3. **Review Periodically**: Check summary reports every 50-100 steps
4. **Combine with Visual Inspection**: Use W&B charts to verify detections
5. **Adjust for Your Task**: Different tasks may need different thresholds
6. **Log Everything**: Better to have too much data than too little

## Examples

See `examples/reward_monitoring_example.py` for comprehensive examples including:
- Basic usage
- Integration with training loops
- Anomaly detection demonstrations
- Training dynamics monitoring

Run examples:
```bash
python examples/reward_monitoring_example.py
```

## API Reference

### RewardHackDetector

```python
detector = RewardHackDetector(config: Optional[DetectionConfig] = None)

detections = detector.analyze_step(
    response: str,
    total_reward: float,
    reward_components: Optional[Dict[str, float]] = None,
    kl_divergence: Optional[float] = None,
    gradient_norm: Optional[float] = None,
    loss: Optional[float] = None
) -> List[DetectionResult]

metrics = detector.get_summary_metrics() -> Dict[str, Any]
```

### RewardFunctionMonitor

```python
monitor = RewardFunctionMonitor(
    reward_fns: List[Callable],
    reward_fn_names: List[str],
    detection_config: Optional[DetectionConfig] = None,
    wandb_enabled: bool = True
)

results = monitor.compute_rewards(
    responses: List[str],
    questions: List[str],
    answers: List[str],
    step: int,
    kl_divergence: Optional[float] = None,
    gradient_norm: Optional[float] = None,
    loss: Optional[float] = None
) -> List[MonitoredRewardResult]

metrics = monitor.get_metrics_for_logging() -> Dict[str, Any]
report = monitor.get_summary_report() -> str
should_alert = monitor.should_alert() -> bool
```

## Contributing

When adding new detection mechanisms:

1. Add the detection logic to appropriate analyzer class in `reward_monitoring.py`
2. Add configuration parameters to `DetectionConfig`
3. Integrate with `RewardHackDetector.analyze_step()`
4. Add corresponding metrics to `get_summary_metrics()`
5. Update documentation and examples
6. Add unit tests

## License

See main project LICENSE file.
