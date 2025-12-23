# Data Quality Flywheel

A comprehensive system for identifying problematic reasoning transcripts and feeding improvements back into the training pipeline.

## Overview

The Data Quality Flywheel is a continuous improvement system that helps researchers:

1. **Evaluate** model-generated reasoning transcripts for quality issues
2. **Detect** problematic patterns automatically
3. **Review** and annotate flagged transcripts
4. **Analyze** quality trends over time
5. **Improve** the training pipeline based on feedback

## Architecture

```
┌─────────────────┐
│  Transcripts    │
│   (Model        │
│   Outputs)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quality Metrics │  ← Evaluate format, answer, reasoning quality
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Issue Detector  │  ← Automatically flag problematic transcripts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feedback Store  │  ← Store annotations and researcher feedback
│   (SQLite DB)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Analytics     │  ← Analyze trends and patterns
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Improvement    │  ← Generate targeted datasets and suggestions
│      Loop       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Training       │  ← Feed improvements back to training
│   Pipeline      │
└─────────────────┘
```

## Components

### 1. Quality Metrics (`metrics.py`)

Comprehensive quality assessment for reasoning transcripts:

- **Format Quality**: Checks for proper `<reasoning>` and `<answer>` tags
- **Answer Quality**: Validates answer correctness and similarity
- **Reasoning Quality**: Evaluates completeness, coherence, and step markers
- **Content Quality**: Checks for numerical content and logical flow

**Example:**
```python
from quality_flywheel import QualityMetrics

metrics = QualityMetrics()

quality = metrics.evaluate(
    transcript_id="example_1",
    question="If a shirt costs $20 and is on sale for 25% off, what is the final price?",
    response="<reasoning>25% of $20 is $5. $20 - $5 = $15</reasoning><answer>$15</answer>",
    expected_answer="15",
)

print(f"Overall score: {quality.overall_score:.2%}")
print(f"Issues: {quality.issues}")
print(f"Severity: {quality.severity}")
```

### 2. Issue Detector (`detector.py`)

Automated detection of problematic transcripts:

- Configurable detection rules
- Severity classification (critical, high, medium, low)
- Priority-based flagging
- Batch processing support

**Example:**
```python
from quality_flywheel import IssueDetector

detector = IssueDetector(
    quality_threshold=0.6,
    auto_flag_critical=True,
    auto_flag_incorrect_answers=True,
)

results = detector.detect_batch(transcripts)

flagged = detector.get_flagged_transcripts(results, min_priority=3)
stats = detector.get_statistics(results)

print(f"Flagged: {stats['flagged_count']} / {stats['total_transcripts']}")
```

### 3. Feedback Store (`feedback.py`)

SQLite-based storage for annotations and feedback:

- Researcher annotations
- Quality history tracking
- Improvement suggestions
- Review status management

**Example:**
```python
from quality_flywheel import FeedbackStore, Annotation, AnnotationType

store = FeedbackStore("./data/quality_feedback.db")

# Add annotation
annotation = Annotation(
    transcript_id="example_1",
    annotation_type=AnnotationType.CORRECTED_ANSWER,
    researcher_id="researcher_1",
    content="The correct answer is $15",
)
store.add_annotation(annotation)

# Get pending reviews
pending = store.get_pending_transcripts(limit=10)
```

### 4. Analytics (`analytics.py`)

Quality trend analysis and visualization:

- Quality trends over time
- Issue pattern detection
- Researcher activity tracking
- Improvement impact measurement

**Example:**
```python
from quality_flywheel import QualityAnalytics, QualityDashboard

analytics = QualityAnalytics(store)
dashboard = QualityDashboard(analytics)

# Generate text dashboard
print(dashboard.generate_text_dashboard())

# Get quality trends
trends = analytics.get_quality_trends(days=30)

# Analyze issue patterns
patterns = analytics.get_issue_patterns(detection_results)
```

### 5. Improvement Loop (`improvement.py`)

Feeds insights back into training:

- Generates targeted training datasets
- Suggests prompt improvements
- Recommends reward function updates
- Creates improvement iterations

**Example:**
```python
from quality_flywheel import ImprovementLoop

improvement = ImprovementLoop(store)

# Generate targeted dataset from flagged transcripts
dataset_stats = improvement.generate_targeted_dataset(
    output_path="./data/targeted_dataset.json",
    max_examples=500,
    include_corrections=True,
)

# Get prompt improvement suggestions
prompt_suggestions = improvement.suggest_prompt_improvements(detection_results)

# Create complete improvement iteration
summary = improvement.create_improvement_iteration(
    iteration_name="iteration_1",
    detection_results=detection_results,
    output_dir="./data/improvements",
)
```

### 6. Review Interface (`review_ui.py`)

Interactive CLI for researchers:

```bash
python -m quality_flywheel.review_ui --db ./data/quality_feedback.db --researcher researcher_1
```

Features:
- Review pending transcripts
- Add comments and corrections
- Rate quality
- View statistics and dashboard
- Track improvement suggestions

## Usage

### Quick Start

1. **Evaluate transcripts:**

```bash
python scripts/run_quality_flywheel.py \
    path/to/transcripts.json \
    --db ./data/quality_feedback.db \
    --output-dir ./quality_reports
```

2. **Review flagged transcripts:**

```bash
python -m quality_flywheel.review_ui \
    --db ./data/quality_feedback.db \
    --researcher your_name
```

3. **Generate improvement iteration:**

```bash
python scripts/run_quality_flywheel.py \
    path/to/transcripts.json \
    --generate-improvements \
    --iteration-name iteration_1
```

### Integration with Training Pipeline

Add quality evaluation to your training workflow:

```python
from quality_flywheel import QualityMetrics, IssueDetector, FeedbackStore

# After generating model outputs
metrics = QualityMetrics()
detector = IssueDetector()
store = FeedbackStore()

# Evaluate quality
detection_results = detector.detect_batch(model_outputs)

# Save to database
for result in detection_results:
    store.add_transcript(
        transcript_id=result.transcript_id,
        question=result.quality.question,
        response=result.quality.response,
        expected_answer=result.quality.expected_answer,
        quality_score=result.quality.overall_score,
        severity=result.quality.severity,
    )

# Get flagged transcripts for review
flagged = detector.get_flagged_transcripts(detection_results)
print(f"Found {len(flagged)} transcripts requiring review")
```

## Workflow

### Complete Quality Flywheel Cycle

1. **Generate Transcripts**
   - Run model on evaluation dataset
   - Collect outputs in JSON/JSONL format

2. **Evaluate Quality**
   - Run quality flywheel script
   - Automatically detect issues
   - Flag problematic transcripts

3. **Researcher Review**
   - Use review interface to examine flagged transcripts
   - Add annotations and corrections
   - Confirm or reject detected issues

4. **Analyze Patterns**
   - View quality dashboard
   - Identify common failure modes
   - Track quality trends over time

5. **Generate Improvements**
   - Create targeted training dataset
   - Review prompt suggestions
   - Review reward function suggestions

6. **Update Training**
   - Incorporate targeted examples
   - Update prompts based on suggestions
   - Adjust reward functions
   - Retrain model

7. **Measure Impact**
   - Evaluate new model outputs
   - Compare quality metrics
   - Track improvement over iterations

## Configuration

### Issue Detection Thresholds

Customize detection sensitivity:

```python
detector = IssueDetector(
    quality_threshold=0.6,        # Minimum acceptable quality (0-1)
    confidence_threshold=0.7,     # Minimum confidence for auto-flagging
    auto_flag_critical=True,      # Auto-flag critical issues
    auto_flag_incorrect_answers=True,  # Auto-flag wrong answers
)
```

### Quality Metrics Weights

Customize how quality scores are calculated by modifying `metrics.py`:

```python
weights = {
    'format': 0.2,              # Format compliance weight
    'answer': 0.3,              # Answer correctness weight
    'reasoning_completeness': 0.2,  # Reasoning completeness weight
    'reasoning_coherence': 0.15,    # Reasoning coherence weight
    'content': 0.15,            # Content quality weight
}
```

## Data Format

### Input Transcript Format

```json
{
  "id": "transcript_123",
  "question": "What is 2 + 2?",
  "response": "<reasoning>Adding 2 and 2 gives us 4</reasoning><answer>4</answer>",
  "answer": "4"
}
```

### Output Quality Assessment

```json
{
  "transcript_id": "transcript_123",
  "overall_score": 0.95,
  "severity": "none",
  "issues": [],
  "quality": {
    "format_score": 1.0,
    "answer_correct": true,
    "reasoning_completeness": 0.9,
    "reasoning_coherence": 0.95
  }
}
```

## Database Schema

The feedback database stores:

- **transcripts**: Flagged transcripts and their metadata
- **annotations**: Researcher feedback and corrections
- **quality_history**: Quality metrics over time
- **improvement_suggestions**: Auto-generated improvement suggestions

## Reports and Outputs

The quality flywheel generates:

1. **dashboard.txt**: Text-based quality dashboard
2. **detailed_report.txt**: Comprehensive quality report
3. **dashboard.json**: JSON summary of all metrics
4. **issue_patterns.json**: Analysis of common failure patterns
5. **flagged_transcripts.json**: List of transcripts requiring review
6. **targeted_dataset.json**: Training dataset from flagged examples
7. **prompt_suggestions.json**: Suggested prompt improvements
8. **reward_suggestions.json**: Suggested reward function updates

## Best Practices

1. **Regular Evaluation**: Run quality evaluation after each training iteration
2. **Timely Review**: Have researchers review flagged transcripts promptly
3. **Track Trends**: Monitor quality metrics over time to measure improvement
4. **Act on Suggestions**: Implement improvement suggestions systematically
5. **Iterative Refinement**: Use each iteration to refine detection rules and thresholds
6. **Diverse Reviewers**: Have multiple researchers review transcripts for consistency
7. **Document Changes**: Track which improvements were implemented and their impact

## Troubleshooting

### No transcripts flagged

- Lower `quality_threshold` parameter
- Check if transcripts are in correct format
- Verify expected answers are provided

### Too many false positives

- Increase `quality_threshold`
- Adjust detection rules in `detector.py`
- Use researcher feedback to refine detection

### Database errors

- Ensure database directory exists and is writable
- Check database file permissions
- Backup database regularly

## Contributing

To extend the quality flywheel:

1. **Add new detection rules**: Modify `detector.py`
2. **Add new quality metrics**: Extend `metrics.py`
3. **Add new annotation types**: Update `feedback.py`
4. **Customize analytics**: Extend `analytics.py`
5. **Add new improvement strategies**: Enhance `improvement.py`

## License

Part of the EE596 Final Project - Gemma3-1B GRPO Fine-tuning System
