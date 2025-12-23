# Data Quality Flywheel System

## Overview

The **Data Quality Flywheel** is a comprehensive continuous improvement system for identifying problematic reasoning transcripts and feeding improvements back into the GRPO training pipeline.

This system helps researchers:
- 🔍 **Identify** quality issues in model-generated reasoning automatically
- 📊 **Track** quality metrics and trends over time
- ✍️ **Annotate** and correct problematic transcripts
- 📈 **Analyze** common failure patterns
- 🔄 **Improve** the training pipeline with targeted datasets and suggestions

## Quick Start

### 1. Install Dependencies

The quality flywheel uses only standard Python libraries:
- `sqlite3` (built-in)
- `json` (built-in)
- `numpy` (should already be installed for the ML pipeline)

### 2. Run the Demo

See the system in action with sample data:

```bash
cd /home/user/ee596-fp
python examples/quality_flywheel_demo.py
```

This will:
- Evaluate 7 sample transcripts
- Detect quality issues
- Generate reports and analytics
- Create improvement suggestions
- Output everything to `./demo_output/`

### 3. Try with Real Data

Evaluate your own model outputs:

```bash
python scripts/run_quality_flywheel.py \
    path/to/your/transcripts.json \
    --output-dir ./quality_reports
```

### 4. Review Flagged Transcripts

Launch the interactive review interface:

```bash
python -m quality_flywheel.review_ui \
    --db ./data/quality_feedback.db \
    --researcher your_name
```

## Architecture

The flywheel consists of 6 main components:

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA QUALITY FLYWHEEL                      │
└──────────────────────────────────────────────────────────────┘

1. Quality Metrics (metrics.py)
   ├─ Format validation (tags, structure)
   ├─ Answer correctness checking
   ├─ Reasoning quality assessment
   └─ Overall quality scoring

2. Issue Detector (detector.py)
   ├─ Automated issue detection
   ├─ Severity classification
   ├─ Priority-based flagging
   └─ Statistical analysis

3. Feedback Store (feedback.py)
   ├─ SQLite database for annotations
   ├─ Researcher feedback tracking
   ├─ Quality history logging
   └─ Improvement suggestions

4. Analytics (analytics.py)
   ├─ Quality trend analysis
   ├─ Issue pattern detection
   ├─ Researcher activity tracking
   └─ Dashboard generation

5. Improvement Loop (improvement.py)
   ├─ Targeted dataset generation
   ├─ Prompt improvement suggestions
   ├─ Reward function recommendations
   └─ Iteration packaging

6. Review Interface (review_ui.py)
   ├─ Interactive transcript review
   ├─ Annotation and correction tools
   ├─ Quality rating
   └─ Statistics viewing
```

## How It Works

### The Flywheel Cycle

```
   ┌─────────────────────────────────────────┐
   │                                         │
   │  1. MODEL GENERATES TRANSCRIPTS         │
   │     (Reasoning + Answers)               │
   │                                         │
   └──────────────┬──────────────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────────────┐
   │                                         │
   │  2. QUALITY EVALUATION                  │
   │     • Format check                      │
   │     • Answer validation                 │
   │     • Reasoning assessment              │
   │                                         │
   └──────────────┬──────────────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────────────┐
   │                                         │
   │  3. ISSUE DETECTION                     │
   │     • Automatic flagging                │
   │     • Severity classification           │
   │     • Priority assignment               │
   │                                         │
   └──────────────┬──────────────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────────────┐
   │                                         │
   │  4. RESEARCHER REVIEW                   │
   │     • Examine flagged items             │
   │     • Add corrections                   │
   │     • Provide feedback                  │
   │                                         │
   └──────────────┬──────────────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────────────┐
   │                                         │
   │  5. ANALYTICS & INSIGHTS                │
   │     • Trend analysis                    │
   │     • Pattern detection                 │
   │     • Impact measurement                │
   │                                         │
   └──────────────┬──────────────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────────────┐
   │                                         │
   │  6. IMPROVEMENT GENERATION              │
   │     • Targeted datasets                 │
   │     • Prompt suggestions                │
   │     • Reward function updates           │
   │                                         │
   └──────────────┬──────────────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────────────┐
   │                                         │
   │  7. TRAINING PIPELINE UPDATE            │
   │     • Retrain with targeted data        │
   │     • Apply prompt changes              │
   │     • Update reward functions           │
   │                                         │
   └──────────────┬──────────────────────────┘
                  │
                  └──────────> BACK TO STEP 1
```

## Key Features

### 🎯 Comprehensive Quality Metrics

The system evaluates multiple quality dimensions:

- **Format Quality**: Proper XML-style tag structure
- **Answer Quality**: Correctness and numerical accuracy
- **Reasoning Quality**: Completeness, coherence, step markers
- **Content Quality**: Logical flow, calculations, numerical content

### 🚨 Intelligent Issue Detection

Automatic detection of 11+ issue types:
- Missing format tags (critical)
- Incorrect answers (high)
- Insufficient reasoning (high)
- Low coherence (medium)
- Missing step markers (medium)
- Weak logical flow (medium)
- And more...

### 💾 Persistent Feedback Storage

SQLite database tracks:
- All flagged transcripts
- Researcher annotations
- Quality history over time
- Improvement suggestions
- Review status

### 📊 Rich Analytics

Track and visualize:
- Quality trends over time
- Common failure patterns
- Issue correlations
- Researcher activity
- Improvement impact

### 🔄 Automated Improvement Loop

Generates:
- **Targeted datasets** from problematic examples
- **Prompt suggestions** based on common issues
- **Reward function recommendations** based on patterns
- **Complete iteration packages** ready for deployment

### 🖥️ Interactive Review Interface

User-friendly CLI for researchers:
- Browse pending transcripts
- Add corrections and comments
- Rate quality
- View statistics
- Track progress

## Usage Examples

### Example 1: Basic Quality Evaluation

```python
from quality_flywheel import QualityMetrics, IssueDetector

# Initialize
metrics = QualityMetrics()
detector = IssueDetector()

# Evaluate a single transcript
quality = metrics.evaluate(
    transcript_id="example_1",
    question="What is 5 + 3?",
    response="<reasoning>5 + 3 = 8</reasoning><answer>8</answer>",
    expected_answer="8"
)

print(f"Quality Score: {quality.overall_score:.2%}")
print(f"Issues: {quality.issues}")

# Detect issues
result = detector.detect(quality)
print(f"Should Flag: {result.should_flag}")
print(f"Priority: {result.priority}")
```

### Example 2: Batch Processing

```python
# Load your transcripts
transcripts = [
    {"id": "1", "question": "...", "response": "...", "answer": "..."},
    {"id": "2", "question": "...", "response": "...", "answer": "..."},
    # ... more transcripts
]

# Batch detection
results = detector.detect_batch(transcripts)

# Get flagged transcripts
flagged = detector.get_flagged_transcripts(results, min_priority=3)
print(f"Flagged {len(flagged)} transcripts for review")

# Get statistics
stats = detector.get_statistics(results)
print(f"Average quality: {stats['average_quality_score']:.2%}")
```

### Example 3: Integration with Training

```python
from quality_flywheel import FeedbackStore, ImprovementLoop

# Initialize
store = FeedbackStore("./data/quality_feedback.db")
improvement = ImprovementLoop(store)

# Generate targeted dataset from flagged transcripts
dataset_stats = improvement.generate_targeted_dataset(
    output_path="./data/targeted_training.json",
    max_examples=500,
    include_corrections=True
)

# Get improvement suggestions
prompt_suggestions = improvement.suggest_prompt_improvements(results)
reward_suggestions = improvement.suggest_reward_function_updates(results)

# Create complete iteration package
summary = improvement.create_improvement_iteration(
    iteration_name="iteration_1",
    detection_results=results,
    output_dir="./improvements"
)
```

## Command-Line Tools

### Main Evaluation Script

```bash
python scripts/run_quality_flywheel.py [OPTIONS] TRANSCRIPTS_FILE

Options:
  --db PATH                 Database path (default: ./data/quality_feedback.db)
  --output-dir PATH         Output directory (default: ./quality_reports)
  --quality-threshold FLOAT Flagging threshold (default: 0.6)
  --no-save                 Don't save to database
  --generate-improvements   Generate improvement iteration
  --iteration-name NAME     Iteration name
```

### Review Interface

```bash
python -m quality_flywheel.review_ui [OPTIONS]

Options:
  --db PATH          Database path
  --researcher NAME  Researcher identifier
```

## Integration with Existing Pipeline

Add to your training workflow:

```python
# In your evaluation script, after generating model outputs:

from quality_flywheel import QualityMetrics, IssueDetector, FeedbackStore

# Initialize components
store = FeedbackStore()
detector = IssueDetector()

# Evaluate outputs
results = detector.detect_batch(model_outputs)

# Save to database
for result in results:
    store.add_transcript(
        transcript_id=result.transcript_id,
        question=result.quality.question,
        response=result.quality.response,
        expected_answer=result.quality.expected_answer,
        quality_score=result.quality.overall_score,
        severity=result.quality.severity,
    )

# Report flagged count
flagged = [r for r in results if r.should_flag]
print(f"🚨 {len(flagged)} transcripts flagged for review")

# Generate report
from quality_flywheel import QualityAnalytics, QualityDashboard

analytics = QualityAnalytics(store)
dashboard = QualityDashboard(analytics)
print(dashboard.generate_text_dashboard())
```

## File Structure

```
src/quality_flywheel/
├── __init__.py              # Package initialization
├── metrics.py               # Quality metrics calculation
├── detector.py              # Issue detection
├── feedback.py              # Feedback storage (SQLite)
├── analytics.py             # Analytics and dashboards
├── improvement.py           # Improvement loop
├── review_ui.py             # Researcher interface
└── README.md                # Detailed documentation

scripts/
└── run_quality_flywheel.py  # Main evaluation script

examples/
└── quality_flywheel_demo.py # Demonstration with samples

data/
├── quality_feedback.db      # Feedback database (created automatically)
└── improvements/            # Generated improvement iterations
```

## Configuration

### Detection Sensitivity

Adjust thresholds in `IssueDetector`:

```python
detector = IssueDetector(
    quality_threshold=0.6,        # Lower = more flagging
    confidence_threshold=0.7,     # Higher = more conservative
    auto_flag_critical=True,      # Auto-flag critical issues
    auto_flag_incorrect_answers=True,  # Auto-flag wrong answers
)
```

### Quality Weights

Customize scoring in `metrics.py` (`_calculate_overall_score`):

```python
weights = {
    'format': 0.2,              # Format compliance
    'answer': 0.3,              # Answer correctness
    'reasoning_completeness': 0.2,
    'reasoning_coherence': 0.15,
    'content': 0.15,
}
```

## Output Files

The flywheel generates:

1. **dashboard.txt** - Human-readable dashboard
2. **dashboard.json** - Machine-readable metrics
3. **detailed_report.txt** - Comprehensive analysis
4. **flagged_transcripts.json** - List of problematic transcripts
5. **issue_patterns.json** - Pattern analysis
6. **improvements/iteration_X/** - Complete improvement package
   - `targeted_dataset.json` - Training data
   - `prompt_suggestions.json` - Prompt updates
   - `reward_suggestions.json` - Reward function updates
   - `iteration_summary.json` - Summary

## Best Practices

1. **Evaluate Regularly**: Run after each training iteration
2. **Review Promptly**: Don't let flagged transcripts accumulate
3. **Track Trends**: Monitor quality over time
4. **Act on Suggestions**: Implement improvements systematically
5. **Iterate**: Use feedback to refine detection rules
6. **Document**: Track what was changed and why

## Troubleshooting

**Q: No transcripts are being flagged**
- Lower `quality_threshold` (try 0.4 or 0.5)
- Check transcript format matches expected structure
- Ensure expected answers are provided

**Q: Too many false positives**
- Raise `quality_threshold` (try 0.7 or 0.8)
- Adjust detection rules in `detector.py`
- Use researcher annotations to train better thresholds

**Q: Database errors**
- Ensure parent directory exists
- Check file permissions
- Try deleting and recreating the database

## Advanced Usage

See `src/quality_flywheel/README.md` for:
- Detailed component documentation
- API reference
- Extension guide
- Database schema
- Custom detection rules

## Performance

The quality flywheel is designed to be efficient:
- **Evaluation**: ~100-1000 transcripts/second (depends on length)
- **Database**: Handles millions of transcripts
- **Memory**: Low footprint, streams large datasets
- **Disk**: SQLite database, typically <10MB per 10K transcripts

## Future Enhancements

Potential additions:
- Web-based review interface
- Multi-user collaboration
- Machine learning-based quality prediction
- A/B testing framework
- Integration with W&B for tracking
- Automated prompt optimization
- Real-time quality monitoring

## Support

For questions or issues:
1. Check the documentation in `src/quality_flywheel/README.md`
2. Run the demo: `python examples/quality_flywheel_demo.py`
3. Review example outputs in `./demo_output/`

## License

Part of the EE596 Final Project - Gemma3-1B GRPO Fine-tuning System

---

**Ready to improve your data quality?** Start with the demo!

```bash
python examples/quality_flywheel_demo.py
```
