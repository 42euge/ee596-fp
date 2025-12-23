# Data Quality Flywheel - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Run the Demo (2 minutes)

```bash
cd /home/user/ee596-fp
python examples/quality_flywheel_demo.py
```

This creates `./demo_output/` with:
- Quality dashboard
- Detailed reports
- Improvement suggestions
- Targeted training dataset

### Step 2: Explore the Outputs (1 minute)

```bash
# View the dashboard
cat demo_output/dashboard.txt

# See improvement suggestions
cat demo_output/improvements/demo_iteration/prompt_suggestions.json
cat demo_output/improvements/demo_iteration/reward_suggestions.json

# Check targeted dataset
cat demo_output/improvements/demo_iteration/targeted_dataset.json
```

### Step 3: Try the Review Interface (2 minutes)

```bash
python -m quality_flywheel.review_ui \
    --db demo_output/demo_feedback.db \
    --researcher your_name
```

Navigate the menu to:
- Review transcripts (option 1)
- View statistics (option 3)
- View dashboard (option 4)
- View suggestions (option 5)

---

## 📊 Use with Your Data

### Evaluate Your Model Outputs

```bash
# Prepare your data in JSON format:
# [
#   {"id": "1", "question": "...", "response": "...", "answer": "..."},
#   {"id": "2", "question": "...", "response": "...", "answer": "..."},
#   ...
# ]

python scripts/run_quality_flywheel.py \
    path/to/your/transcripts.json \
    --output-dir ./quality_reports \
    --generate-improvements \
    --iteration-name iteration_1
```

This will:
1. ✓ Evaluate quality metrics
2. ✓ Detect problematic transcripts
3. ✓ Generate analytics reports
4. ✓ Create improvement suggestions
5. ✓ Build targeted training dataset

### Review Flagged Transcripts

```bash
python -m quality_flywheel.review_ui \
    --db ./data/quality_feedback.db \
    --researcher your_name
```

### Check Results

```bash
# View dashboard
cat quality_reports/dashboard.txt

# See flagged transcripts
cat quality_reports/flagged_transcripts.json

# Get improvement suggestions
cat quality_reports/improvements/iteration_1/prompt_suggestions.json
cat quality_reports/improvements/iteration_1/reward_suggestions.json

# Use targeted dataset for retraining
cat quality_reports/improvements/iteration_1/targeted_dataset.json
```

---

## 🔧 Integration with Training

### Add to Your Evaluation Script

```python
from quality_flywheel import IssueDetector, FeedbackStore, QualityDashboard, QualityAnalytics

# After generating model outputs
detector = IssueDetector(quality_threshold=0.6)
store = FeedbackStore("./data/quality_feedback.db")

# Evaluate
results = detector.detect_batch(your_model_outputs)

# Save
for result in results:
    store.add_transcript(
        transcript_id=result.transcript_id,
        question=result.quality.question,
        response=result.quality.response,
        expected_answer=result.quality.expected_answer,
        quality_score=result.quality.overall_score,
        severity=result.quality.severity,
    )

# Report
flagged = [r for r in results if r.should_flag]
print(f"🚨 Flagged {len(flagged)} transcripts")

# Dashboard
analytics = QualityAnalytics(store)
dashboard = QualityDashboard(analytics)
print(dashboard.generate_text_dashboard())
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `examples/quality_flywheel_demo.py` | Demo with sample data |
| `scripts/run_quality_flywheel.py` | Main evaluation script |
| `src/quality_flywheel/review_ui.py` | Interactive review interface |
| `QUALITY_FLYWHEEL.md` | Complete documentation |
| `src/quality_flywheel/README.md` | Technical details |

---

## 🎯 Common Tasks

### Task: Find Low Quality Transcripts

```bash
python scripts/run_quality_flywheel.py transcripts.json --quality-threshold 0.5
cat quality_reports/flagged_transcripts.json
```

### Task: Generate Training Data from Issues

```bash
python scripts/run_quality_flywheel.py transcripts.json --generate-improvements
# Output: quality_reports/improvements/iteration_*/targeted_dataset.json
```

### Task: Get Prompt Improvement Suggestions

```bash
python scripts/run_quality_flywheel.py transcripts.json --generate-improvements
cat quality_reports/improvements/iteration_*/prompt_suggestions.json
```

### Task: Track Quality Over Time

```python
from quality_flywheel import QualityAnalytics, FeedbackStore

store = FeedbackStore("./data/quality_feedback.db")
analytics = QualityAnalytics(store)

trends = analytics.get_quality_trends(days=30)
print(f"Quality trend: {trends['timestamps']}")
print(f"Scores: {trends['overall_scores']}")
```

### Task: Review Specific Transcript

```bash
python -m quality_flywheel.review_ui --db ./data/quality_feedback.db
# Choose option 2: Search for specific transcript
# Enter transcript ID
```

---

## 🔄 Complete Workflow

1. **Generate Outputs**: Run your model on evaluation data
2. **Evaluate Quality**: `python scripts/run_quality_flywheel.py outputs.json --generate-improvements`
3. **Review Issues**: `python -m quality_flywheel.review_ui`
4. **Apply Improvements**: Use targeted dataset and suggestions
5. **Retrain Model**: Incorporate improvements into next iteration
6. **Repeat**: Run flywheel again to measure impact

---

## 💡 Tips

- **Start with the demo**: Always run `python examples/quality_flywheel_demo.py` first
- **Adjust thresholds**: Use `--quality-threshold` to control sensitivity
- **Review regularly**: Don't let flagged transcripts accumulate
- **Track trends**: Run on each training iteration to see improvement
- **Use corrections**: Researcher corrections improve the targeted dataset

---

## 🆘 Need Help?

1. **Run the demo**: `python examples/quality_flywheel_demo.py`
2. **Check documentation**: `QUALITY_FLYWHEEL.md`
3. **View examples**: Look in `./demo_output/` after running demo
4. **Review code**: All modules have docstrings and comments

---

**Next step**: Run the demo!

```bash
python examples/quality_flywheel_demo.py
```
