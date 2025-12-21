# TPU Pricing & Cost Analysis

## TPU Options for Gemma 3 1B-IT Training

Based on GRPO training with ~3,364 steps, LoRA rank 64.

| Type | $/hr On-Demand | $/hr Spot | Est. Train Time | Cost (On-Demand) | Cost (Spot) | Efficiency |
|------|----------------|-----------|-----------------|------------------|-------------|------------|
| v5litepod-1 | $1.38 | ~$0.41 | ~18-20 hrs | ~$27 | ~$8 | Best $/perf |
| v5litepod-4 | $5.50 | ~$1.65 | ~5-6 hrs | ~$30 | ~$9 | Good balance |
| v5litepod-8 | $11.00 | ~$3.30 | ~3-4 hrs | ~$38 | ~$12 | Fastest |
| v3-8 | $8.00 | ~$2.40 | ~8-10 hrs | ~$72 | ~$22 | Slower |
| v2-32 | ~$32.00 | ~$9.60 | ~4-5 hrs | ~$144 | ~$43 | Overkill |

## Cost Efficiency Notes

- **Cost Efficiency** = (Performance per hour) / (Cost per hour)
- v5litepod-1 is ~2.5x more cost-efficient than v3-8
- Spot/preemptible pricing is ~70% cheaper but can be interrupted

## Recommendations

| Use Case | TPU | Est. Cost | Notes |
|----------|-----|-----------|-------|
| Budget-conscious | v5litepod-1 spot | ~$8 | Longest time (~18-20 hrs) |
| Balance speed/cost | v5litepod-4 spot | ~$9 | Good middle ground (~5-6 hrs) |
| Fastest training | v5litepod-8 on-demand | ~$38 | Done in 3-4 hrs |

## Free Option

Kaggle provides free TPU v5e (8 chips) with:
- 20 hours/month quota
- 9-hour daily limit

## TPU Availability by Zone

Common zones for TPU availability:
- `us-central1-a`
- `us-central1-b`
- `us-central1-f`
- `europe-west4-a`

## Sources

- [Google Cloud TPU Pricing](https://cloud.google.com/tpu/pricing)
- [TPU Regions & Zones](https://cloud.google.com/tpu/docs/regions)
