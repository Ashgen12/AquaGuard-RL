---
title: AquaGuard-RL
emoji: 💧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - groundwater
  - agriculture
  - india
---

# AquaGuard-RL

India Groundwater and Agricultural Resource Management RL Environment  
Meta PyTorch OpenEnv Hackathon - Round 1 Submission

## The Problem This Solves

India's groundwater aquifers are being drained at catastrophic rates:

| Metric | Value | Source |
|---|---:|---|
| Groundwater extraction increase (1990-2020) | +500% | Nature, 2024 |
| Average aquifer level decline (North India) | 8 meters | CGWB Annual Report 2023 |
| Indian districts classified as water-stressed | 60% | NITI Aayog Water Report |
| Rice water requirement per kg produced | 3,500-5,000 L | FAO AQUASTAT |
| Millet water requirement per kg produced | 500-800 L | FAO AQUASTAT |

The root cause: Green Revolution subsidy dynamics (MSP policy) lock farmers into water-intensive rice and wheat cultivation. Switching is often economically irrational in the short term, but continuing current patterns depletes aquifers and increases long-term collapse risk.

AquaGuard-RL forces an agent to discover policy transitions that reduce water stress without collapsing farmer income or food security.

## What AquaGuard-RL Is

AquaGuard-RL is a text-based mini-RL environment for OpenEnv.  
The agent acts as a District Agricultural Commissioner making seasonal policy decisions across a 3-zone agricultural system.

The agent controls:
- Crop area allocation (rice/wheat/millet/pulses/oilseeds/vegetables)
- Water quotas per zone (mm/season)
- Irrigation method per zone (flood/sprinkler/drip)
- Groundwater extraction limits per zone (m/season)
- Subsidy adjustments per crop (relative MSP change)
- Natural language justification (used by LLM grader)

The agent observes:
- Groundwater depth per zone
- Soil fertility and salinity per zone
- Crop yields, MSP prices, demand
- Farmer income and poverty fraction
- Food security ratio (production/requirement)
- Rainfall forecast and temperature anomaly
- Shannon diversity index
- Full natural language scenario description

## Project Structure

```text
AquaGuard-RL/
├── inference.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── openenv.yaml
├── uv.lock
├── .env.example
├── server/
│   └── app.py                  # OpenEnv entry point (wraps src/server/app.py)
├── src/
│   ├── models.py
│   ├── constants.py
│   ├── client.py
│   └── server/
│       ├── app.py
│       ├── aquaguard_environment.py
│       ├── reward.py
│       ├── simulation/
│       │   ├── groundwater.py
│       │   ├── crop_growth.py
│       │   ├── economic.py
│       │   └── season.py
│       ├── grader/
│       │   ├── programmatic.py
│       │   └── llm_grader.py
│       ├── tasks/
│       │   └── task_definitions.py
│       └── utils/
│           └── description_builder.py
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_environment.py
│   ├── test_simulation.py
│   ├── test_reward.py
│   └── test_grader.py
└── scripts/
    ├── run_local.sh
    ├── build_docker.sh
    └── run_sample_agent.py
```

## Quick Start

### Option 1: Local (No Docker)

```bash
git clone https://github.com/ashgen12/AquaGuard-RL
cd AquaGuard-RL

pip install -r requirements.txt

# Linux/macOS
export PYTHONPATH=src
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

PowerShell:

```powershell
$env:PYTHONPATH="src"
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run demo agent in another terminal:

```bash
python scripts/run_sample_agent.py --task baseline
```

### Option 2: Docker

```bash
docker build -t aquaguard-env:latest .
docker run -p 8000:8000 aquaguard-env:latest
```

Test:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task":"baseline","seed":42}'
```

### Option 3: Run Inference (LLM Agent)

```bash
# NVIDIA NIM API (free — recommended, get key at build.nvidia.com)
export API_BASE_URL="https://integrate.api.nvidia.com/v1"
export MODEL_NAME="nvidia/nemotron-3-super-120b-a12b"
export OPENAI_API_KEY="nvapi-..."
export ENV_SERVER_URL="http://localhost:8000"

python inference.py
```

Heuristic fallback (no LLM API):

```bash
python inference.py --heuristic
```

### Option 4: Hugging Face Spaces (Hackathon Submission)

To deploy this environment to Hugging Face Spaces (required for the hackathon):

1. **Login to Hugging Face CLI**:
   ```bash
   huggingface-cli login
   ```
2. **Push the Environment**:
   ```bash
   openenv push
   ```
   *Follow the prompts to name your space and select the hardware.*
3. **Live HF Space**: https://huggingface.co/spaces/Ashgen12/AquaGuard-RL
   Once deployed, you can run the agent against the live URL:
   ```bash
   python scripts/run_sample_agent.py --server https://ashgen12-aquaguard-rl.hf.space
   ```
   For testing all the easy, medium and hard modes as specified below, run this:
   ```bash
   set ENV_SERVER_URL=https://ashgen12-aquaguard-rl.hf.space
   python inference.py --heuristic --tasks baseline crisis policy_shift
   ```

## HTTP API

| Endpoint | Method | Description |
|---|---|---|
| /reset | POST | Start new episode. Body: {"task":"baseline","seed":42} |
| /step | POST | Execute one action. Body: AquaGuardAction JSON |
| /state | GET | Episode metadata and cumulative stats |
| /health | GET | Health check |
| /info | GET | Environment metadata |

## Action Space

`AquaGuardAction` (one per growing season):

```json
{
  "crop_allocation": {
    "rice": 0.20,
    "wheat": 0.25,
    "millet": 0.25,
    "pulses": 0.15,
    "oilseeds": 0.10,
    "vegetables": 0.05
  },
  "water_quotas": {
    "zone_a": 750,
    "zone_b": 700,
    "zone_c": 600
  },
  "irrigation_methods": {
    "zone_a": "drip",
    "zone_b": "sprinkler",
    "zone_c": "drip"
  },
  "extraction_limits": {
    "zone_a": 20.0,
    "zone_b": 18.0,
    "zone_c": 12.0
  },
  "subsidy_adjustments": {
    "rice": -0.15,
    "wheat": -0.05,
    "millet": 0.10,
    "pulses": 0.10,
    "oilseeds": 0.05,
    "vegetables": 0.0
  },
  "justification": "Reducing rice because it needs far more water than millet while protecting food and income."
}
```

Constraints:
- `crop_allocation` sum <= 1.0
- `water_quotas` in [0, 2000] per zone
- `irrigation_methods` in {flood, sprinkler, drip}
- `extraction_limits` in [0, 60] per zone
- `subsidy_adjustments` in [-1.0, 1.0]
- `justification` max length: 2000 chars

## Reward Function

Multi-objective weighted reward:

```text
R = w_gw*R_gw + w_food*R_food + w_income*R_income + w_diversity*R_diversity
```

Default weights:
- Groundwater: 0.35
- Food security: 0.30
- Farmer income: 0.25
- Diversity: 0.10

Scaled to `[-10, +10]` with hard penalties for catastrophic failures.

## Tasks

| Task | Difficulty | Max Steps | Initial GW Depths | Challenge |
|---|---|---:|---|---|
| baseline | EASY | 10 | A:22m, B:26m, C:30m | Maintain all zones |
| crisis | HARD | 12 | A:30m, B:35m, C:37m | Recover near-critical zone |
| policy_shift | MEDIUM | 8 | A:28m, B:30m, C:32m | Increase diversity without income crash |
| climate_shock | VERY_HARD | 6 | A:20m, B:24m, C:28m | Survive drought shock |
| multi_district | EXPERT | 15 | A:18m, B:28m, C:36m | Equity and shared aquifer coordination |

## Grader

Two-stage scoring:

```text
Total Score = 0.60 * Programmatic Score + 0.40 * LLM Score
```

Programmatic grader runs 12 automated checks.  
LLM grader evaluates reasoning quality on 5 dimensions:
- Causal reasoning
- Trade-off acknowledgment
- Domain knowledge
- Policy coherence
- Risk awareness

## Running Tests

```bash
pytest tests/ -v --tb=short
```
## Result of baseline:

```bash
python scripts/run_sample_agent.py --server https://ashgen12-aquaguard-rl.hf.space

AquaGuard-RL Demo — Task: baseline (seed=42)
============================================================
Connecting to: https://ashgen12-aquaguard-rl.hf.space

Task: baseline | Season: kharif | Year 1
Initial GW: 26.0m | Food: 7.24 | Poverty: 55.0%
Shannon diversity: 1.583

Step  1 [rabi   Y1]: reward=+1.54 | GW=26.9m | food=5.22 | poverty=84.7% | H=1.690
          → The justification correctly notes the aquifer's vulnerability and links crop shifts to water savings
Step  2 [zaid   Y1]: reward=+1.17 | GW=28.6m | food=4.67 | poverty=94.2% | H=1.690
          → The justification correctly notes the aquifer warning and links crop shifts to water savings, but it
Step  3 [kharif Y2]: reward=+1.00 | GW=30.3m | food=4.65 | poverty=91.2% | H=1.690
          → The justification correctly notes aquifer stress and links crop shifts to water savings, but it over
Step  4 [rabi   Y2]: reward=+3.75 | GW=30.7m | food=4.57 | poverty=57.6% | H=1.670
          → The justification correctly identifies the aquifer depletion and links specific policy actions (rice
Step  5 [zaid   Y2]: reward=+3.83 | GW=31.6m | food=4.36 | poverty=49.8% | H=1.670
          → The justification correctly links high aquifer depth to water‑saving actions and ties subsidy shifts
Step  6 [kharif Y3]: reward=+4.99 | GW=32.6m | food=4.34 | poverty=22.9% | H=1.670
          → The justification correctly identifies the aquifer crisis and links specific policy levers (crop shi
Step  7 [rabi   Y3]: reward=+5.86 | GW=32.9m | food=4.72 | poverty=10.0% | H=1.670
          → The justification correctly links the aquifer's critical state to specific water-saving actions (ric
Step  8 [zaid   Y3]: reward=+4.33 | GW=33.9m | food=4.37 | poverty=4.7% | H=1.670
          → The justification correctly links the aquifer's critical state to specific water-saving actions (ric
Step  9 [kharif Y4]: reward=+4.25 | GW=35.0m | food=4.34 | poverty=3.3% | H=1.670
          → The justification correctly identifies the aquifer crisis and links specific policy levers (crop shi
Step 10 [rabi   Y4]: reward=+4.58 | GW=35.1m | food=4.83 | poverty=1.7% | H=1.670
          → The justification correctly identifies the aquifer crisis and links rice reduction and drip irrigati

============================================================
Episode complete: 10 seasons
Total reward:               35.29
Final GW depth (avg):        35.1 m
Final food security:         4.83
Final poverty:                1.7 %
Best Shannon diversity:     1.690
Food security failures:         0
GW crisis triggered:        False
```

## Data Sources

- FAO AQUASTAT: https://www.fao.org/aquastat/en/
- CGWB Annual Report 2023: https://cgwb.gov.in/en/reports
- GoI MSP / CACP: https://cacp.dacnet.nic.in
- NSSO SAS 2021: https://mospi.gov.in/web/mospi
- Nature 2024 groundwater paper: https://www.nature.com/articles/s41586-024-07349-8
- NITI Aayog water stress reports: https://niti.gov.in

## Social Impact

This environment models one of India's highest-impact policy problems: balancing groundwater sustainability, food security, and farmer livelihoods under climate and market uncertainty.

## License

MIT License.
