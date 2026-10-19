# Dataset Generation Notes — MME606A Assessment 2 Paper A

**Student:** STUDENT_NAME (`STUDENT_ID`)  
**Dataset:** `manufacturing_operations.csv` (synthetic, not from public sources)

## Scenario

The dataset models **monthly operational records** for **85 machines** across three Victorian manufacturing plants (`Plant_A_Melbourne`, `Plant_B_Geelong`, `Plant_C_Ballarat`). Machine families include CNC, Lathe, Milling, 3D Printer, Press, and Robot Arm. Each row is one machine in one calendar month from January 2022 through February 2027 (62 months → **5,270 unique** machine–month rows after deduplication; **5,282** raw rows including deliberate duplicates).

## Required variables

All assessment-mandated columns are present. Additional columns (`Record_Month`, `Output_Units`, `Age_Years`) support feature engineering and dashboard filtering.

| Variable | Generation logic (summary) |
|----------|---------------------------|
| `Operating_Hours` | Normal draw scaled by machine profile, seasonal factor (±8%), and machine reliability |
| `Power_Consumption (kWh)` | Operating hours × type-specific kWh/hr with log-normal noise and age/reliability uplift |
| `Temperature (°C)` | Base type temperature + load effect + reliability degradation |
| `Vibration_Level (mm/s)` | Function of hours, temperature excursion, gamma noise, reliability |
| `Maintenance_Cost (USD)` | Base + fault count × repair band + downtime labour + age factor |
| `Downtime (Hours)` | Poisson faults × repair duration + exponential unplanned component |
| `Faults_Reported` | Poisson rate increasing with vibration and poor reliability |
| `Efficiency (%)` | Clipped function of output per hour, vibration, temperature |
| `Last_Maintenance_Date` | Synthetic date 5–120 days before month start |
| `Predicted_Failure` | Composite risk score vs threshold (~22% `Yes`) |

## Failure label (engineering rule)

A latent **failure score** combines vibration, faults, downtime, efficiency deficit, maintenance interval, and machine reliability:

\[
\text{score} = 0.30\frac{vib}{12} + 0.22\min(\frac{faults}{3},1.5) + \cdots - 0.20 \cdot R_{machine} + \mathcal{N}(0,0.12)
\]

`Predicted_Failure = Yes` if score > 0.52. This mimics a maintenance engineer’s risk flag rather than a confirmed breakdown.

## Imperfections (intentional)

- **Missing values (~1.5–2.5%** per sensor/economic field): `Temperature`, `Vibration`, `Maintenance_Cost`, `Efficiency`
- **Outliers:** power spikes (1.8–2.6×), vibration spikes, extreme downtime (80–160 h)
- **Data entry errors:** negative downtime on 8 rows
- **Duplicate machine–month rows:** 12 injected copies
- **Gaussian noise** on operating hours

## Limitations

- Monthly aggregation hides intra-month failure sequences.
- Labels are rule-based, so models learn the generative story plus noise.
- Plants are fictitious; correlations are plausible but not validated against shop-floor data.

## Reproduction

```bash
python3 scripts/01_generate_data.py
```

Random seed: `42` (`numpy.random.default_rng`).
