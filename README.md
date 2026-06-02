# Monetary Policy Transmission in Poland — Bayesian VAR

Bayesian vector autoregression (BVAR) for **Polish inflation**, the **short-term interest rate**, and the **nominal PLN/EUR exchange rate**, with structural impulse responses for monetary policy transmission.

**Course:** Bayesian Time-Series Econometrics, WNE UW  
**Instructor:** Prof. Andrzej Kocięcki  
**Author:** Otajon Yuldashev (2026)

## Research question

How does an **exogenous policy-rate shock** propagate to **CPI inflation** and the **exchange rate** in Poland (2010–2025)?

## Data

| Variable | File | Source | Transformation |
|----------|------|--------|----------------|
| CPI | `POLCPIALLMINMEI.csv` | OECD / FRED | Year-on-year % change from index |
| Policy rate | `IRSTCI01PLM156N.csv` | OECD / FRED | Level (%); sample from 2010 when series is on a percent scale |
| Exchange rate | `data.csv` | ECB | **ln(PLN/EUR)** monthly average |

Sample: **January 2010 – March 2025** (monthly).

## Methodology (summary)

1. **Stationarity:** ADF and KPSS on levels and first differences; **Johansen** trace test for cointegration among levels.
2. **Lag order:** BIC over \(p \in \{1,\ldots,4\}\) on the baseline specification.
3. **Estimation:** BVAR with **Minnesota prior** (Litterman-style shrinkage toward independent unit-root-type dynamics) and **Gibbs sampling** from the normal–inverse-Wishart posterior:
   - \(B \mid \Sigma, Y \sim \mathcal{MN}(B_n, \Omega_n, \Sigma)\)
   - \(\Sigma \mid B, Y \sim \mathcal{IW}(S_n, \nu_0 + T)\)
4. **Identification:** **Cholesky** ordering with the **policy rate first** (no contemporaneous response of the rate to CPI or FX within a month), then CPI, then ln(FX). Alternative orderings are reported as sensitivity.
5. **Inference:** Posterior IRFs (median + 68% / 90% bands), MCMC diagnostics (ESS, Geweke), stability of the **companion matrix**.
6. **Robustness:** Full first-difference VAR; mixed specification (CPI & rate in levels, \(\Delta \ln\) FX); prior \(\lambda_1\) sensitivity; classical OLS VAR benchmark.

Levels with Minnesota shrinkage are standard when series are near unit root but not necessarily cointegrated; Johansen and difference-based models check that conclusions are not driven by a single specification.

## Reproduce results

```bash
pip install -r requirements.txt
python bvar_poland.py
```

Outputs are written to `output_bvar/` (tables and figures).

### Main output files

| File | Content |
|------|---------|
| `stationarity_tests.csv` | ADF + KPSS |
| `cointegration_johansen.csv` | Johansen trace / max-eigenvalue tests |
| `lag_selection.csv` | Information criteria |
| `posterior_baseline.csv` | Posterior summaries + MCMC diagnostics |
| `stability_baseline.csv` | Companion-matrix eigenvalue moduli |
| `irf_policy_shock_summary.csv` | Peak / impact CPI & FX responses to policy shock |
| `02_irf_shock*.png` | Structural IRFs by shock |
| `04_irf_comparison_*.png` | BVAR vs classical VAR |

## Interpretation notes

- **Policy-rate IRF:** With rate ordered first, the rate equation’s own shock is the monetary policy innovation on impact.
- **ln(PLN/EUR):** A positive IRF is an **appreciation** of the zloty (more PLN per euro).
- **CPI in levels (YoY %):** IRFs are in percentage points; persistence partly reflects inflation dynamics already in YoY form.

## Written report

The full academic write-up (introduction, methodology, results, limitations, references) is in **[REPORT.md](REPORT.md)**. Export to PDF via Pandoc, VS Code, or print-to-PDF if your course requires a PDF submission.

## License

Academic use; cite data providers (OECD/FRED, ECB) when reusing the series.
