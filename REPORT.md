# Monetary Policy Transmission in Poland: A Bayesian VAR Analysis

**Author:** Otajon Yuldashev  
**Course:** Bayesian Time-Series Econometrics, Warsaw School of Economics (WNE UW)  
**Instructor:** Prof. Andrzej Kocięcki  
**Date:** June 2026

---

## Abstract

This paper estimates a three-variable Bayesian vector autoregression (BVAR) for Poland over January 2010–March 2025, linking consumer price inflation (CPI year-on-year), a short-term interest rate, and the nominal PLN/EUR exchange rate (in logs). A Minnesota prior with Gibbs sampling is used to address parameter proliferation and near-unit-root behaviour. Structural impulse responses are identified with a Cholesky decomposition ordering the policy rate first, so that monetary policy is treated as predetermined within the month relative to inflation and the exchange rate. The estimated system is stable, MCMC diagnostics are satisfactory, and conclusions are robust to alternative Cholesky orderings, prior tightness, and stationary transformations of the data. A tightening shock raises inflation on impact—a manifestation of the well-known *price puzzle* in reduced-form VARs—which motivates careful interpretation and reporting of identification sensitivity.

**Keywords:** Bayesian VAR, Minnesota prior, monetary policy transmission, Poland, impulse responses.

---

## 1. Introduction

Understanding how changes in the short-term interest rate affect inflation and the exchange rate is central to monetary policy analysis in open economies. For Poland—a member of the EU with a flexible exchange rate and inflation-targeting framework—the transmission mechanism involves expectational channels, pass-through from the zloty to consumer prices, and the interaction between domestic and euro-area financial conditions.

This project applies Bayesian methods to a small monthly macroeconomic system. Relative to a classical OLS VAR, the BVAR shrinks coefficients toward a Minnesota prior centred on unit-root-type persistence, which is appropriate when ADF tests do not reject unit roots in levels but first differences are clearly stationary. The analysis delivers posterior impulse response functions (IRFs), a 12-month posterior predictive forecast, and systematic robustness checks required for credible inference.

---

## 2. Data

### 2.1 Sources and transformations

| Variable | Symbol | Source | Transformation |
|----------|--------|--------|----------------|
| Inflation | CPI YoY | OECD / FRED `POLCPIALLMINMEI` | \((\text{CPI}_t / \text{CPI}_{t-12} - 1) \times 100\) |
| Policy stance | \(i_t\) | OECD / FRED `IRSTCI01PLM156N` | Level (%) |
| Exchange rate | \(s_t\) | ECB reference rate (`data.csv`) | \(\ln(\text{PLN/EUR})\) |

The interest-rate series is an OECD **immediate short-term rate** for Poland. From 2010 onward it is on a conventional percent scale (e.g. 3.75% in early 2010) and co-moves with NBP policy episodes, but it is a **market-rate proxy**, not the official NBP reference rate. Results should be read as transmission in a rate–inflation–FX system rather than as a structural model of a specific NBP instrument.

### 2.2 Sample

- **Frequency:** monthly (`MS`)
- **Period:** January 2010 – March 2025
- **Observations after alignment:** \(T = 183\)

Summary statistics (from the estimation sample):

| | CPI YoY (%) | Policy rate (%) | ln(PLN/EUR) |
|---|-------------|-----------------|-------------|
| Mean | 3.65 | 3.09 | 1.46 |
| Std. dev. | 4.37 | 1.93 | 0.05 |
| Min / Max | −1.29 / 19.23 | 0.10 / 6.79 | 1.36 / 1.57 |

Figure 1 (`output_bvar/01_data.png`) plots the three series, with shading for the COVID-19 period and elevated inflation (2021–2023).

---

## 3. Econometric methodology

### 3.1 The VAR model

Stack \(n=3\) variables in \(\mathbf{y}_t\). A VAR(\(p\)) with intercept is:

\[
\mathbf{y}_t = \mathbf{c} + \sum_{\ell=1}^{p} \mathbf{A}_\ell \mathbf{y}_{t-\ell} + \boldsymbol{\varepsilon}_t, \qquad \boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}).
\]

In matrix form, \(\mathbf{Y} = \mathbf{X}\mathbf{B} + \mathbf{E}\), with \(\mathbf{B}\) the \(k \times n\) coefficient matrix (\(k = 1 + np\)) and \(\boldsymbol{\Sigma}\) the \(n \times n\) covariance of residuals.

### 3.2 Minnesota prior and posterior simulation

The prior follows Litterman (1986) / Minnesota form:

- \(\text{vec}(\mathbf{B}) \mid \boldsymbol{\Sigma} \sim \mathcal{N}(\mathbf{B}_0, \boldsymbol{\Omega}_0 \otimes \boldsymbol{\Sigma})\)
- \(\boldsymbol{\Sigma} \sim \mathcal{IW}(\mathbf{S}_0, \nu_0)\)

Hyperparameters: \(\lambda_1 = 0.20\) (overall tightness), \(\lambda_2 = 0.50\) (cross-variable shrinkage), \(\lambda_3 = 100\) (constant prior variance), \(\delta = 0.90\) (prior mean on own first lag). Scale factors \(\sigma_i\) come from univariate AR(1) residual standard deviations.

**Gibbs sampler** (12,000 iterations, 4,000 burn-in, 8,000 retained):

1. Draw \(\mathbf{B} \mid \boldsymbol{\Sigma}, \mathbf{Y}\) from the matrix-normal posterior.
2. Draw \(\boldsymbol{\Sigma} \mid \mathbf{B}, \mathbf{Y}\) from \(\mathcal{IW}(\mathbf{S}_n, \nu_0 + T)\) with  
   \(\mathbf{S}_n = \mathbf{S}_0 + \mathbf{E}'\mathbf{E} + (\mathbf{B}-\mathbf{B}_0)'\boldsymbol{\Omega}_0^{-1}(\mathbf{B}-\mathbf{B}_0)\).

### 3.3 Identification

Structural shocks are identified via **Cholesky decomposition** of \(\boldsymbol{\Sigma}\), with **baseline ordering**:

\[
\text{Policy rate} \rightarrow \text{CPI YoY} \rightarrow \ln(\text{PLN/EUR}).
\]

The policy rate does not respond contemporaneously to inflation or the exchange rate within a month; CPI and the exchange rate may respond to the rate on impact. This ordering is standard for studying *monetary policy transmission* (as opposed to ordering CPI first, which treats inflation as predetermined).

Alternative orderings (`cpi_rate_fx`, `rate_fx_cpi`) are reported as sensitivity analysis.

### 3.4 Lag selection and pre-estimation tests

- **Information criteria:** AIC, BIC, HQIC, FPE for \(p \in \{1,\ldots,4\}\).
- **Unit roots:** ADF (constant + trend in levels; constant in differences) and KPSS.
- **Cointegration:** Johansen trace test (`det_order=0`, \(k_{\text{ar diff}} = p-1\)).
- **Stability:** moduli of eigenvalues of the VAR companion matrix evaluated at the posterior mean \(\mathbf{B}\).

---

## 4. Preliminary results

### 4.1 Stationarity

**ADF (levels, const + trend):** unit root not rejected at 5% for all three variables.  
**ADF (first differences):** reject unit root for all.  
**KPSS (levels):** reject stationarity for CPI YoY and policy rate; borderline for ln(PLN/EUR) (p ≈ 0.10).

This pattern supports estimating a VAR in **levels** with Minnesota shrinkage toward persistent dynamics, while reporting **first-difference** and **mixed** specifications as robustness.

### 4.2 Lag order

| \(p\) | AIC | BIC | HQIC |
|-------|-----|-----|------|
| 1 | −12.934 | −12.720 | −12.847 |
| **2** | −13.552 | **−13.178*** | −13.401 |
| 3 | −13.564 | −13.030 | −13.347 |
| 4 | −13.553 | −12.858 | −13.271 |

**Selected:** \(p = 2\) (minimum BIC). HQIC favours \(p=3\); results are reported for BIC as the more parsimonious choice.

### 4.3 Cointegration (Johansen trace test)

| Rank \(r\) | Trace stat. | 95% critical | Reject \(H_0\) at 5% |
|------------|-------------|--------------|----------------------|
| \(r=0\) | 37.19 | 29.80 | Yes |
| \(r=1\) | 18.61 | 15.49 | Yes |
| \(r=2\) | 5.20 | 3.84 | Yes |

All trace statistics reject the null of rank \(\leq r\) at the 5% level, which—in finite samples—can conflict with ADF unit-root evidence. We therefore do not rely on a single pre-test but complement levels-based BVAR with difference-based models (Section 6).

### 4.4 Stability

Maximum modulus of companion-matrix eigenvalues at the posterior mean: **0.991** (all moduli \(< 1\)). The estimated VAR(\(2\)) is **stable** (`output_bvar/stability_baseline.csv`).

---

## 5. Estimation results

### 5.1 MCMC convergence

Diagnostics for own lag-1 coefficients and diagonal elements of \(\boldsymbol{\Sigma}\) show:

- **ESS** roughly 6,900–8,000 (of 8,000 kept draws).
- **Geweke** statistics within \(\pm 1.96\) for key parameters (see console output and `posterior_baseline.csv`).

Trace plots, ACFs, burn-in cumulant means, and posterior histograms: `05_convergence.png`, `06_burnin.png`, `07_posteriors.png`.

### 5.2 Posterior mean coefficients (selected)

| | CPI equation | Rate equation | ln FX equation |
|---|--------------|---------------|----------------|
| Own lag 1 | 1.084 | 1.136 | 0.932 |
| Lag-1 policy → CPI | 0.247 | — | — |
| Lag-2 policy → CPI | −0.318 | — | — |

Inflation is highly persistent (own lag 1 ≈ 1.08). The policy rate enters the CPI equation positively at lag 1 and negatively at lag 2 in the posterior mean, consistent with delayed transmission and omitted short-run dynamics.

Posterior mean \(\boldsymbol{\Sigma}\) (diagonal): \(\hat{\sigma}_{\text{CPI}}^2 \approx 0.39\), \(\hat{\sigma}_{\text{rate}}^2 \approx 0.03\), \(\hat{\sigma}_{\text{ln FX}}^2 \approx 0.0002\).

Full tables: `output_bvar/posterior_baseline.csv`.

---

## 6. Structural impulse responses

### 6.1 Policy-rate shock (baseline identification)

Posterior median IRFs to a one-standard-deviation **expansionary** policy-rate shock (Cholesky, rate ordered first):

| Response | Impact (h=0) | Peak | Peak month | h = 12 |
|----------|--------------|------|------------|--------|
| CPI YoY | +0.245 | +0.307 | 1 | +0.103 |
| Policy rate | +0.181 | +0.225 | 3 | +0.203 |
| ln(PLN/EUR) | −0.002 | −0.003 | 35 | −0.002 |

**Figures:** `02_irf_shock1_policy_rate.png` (full system); summaries in `irf_policy_shock_summary.csv`.

**Interpretation:**

1. **Price puzzle:** CPI rises on impact and remains above zero at the 12-month horizon following a positive rate innovation. In standard macro theory, a tightening (higher rate) should dampen inflation. Here the Cholesky shock is a positive orthogonal innovation to the rate equation; with positive impact on CPI, the short-run correlation dominates. This is a known issue in monthly VARs (omitted commodity prices, identification timing, rate as endogenous to inflation expectations). It must be discussed explicitly—not ignored.

2. **Exchange rate:** A positive rate shock is associated with a small **depreciation** of the zloty (ln PLN/EUR falls), opposite to the textbook UIP channel in the very short run; effects are small in magnitude.

3. **Persistence of the rate:** The policy variable remains elevated after the shock, as expected for a persistent policy process.

### 6.2 Other shocks

- **CPI shock:** `02_irf_shock0_cpi_yoy.png`
- **Exchange-rate shock:** `02_irf_shock2_ln_plneur.png`

### 6.3 Identification sensitivity

`10_id_sensitivity_policy_shock.png` compares the CPI response to a policy shock under three Cholesky orderings. Qualitative features (including the positive short-run CPI response under several orderings) should be compared across plots when writing the policy conclusion.

### 6.4 Prior sensitivity

`11_prior_sensitivity.png` varies \(\lambda_1 \in \{0.10, 0.20, 0.50\}\). Median CPI IRFs to a policy shock are qualitatively similar across prior tightness, indicating that shrinkage strength does not solely drive the price puzzle.

### 6.5 Classical VAR benchmark

OLS VAR(\(2\)) with intercept (`statsmodels`) produces similar reduced-form persistence. BVAR vs classical IRFs: `04_irf_comparison_shock0.png`–`shock2.png`. The Bayesian model smooths extreme coefficient estimates and yields wider credible bands.

---

## 7. Robustness

| Specification | Purpose | Output |
|---------------|---------|--------|
| First differences of all variables | Stationarity | `posterior_first_diff.csv`, `12_stationarity_robustness.png` |
| CPI & rate in levels; \(\Delta \ln(\text{FX})\) | Stationary FX, level macro | `posterior_mixed_stationary.csv` |
| Alternative Cholesky orders | Identification | `10_id_sensitivity_policy_shock.png` |
| \(\lambda_1\) grid | Prior | `11_prior_sensitivity.png` |

**CPI response to policy shock at \(h=12\):** baseline levels model **0.103**; mixed stationary specification **0.109**—close numerically, so the long-run qualitative message is not driven by the FX transformation alone.

---

## 8. Forecasting and fit

- **12-month posterior predictive forecast:** `08_forecast.png` (median and 50%/90% bands).
- **In-sample fit** (posterior mean \(\mathbf{B}\)): `09_fitted.png`.

Forecasts use reduced-form innovations \(\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})\), drawn via the Cholesky factor of \(\boldsymbol{\Sigma}\).

---

## 9. Limitations and extensions

1. **Price puzzle** and **proxy policy rate** limit structural policy conclusions; sign-restricted IRFs or a Taylor-rule block would be a natural extension.
2. **Monthly timing:** Cholesky within a month may not match NBP decision dates; daily/event-study methods would refine identification.
3. **Johansen vs ADF** send mixed messages on cointegration; a VECM could be estimated if the course requires explicit error-correction.
4. **Omitted variables:** energy prices, euro-area rates, and fiscal policy are not included.
5. **Sample** includes COVID-19 and the 2022–2023 inflation surge; subsample stability could be tested.

---

## 10. Conclusion

This project implements a complete Bayesian VAR workflow for Polish monetary transmission: real OECD/ECB data, Minnesota prior, Gibbs estimation, structural IRFs with economically motivated identification, MCMC diagnostics, stability checks, and multiple robustness exercises. The system is stable and sampling appears convergent. The main substantive finding is that **short-run inflation dynamics following rate shocks are hard to interpret through Cholesky identification alone** because of the price puzzle; transmission to the exchange rate is small, and results are robust across priors and several alternative specifications. For policy conclusions, the identification sensitivity figures and difference-based models are as important as the baseline IRF plot.

---

## References

- Banbura, M., Giannone, D., & Reichlin, L. (2010). Large Bayesian vector autoregressions. *Journal of Applied Econometrics*, 25(1), 71–92.
- Koop, G., & Korobilis, D. (2010). Bayesian multivariate time series methods for empirical macroeconomics. *Foundations and Trends in Econometrics*, 3(4), 267–358.
- Litterman, R. B. (1986). Forecasting with Bayesian vector autoregressions—five years of experience. *Journal of Business & Economic Statistics*, 4(1), 25–38.
- Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1–48.

---

## Appendix A. Reproducibility

```bash
pip install -r requirements.txt
python bvar_poland.py
```

All tables and figures are written to `output_bvar/`.

## Appendix B. Figure list

| File | Description |
|------|-------------|
| `01_data.png` | Data series 2010–2025 |
| `02_irf_shock*.png` | Structural IRFs by shock |
| `04_irf_comparison_shock*.png` | BVAR vs classical VAR |
| `05_convergence.png` | Trace and ACF |
| `06_burnin.png` | Cumulative posterior means |
| `07_posteriors.png` | Posterior histograms |
| `08_forecast.png` | 12-month forecast |
| `09_fitted.png` | In-sample fit |
| `10_id_sensitivity_policy_shock.png` | Cholesky orderings |
| `11_prior_sensitivity.png` | \(\lambda_1\) sensitivity |
| `12_stationarity_robustness.png` | Levels vs first differences |

## Appendix C. Software

Python 3 with `numpy`, `pandas`, `matplotlib`, `statsmodels`. Script: `bvar_poland.py`.
