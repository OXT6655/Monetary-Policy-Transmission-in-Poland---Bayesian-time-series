#!/usr/bin/env python3
"""
Bayesian VAR Analysis: Monetary Policy, Inflation & Exchange Rate in Poland
============================================================================
Course : Bayesian Time-Series Econometrics — WNE UW
Instructor: Prof. Andrzej Kocięcki
Author : Otajon Yuldashev, 2026

DATA SOURCES (real, official)
------------------------------
  CPI YoY    : OECD / FRED series POLCPIALLMINMEI (CPI index 2015=100, monthly)
               YoY computed as (CPI_t / CPI_{t-12} - 1)*100
  Policy Rate: OECD / FRED series IRSTCI01PLM156N (short-term interest rate, %)
  PLN/EUR    : ECB reference exchange rate, monthly average

SAMPLE: January 2010 – March 2025

METHODOLOGY
-----------
Model    :  Y = X B + E,    E ~ MN(0, I_T, Sigma)
Prior    :  B|Sigma ~ MN(B0, Omega0, Sigma),   Sigma ~ IW(S0, nu0)   [Minnesota / NIW]
Posterior:  Gibbs sampler
            • B   | Sigma, Y ~ MN(Bn, Omegan, Sigma)
            • Sigma | B, Y ~ IW(S0 + E'E + (B-B0)'Omega0^-1(B-B0), nu0+T+k)
Ident.   :  Cholesky structural identification (baseline: CPI, Rate, PLN/EUR)
            + sensitivity to alternative orderings
Diagnost.:  ESS (Geyer IMS), Geweke Z, trace / ACF / burn-in / posterior plots
Forecast :  12-month posterior predictive
Benchmark:  Classical OLS VAR via statsmodels
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*unsupported index.*')

np.random.seed(2024)
plt.rcParams.update({'figure.dpi': 120, 'font.size': 9,
                     'axes.titlesize': 9, 'axes.labelsize': 8,
                     'legend.fontsize': 7})

COLORS  = ['#A23B72', '#2E86AB', '#F18F01']
VARLAB  = ['CPI YoY (%)', 'Policy Rate (%)', 'PLN/EUR']
VARNAME = ['cpi_yoy', 'policy_rate', 'plneur']
OUT     = 'output_bvar'
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Cholesky orderings for identification sensitivity (indices into [CPI, Rate, FX])
BASE_ORDER = [0, 1, 2]
ID_ORDERS = {
    'cpi_rate_fx': [0, 1, 2],
    'rate_cpi_fx': [1, 0, 2],
    'rate_fx_cpi': [1, 2, 0],
}
PRIOR_LAMBDAS = [0.10, 0.20, 0.50]


def configure_console():
    """Use UTF-8 on Windows so box-drawing characters in the console."""
    os.makedirs(OUT, exist_ok=True)
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except (OSError, ValueError):
            pass


def ok_mark(geweke_z):
    return 'OK' if abs(geweke_z) < 1.96 else 'FAIL'


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------

def build_data() -> pd.DataFrame:
    """Load and align monthly data, 2010-01 through 2025-03."""
    fx = pd.read_csv(f'{DATA_DIR}/data.csv')
    fx['date'] = pd.to_datetime(fx['TIME_PERIOD'])
    fx = (fx[['date', 'OBS_VALUE']]
          .rename(columns={'OBS_VALUE': 'plneur'})
          .set_index('date').sort_index())

    cpi_raw = (pd.read_csv(f'{DATA_DIR}/POLCPIALLMINMEI.csv',
                            parse_dates=['observation_date'])
               .set_index('observation_date')
               .sort_index()['POLCPIALLMINMEI'])
    cpi_yoy = (cpi_raw / cpi_raw.shift(12) - 1) * 100
    cpi_yoy.name = 'cpi_yoy'

    ir = (pd.read_csv(f'{DATA_DIR}/IRSTCI01PLM156N.csv',
                      parse_dates=['observation_date'])
          .set_index('observation_date')
          .sort_index()['IRSTCI01PLM156N'])
    ir.name = 'policy_rate'

    dates = pd.date_range('2010-01-01', '2025-03-01', freq='MS')
    df = pd.DataFrame({
        'cpi_yoy'    : cpi_yoy.reindex(dates),
        'policy_rate': ir.reindex(dates),
        'plneur'     : fx['plneur'].reindex(dates),
    }).dropna()
    return df


def make_differenced(df: pd.DataFrame) -> pd.DataFrame:
    """First differences for stationarity robustness."""
    d = df.diff().dropna()
    d.columns = [f'd_{c}' for c in df.columns]
    return d


# ---------------------------------------------------------------------------
# 2. STATIONARITY & LAG SELECTION
# ---------------------------------------------------------------------------

def adf_test(series, regression='ct'):
    r = adfuller(series.dropna(), maxlag=12, autolag='AIC', regression=regression)
    return {
        'ADF stat': round(r[0], 3),
        'p-value': round(r[1], 3),
        '5% CV': round(r[4]['5%'], 3),
        'H0 rejected (5%)': 'YES' if r[1] < 0.05 else 'NO',
    }


def stationarity_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        lvl = adf_test(df[col], regression='ct')
        rows.append({'Variable': col, 'Spec': 'Levels (const+trend)', **lvl})
        diff = adf_test(df[col].diff().dropna(), regression='c')
        rows.append({'Variable': col, 'Spec': 'First difference (const)', **diff})
    return pd.DataFrame(rows).set_index(['Variable', 'Spec'])


def select_var_lag(df: pd.DataFrame, maxlags: int = 4) -> pd.DataFrame:
    sel = VAR(df).select_order(maxlags)
    rows = []
    for p in range(1, maxlags + 1):
        rows.append({
            'p': p,
            'AIC': round(sel.ics['aic'][p], 3),
            'BIC': round(sel.ics['bic'][p], 3),
            'HQIC': round(sel.ics['hqic'][p], 3),
            'FPE': round(sel.ics['fpe'][p], 6),
        })
    table = pd.DataFrame(rows).set_index('p')
    table.loc[sel.selected_orders['bic'], 'BIC pick'] = '*'
    table.loc[sel.selected_orders['aic'], 'AIC pick'] = '*'
    return table.fillna('')


# ---------------------------------------------------------------------------
# 3. VAR MATRICES
# ---------------------------------------------------------------------------

def make_var_matrices(Y, p):
    T_full, n = Y.shape
    T = T_full - p
    Y_dep = Y[p:, :]
    Xblocks = [np.ones((T, 1))]
    for lag in range(1, p + 1):
        Xblocks.append(Y[p - lag: T_full - lag, :])
    return Y_dep, np.hstack(Xblocks)


# ---------------------------------------------------------------------------
# 4. MINNESOTA / NIW PRIOR
# ---------------------------------------------------------------------------

def build_minnesota_prior(n, p, k, sigma_ar, lam1=0.20, lam2=0.50,
                          lam3=100.0, delta=0.9):
    B0 = np.zeros((k, n))
    for i in range(n):
        B0[1 + i, i] = delta

    omega_d = np.zeros(k)
    omega_d[0] = lam3 ** 2
    for lag in range(1, p + 1):
        for j in range(n):
            idx   = 1 + (lag - 1) * n + j
            scale = lam1 / lag if lag == 1 else lam1 * lam2 / lag
            omega_d[idx] = scale ** 2 / max(sigma_ar[j] ** 2, 1e-6)

    Omega0     = np.diag(omega_d)
    Omega0_inv = np.diag(1.0 / omega_d)
    nu0        = n + 2
    S0         = np.diag(sigma_ar ** 2) * (nu0 - n - 1)
    return B0, Omega0, Omega0_inv, S0, nu0


def estimate_sigma_ar(Y):
    n = Y.shape[1]
    sigma_ar = np.zeros(n)
    for i in range(n):
        yi   = Y[:, i]
        Xar  = np.column_stack([np.ones(len(yi) - 1), yi[:-1]])
        bar_ = np.linalg.lstsq(Xar, yi[1:], rcond=None)[0]
        sigma_ar[i] = np.std(yi[1:] - Xar @ bar_, ddof=1)
    return sigma_ar


# ---------------------------------------------------------------------------
# 5. SAMPLERS
# ---------------------------------------------------------------------------

def draw_matrix_normal(M, U, V):
    k, n = M.shape
    Lu = np.linalg.cholesky(U + np.eye(k) * 1e-9)
    Lv = np.linalg.cholesky(V + np.eye(n) * 1e-9)
    return M + Lu @ np.random.randn(k, n) @ Lv.T


def draw_inv_wishart(Psi, nu):
    n  = Psi.shape[0]
    L  = np.linalg.cholesky(np.linalg.inv(Psi + np.eye(n) * 1e-10))
    A  = np.zeros((n, n))
    for i in range(n):
        A[i, i] = np.sqrt(np.random.chisquare(max(nu - i, 1)))
        for j in range(i):
            A[i, j] = np.random.randn()
    W = L @ A
    W = W @ W.T + np.eye(n) * 1e-10
    S = np.linalg.inv(W)
    return (S + S.T) / 2


# ---------------------------------------------------------------------------
# 6. GIBBS SAMPLER
# ---------------------------------------------------------------------------

def run_gibbs(Y_dep, X, B0, Omega0_inv, S0, nu0,
              n_draw=12000, n_burn=4000, verbose=True):
    """
    Full conditional Sigma | B, Y:
      Sn = S0 + E'E + (B-B0)'Omega0^-1(B-B0)
      nu_n = nu0 + T + k
    """
    T, n   = Y_dep.shape
    k      = X.shape[1]
    n_keep = n_draw - n_burn

    Omega_n_inv = Omega0_inv + X.T @ X
    Omega_n     = np.linalg.inv(Omega_n_inv)
    Omega_n     = (Omega_n + Omega_n.T) / 2
    B_n         = Omega_n @ (Omega0_inv @ B0 + X.T @ Y_dep)
    nu_post     = nu0 + T + k

    B_store     = np.zeros((n_keep, k, n))
    Sigma_store = np.zeros((n_keep, n, n))

    B_cur   = np.linalg.lstsq(X, Y_dep, rcond=None)[0]
    E0      = Y_dep - X @ B_cur
    Sig_cur = (E0.T @ E0) / T
    Sig_cur = (Sig_cur + Sig_cur.T) / 2

    if verbose:
        print(f"  Gibbs: {n_draw} draws, {n_burn} burn-in -> {n_keep} kept")

    for d in range(n_draw):
        B_cur   = draw_matrix_normal(B_n, Omega_n, Sig_cur)
        E       = Y_dep - X @ B_cur
        dB      = B_cur - B0
        S_post  = S0 + E.T @ E + dB.T @ Omega0_inv @ dB
        S_post  = (S_post + S_post.T) / 2
        Sig_cur = draw_inv_wishart(S_post, nu_post)
        if d >= n_burn:
            i = d - n_burn
            B_store[i]     = B_cur
            Sigma_store[i] = Sig_cur
        if verbose and (d + 1) % 4000 == 0:
            print(f"    iter {d + 1}/{n_draw}")

    if verbose:
        print("  Done.")
    return B_store, Sigma_store


def fit_bvar(df, p, lam1=0.20, n_draw=12000, n_burn=4000, verbose=True):
    n = df.shape[1]
    Y = df.values.copy()
    Y_dep, X = make_var_matrices(Y, p)
    k = X.shape[1]
    sigma_ar = estimate_sigma_ar(Y)
    B0, _, Omega0_inv, S0, nu0 = build_minnesota_prior(
        n, p, k, sigma_ar, lam1=lam1, lam2=0.50, lam3=100.0, delta=0.9)
    B_store, Sig_store = run_gibbs(
        Y_dep, X, B0, Omega0_inv, S0, nu0,
        n_draw=n_draw, n_burn=n_burn, verbose=verbose)
    return {
        'df': df, 'Y': Y, 'p': p, 'n': n, 'k': k,
        'B_store': B_store, 'Sig_store': Sig_store,
        'B_mean': B_store.mean(0), 'Sigma_mean': Sig_store.mean(0),
        'sigma_ar': sigma_ar, 'nu0': nu0,
    }


# ---------------------------------------------------------------------------
# 7. STRUCTURAL IRF
# ---------------------------------------------------------------------------

def permute_var_coefficients(B, order, n, p):
    B_new = np.zeros_like(B)
    for i in range(n):
        B_new[0, i] = B[0, order[i]]
        for lag in range(1, p + 1):
            for j in range(n):
                row_old = 1 + (lag - 1) * n + order[j]
                B_new[1 + (lag - 1) * n + j, i] = B[row_old, order[i]]
    return B_new


def map_irf_to_original(irf_perm, order):
    inv = np.argsort(order)
    n = len(order)
    h_len = irf_perm.shape[2]
    irf_orig = np.zeros_like(irf_perm)
    for h in range(h_len):
        for i in range(n):
            for j in range(n):
                irf_orig[i, j, h] = irf_perm[inv[i], inv[j], h]
    return irf_orig


def compute_irf_single(B, Sigma, n, p, h_max, order=None):
    if order is None:
        order = list(range(n))
    B_use = permute_var_coefficients(B, order, n, p)
    Sigma_use = Sigma[np.ix_(order, order)]

    A_comp = np.zeros((n * p, n * p))
    for lag in range(p):
        A_comp[:n, lag * n:(lag + 1) * n] = B_use[1 + lag * n:1 + (lag + 1) * n, :].T
    if p > 1:
        A_comp[n:, :n * (p - 1)] = np.eye(n * (p - 1))
    try:
        P = np.linalg.cholesky(Sigma_use + np.eye(n) * 1e-9)
    except np.linalg.LinAlgError:
        P = np.diag(np.sqrt(np.maximum(np.diag(Sigma_use), 1e-8)))
    irf = np.zeros((n, n, h_max + 1))
    A_h = np.eye(n * p)
    for h in range(h_max + 1):
        irf[:, :, h] = A_h[:n, :n] @ P
        A_h = A_h @ A_comp
    return map_irf_to_original(irf, order)


def posterior_irfs(B_store, Sigma_store, n, p, h_max=36, order=None):
    m = len(B_store)
    irf_all = np.zeros((m, n, n, h_max + 1))
    for i in range(m):
        irf_all[i] = compute_irf_single(
            B_store[i], Sigma_store[i], n, p, h_max, order=order)
    return irf_all


# ---------------------------------------------------------------------------
# 8. CONVERGENCE DIAGNOSTICS
# ---------------------------------------------------------------------------

def ess(chain):
    """Geyer initial positive-sequence ESS estimator."""
    N = len(chain)
    if N < 2:
        return float(N)
    v = np.var(chain, ddof=0)
    if v < 1e-14:
        return float(N)
    rho_sum = 0.0
    for k in range(1, N):
        rho_k = np.corrcoef(chain[:-k], chain[k:])[0, 1]
        if rho_k < 0:
            break
        rho_sum += rho_k
    return min(float(N), N / max(1.0, 1 + 2 * rho_sum))


def geweke(chain, fa=0.10, fb=0.50):
    N  = len(chain)
    a  = chain[:int(fa * N)]
    b  = chain[int((1 - fb) * N):]
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    return (a.mean() - b.mean()) / max(se, 1e-12)


def print_diagnostics(B_store, Sigma_store, n, k):
    print("\n-- MCMC Convergence Diagnostics --------------------------------")
    print(f"  {'Parameter':<30}  {'ESS':>8}  {'Geweke Z':>10}  {'OK?':>4}")
    print("  " + "-" * 58)
    for i in range(n):
        for j in [0, 1 + i]:
            ch    = B_store[:, j, i]
            label = f"B[eq={VARLAB[i][:9]}, row={j}]"
            print(f"  {label:<30}  {ess(ch):>8.0f}  {geweke(ch):>10.3f}  "
                  f"{ok_mark(geweke(ch)):>4}")
    for i in range(n):
        ch    = Sigma_store[:, i, i]
        label = f"Sigma[{i},{i}] ({VARLAB[i][:9]})"
        print(f"  {label:<30}  {ess(ch):>8.0f}  {geweke(ch):>10.3f}  "
              f"{ok_mark(geweke(ch)):>4}")
    print()


def export_posterior_tables(result, adf_df, lag_df, tag='baseline'):
    B_store = result['B_store']
    p, n = result['p'], result['n']
    row_labels = ['const'] + [
        f'{v}_l{l}' for l in range(1, p + 1) for v in ['cpi', 'rate', 'fx']]

    rows = []
    for j, rl in enumerate(row_labels):
        for i in range(n):
            ch = B_store[:, j, i]
            rows.append({
                'model': tag, 'row': rl, 'equation': VARNAME[i],
                'mean': ch.mean(), 'std': ch.std(),
                'p05': np.percentile(ch, 5), 'p95': np.percentile(ch, 95),
                'ess': ess(ch), 'geweke_z': geweke(ch),
            })
    pd.DataFrame(rows).to_csv(f'{OUT}/posterior_{tag}.csv', index=False)
    adf_df.to_csv(f'{OUT}/stationarity_adf.csv')
    lag_df.to_csv(f'{OUT}/lag_selection.csv')
    print(f"  [saved] stationarity_adf.csv, lag_selection.csv, posterior_{tag}.csv")


# ---------------------------------------------------------------------------
# 9. FORECAST
# ---------------------------------------------------------------------------

def posterior_forecast(Y, B_store, Sigma_store, p, h=12):
    m, _, n = B_store.shape
    fc = np.zeros((m, h, n))
    for d in range(m):
        B   = B_store[d]
        Sig = Sigma_store[d]
        try:
            P = np.linalg.cholesky(Sig + np.eye(n) * 1e-10)
        except np.linalg.LinAlgError:
            P = np.diag(np.sqrt(np.maximum(np.diag(Sig), 1e-8)))
        hist = list(Y[-p:][::-1])
        for s in range(h):
            x     = np.concatenate([[1.0], *[hist[l] for l in range(p)]])
            y_new = B.T @ x + P @ np.random.randn(n)
            fc[d, s, :] = y_new
            hist.insert(0, y_new)
    return fc


# ---------------------------------------------------------------------------
# 10. PLOTS
# ---------------------------------------------------------------------------

def _shade(ax):
    for s, e, c in [('2020-01', '2020-12', '#d6eaf8'),
                    ('2021-10', '2023-09', '#fef9e7')]:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.28, color=c, zorder=0)


def plot_raw_data(df):
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for ax, col, lbl, c in zip(axes, df.columns, VARLAB, COLORS):
        ax.plot(df.index, df[col], color=c, lw=1.6)
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.35)
        _shade(ax)
    axes[0].set_title(
        'Polish Macro Data 2010-2025  [OECD/FRED & ECB]\n'
        'CPI YoY  |  Short-term Interest Rate  |  PLN/EUR',
        fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/01_data.png', bbox_inches='tight')
    plt.close(fig)
    print("  [saved] 01_data.png")


def plot_irfs(irf_all, shock_idx, fname, order_label='Cholesky [CPI, Rate, FX]'):
    n, h_max = irf_all.shape[1], irf_all.shape[3] - 1
    hz   = np.arange(h_max + 1)
    q05  = np.quantile(irf_all[:, :, shock_idx, :], 0.05, axis=0)
    q16  = np.quantile(irf_all[:, :, shock_idx, :], 0.16, axis=0)
    q50  = np.median(irf_all[:, :, shock_idx, :], axis=0)
    q84  = np.quantile(irf_all[:, :, shock_idx, :], 0.84, axis=0)
    q95  = np.quantile(irf_all[:, :, shock_idx, :], 0.95, axis=0)
    fig, axes = plt.subplots(1, n, figsize=(13, 4))
    for i, ax in enumerate(axes):
        ax.fill_between(hz, q05[i], q95[i], alpha=0.18, color=COLORS[i])
        ax.fill_between(hz, q16[i], q84[i], alpha=0.40, color=COLORS[i])
        ax.plot(hz, q50[i], color=COLORS[i], lw=2.2, label='Median')
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(f'Response of {VARLAB[i]}', fontweight='bold')
        ax.set_xlabel('Months')
        ax.grid(True, alpha=0.30)
        if i == 0:
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(color=COLORS[i], alpha=0.40, label='68% CI'),
                Patch(color=COLORS[i], alpha=0.18, label='90% CI'),
                plt.Line2D([0], [0], color=COLORS[i], lw=2, label='Median'),
            ], fontsize=7)
    fig.suptitle(
        f'Structural IRF — 1 S.D. Shock to {VARLAB[shock_idx]}  ({order_label})',
        fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/{fname}.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] {fname}.png")


def plot_irf_comparison(irf_all, cls_irf, shock_idx, h_max):
    n  = irf_all.shape[1]
    hz = np.arange(h_max + 1)
    q10 = np.quantile(irf_all[:, :, shock_idx, :h_max + 1], 0.10, axis=0)
    q50 = np.median(irf_all[:, :, shock_idx, :h_max + 1], axis=0)
    q90 = np.quantile(irf_all[:, :, shock_idx, :h_max + 1], 0.90, axis=0)
    fig, axes = plt.subplots(1, n, figsize=(13, 4))
    for i, ax in enumerate(axes):
        ax.fill_between(hz, q10[i], q90[i], alpha=0.25, color=COLORS[i], label='BVAR 80% CI')
        ax.plot(hz, q50[i], color=COLORS[i], lw=2.2, label='BVAR median')
        ax.plot(hz, cls_irf[:h_max + 1, i, shock_idx], 'k--', lw=1.6, label='Classical VAR')
        ax.axhline(0, color='k', lw=0.7, ls=':')
        ax.set_title(f'Response of {VARLAB[i]}', fontweight='bold')
        ax.set_xlabel('Months')
        if i == 0:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.30)
    fig.suptitle(f'BVAR vs Classical VAR — IRF to {VARLAB[shock_idx]} Shock',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/04_irf_comparison_shock{shock_idx}.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] 04_irf_comparison_shock{shock_idx}.png")


def plot_identification_sensitivity(irf_by_order, shock_idx=1, h_max=24):
    hz = np.arange(h_max + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    colors_ord = ['#2E86AB', '#A23B72', '#F18F01']
    for ax, (name, irf_all), c in zip(axes, irf_by_order.items(), colors_ord):
        q50 = np.median(irf_all[:, 0, shock_idx, :h_max + 1], axis=0)
        q10 = np.quantile(irf_all[:, 0, shock_idx, :h_max + 1], 0.10, axis=0)
        q90 = np.quantile(irf_all[:, 0, shock_idx, :h_max + 1], 0.90, axis=0)
        ax.fill_between(hz, q10, q90, alpha=0.20, color=c)
        ax.plot(hz, q50, color=c, lw=2.2, label='Median')
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(name.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('Months')
        ax.grid(True, alpha=0.30)
    axes[0].set_ylabel(f'Response of {VARLAB[0]}')
    fig.suptitle(
        f'Identification Sensitivity — CPI Response to {VARLAB[shock_idx]} Shock',
        fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/10_id_sensitivity_policy_shock.png', bbox_inches='tight')
    plt.close(fig)
    print("  [saved] 10_id_sensitivity_policy_shock.png")


def plot_prior_sensitivity(irf_by_lambda, shock_idx=1, h_max=24):
    hz = np.arange(h_max + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_l = ['#95a5a6', '#2E86AB', '#1a5276']
    for lam, irf_all, c in zip(PRIOR_LAMBDAS, irf_by_lambda, colors_l):
        q50 = np.median(irf_all[:, 0, shock_idx, :h_max + 1], axis=0)
        ax.plot(hz, q50, color=c, lw=2.0, label=f'lambda1 = {lam:.2f}')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Months')
    ax.set_ylabel(f'Response of {VARLAB[0]}')
    ax.set_title(
        f'Prior Sensitivity — CPI Response to {VARLAB[shock_idx]} Shock (median IRF)',
        fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.30)
    fig.tight_layout()
    fig.savefig(f'{OUT}/11_prior_sensitivity.png', bbox_inches='tight')
    plt.close(fig)
    print("  [saved] 11_prior_sensitivity.png")


def plot_robustness_comparison(irf_levels, irf_diff, shock_idx=1, h_max=24):
    hz = np.arange(h_max + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for irf_all, lbl, c in [(irf_levels, 'Levels (baseline)', '#2E86AB'),
                            (irf_diff, 'First differences', '#F18F01')]:
        q50 = np.median(irf_all[:, 0, shock_idx, :h_max + 1], axis=0)
        q10 = np.quantile(irf_all[:, 0, shock_idx, :h_max + 1], 0.10, axis=0)
        q90 = np.quantile(irf_all[:, 0, shock_idx, :h_max + 1], 0.90, axis=0)
        ax.fill_between(hz, q10, q90, alpha=0.15, color=c)
        ax.plot(hz, q50, color=c, lw=2.2, label=lbl)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Months')
    ax.set_ylabel(f'Response of inflation / d(CPI YoY)')
    ax.set_title(
        f'Stationarity Robustness — CPI Response to {VARLAB[shock_idx]} Shock',
        fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.30)
    fig.tight_layout()
    fig.savefig(f'{OUT}/12_stationarity_robustness.png', bbox_inches='tight')
    plt.close(fig)
    print("  [saved] 12_stationarity_robustness.png")


def plot_convergence(B_store, Sigma_store):
    n_keep = len(B_store)
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    for i in range(3):
        chain = B_store[:, 1 + i, i]
        x     = np.arange(n_keep)
        ax    = axes[i, 0]
        ax.plot(x, chain, lw=0.35, color=COLORS[i], alpha=0.75)
        ax.axhline(chain.mean(), color='k', lw=1.2, ls='--',
                   label=f'Mean={chain.mean():.4f}')
        ax.set_title(f'Trace — B[own lag 1, {VARLAB[i]}]', fontweight='bold')
        ax.set_xlabel('Post-burn-in iteration')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.30)
        ax2   = axes[i, 1]
        lags  = 60
        acf_  = [np.corrcoef(chain[:-k], chain[k:])[0, 1] for k in range(1, lags + 1)]
        ax2.bar(range(1, lags + 1), acf_, color=COLORS[i], alpha=0.7)
        bnd = 1.96 / np.sqrt(n_keep)
        ax2.axhline(bnd, color='red', lw=0.9, ls='--', label='±1.96/sqrt(N)')
        ax2.axhline(-bnd, color='red', lw=0.9, ls='--')
        ax2.axhline(0, color='k', lw=0.5)
        ax2.set_title(f'ACF — B[own lag 1, {VARLAB[i]}]', fontweight='bold')
        ax2.set_xlabel('Lag')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.30)
    fig.suptitle('MCMC Diagnostics: Trace Plots & Autocorrelation Functions',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/05_convergence.png', bbox_inches='tight')
    plt.close(fig)
    print("  [saved] 05_convergence.png")


def plot_burnin_check(B_store):
    n_keep = len(B_store)
    x      = np.arange(1, n_keep + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for i, ax in enumerate(axes):
        chain = B_store[:, 1 + i, i]
        cm    = np.cumsum(chain) / x
        ax.plot(x, cm, color=COLORS[i], lw=1.5)
        ax.axhline(cm[-1], color='k', lw=0.9, ls='--',
                   label=f'Final={cm[-1]:.4f}')
        ax.set_title(f'Cum. Mean  B[own lag 1, {VARLAB[i]}]', fontweight='bold')
        ax.set_xlabel('Post-burn-in draw')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.35)
    fig.suptitle('Burn-in Check: Cumulative Posterior Means', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/06_burnin.png', bbox_inches='tight')
    plt.close(fig)
    print("  [saved] 06_burnin.png")


def plot_posteriors(B_store, Sigma_store):
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)
    for i in range(3):
        for row, (store, tag) in enumerate([
            (B_store[:, 1 + i, i], 'own lag-1 coeff'),
            (Sigma_store[:, i, i], 'residual variance Sigma_ii'),
            (B_store[:, 0, i], 'constant'),
        ]):
            ax = fig.add_subplot(gs[row, i])
            ax.hist(store, bins=60, color=COLORS[i], alpha=0.75, density=True)
            ax.axvline(store.mean(), color='k', lw=1.6,
                       label=f'Mean={store.mean():.3f}')
            lo, hi = np.percentile(store, 5), np.percentile(store, 95)
            ax.axvline(lo, color='red', lw=1.1, ls='--')
            ax.axvline(hi, color='red', lw=1.1, ls='--',
                       label=f'90%CI [{lo:.3f},{hi:.3f}]')
            ax.set_title(f'{tag}\n{VARLAB[i]}', fontweight='bold')
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.30)
    fig.suptitle('Marginal Posterior Distributions — Selected Parameters',
                 fontweight='bold', y=1.01)
    fig.savefig(f'{OUT}/07_posteriors.png', bbox_inches='tight')
    plt.close(fig)
    print("  [saved] 07_posteriors.png")


def plot_forecast(df, fc_draws, p, h=12):
    q05 = np.quantile(fc_draws, 0.05, axis=0)
    q25 = np.quantile(fc_draws, 0.25, axis=0)
    q50 = np.median(fc_draws, axis=0)
    q75 = np.quantile(fc_draws, 0.75, axis=0)
    q95 = np.quantile(fc_draws, 0.95, axis=0)
    last   = df.index[-1]
    fc_idx = pd.date_range(last + pd.DateOffset(months=1), periods=h, freq='MS')
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=False)
    for i, (ax, col, lbl, c) in enumerate(zip(axes, df.columns, VARLAB, COLORS)):
        hist = df[col].iloc[-48:]
        ax.plot(hist.index, hist.values, color=c, lw=1.6, label='Historical')
        ax.fill_between(fc_idx, q05[:, i], q95[:, i], alpha=0.15, color=c, label='90% CI')
        ax.fill_between(fc_idx, q25[:, i], q75[:, i], alpha=0.35, color=c, label='50% CI')
        ax.plot(fc_idx, q50[:, i], color=c, lw=2.2, ls='--', label='Median')
        ax.axvline(last, color='grey', lw=0.9, ls=':')
        ax.set_ylabel(lbl)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.35)
    axes[0].set_title(f'BVAR({p}) Posterior Predictive Forecast — 12 months ahead',
                      fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/08_forecast.png', bbox_inches='tight')
    plt.close(fig)
    print("  [saved] 08_forecast.png")


def plot_historical_fit(df, B_mean, p):
    Y = df.values
    Y_dep, X = make_var_matrices(Y, p)
    fitted    = X @ B_mean
    dates_fit = df.index[p:]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for i, (ax, lbl, c) in enumerate(zip(axes, VARLAB, COLORS)):
        ax.plot(df.index, df.iloc[:, i], color=c, lw=1.4, alpha=0.8, label='Actual')
        ax.plot(dates_fit, fitted[:, i], 'k--', lw=1.0, alpha=0.85,
                label='Fitted (posterior mean B)')
        ax.set_ylabel(lbl)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.30)
    axes[0].set_title(f'In-sample Fit: BVAR({p}) Posterior Mean', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/09_fitted.png', bbox_inches='tight')
    plt.close(fig)
    print("  [saved] 09_fitted.png")


# ---------------------------------------------------------------------------
# 11. MAIN
# ---------------------------------------------------------------------------

def main():
    configure_console()
    bar = "=" * 72
    print(bar)
    print("  BVAR — Polish Monetary Policy, Inflation & Exchange Rate")
    print("  Data: OECD/FRED (CPI, rate) + ECB (PLN/EUR)  |  2010-2025")
    print(bar)

    # 1. Data
    print("\n[1] Loading real data ...")
    df = build_data()
    print(f"    {len(df)} obs:  {df.index[0]:%Y-%m} -> {df.index[-1]:%Y-%m}")
    print(df.describe().round(3).to_string())
    plot_raw_data(df)

    # 2. Stationarity
    print("\n[2] Stationarity (ADF: levels + first differences)")
    adf_df = stationarity_report(df)
    print(adf_df.to_string())
    n_nonstat_levels = (adf_df.xs('Levels (const+trend)', level='Spec')['H0 rejected (5%)'] == 'NO').sum()
    if n_nonstat_levels == len(df.columns):
        print("    NOTE: All series fail to reject unit root in levels.")
        print("          First-difference specs and robustness check follow below.")

    # 3. Lag selection
    print("\n[3] Lag order selection (AIC / BIC / HQIC)")
    lag_df = select_var_lag(df, maxlags=4)
    print(lag_df.to_string())
    p = int(lag_df.index[lag_df['BIC pick'] == '*'][0])
    print(f"    Selected lag order: p = {p} (min BIC)")

    # 4. Baseline BVAR
    print(f"\n[4] Baseline BVAR({p}) — Minnesota prior (lambda1=0.20)")
    result = fit_bvar(df, p, lam1=0.20, n_draw=12000, n_burn=4000)
    B_store, Sig_store = result['B_store'], result['Sig_store']
    B_mean, Sigma_mean = result['B_mean'], result['Sigma_mean']
    n, k = result['n'], result['k']
    export_posterior_tables(result, adf_df, lag_df, tag='baseline')

    print(f"    sigma_ar: {dict(zip(VARLAB, result['sigma_ar'].round(4)))}")
    print(f"    Prior: nu0={result['nu0']}, diag(S0)={np.diag(build_minnesota_prior(n, p, k, result['sigma_ar'])[3]).round(5)}")

    row_labels = ['const'] + [
        f'{v}_l{l}' for l in range(1, p + 1) for v in ['cpi', 'rate', 'fx']]
    print("\n    Posterior B — mean [std] (5%,95%)")
    header = f"  {'':15}" + "".join(f"  {lb:>24}" for lb in VARLAB)
    print(header)
    for j, rl in enumerate(row_labels):
        vals = ""
        for i in range(n):
            lo = np.percentile(B_store[:, j, i], 5)
            hi = np.percentile(B_store[:, j, i], 95)
            vals += f"  {B_mean[j,i]:6.4f}[{B_store[:,j,i].std():.4f}]({lo:.3f},{hi:.3f})"
        print(f"  {rl:<15}{vals}")
    print(f"\n    Posterior mean Sigma:\n{np.round(Sigma_mean, 6)}")

    # 5. Diagnostics
    print("\n[5] Convergence diagnostics")
    print_diagnostics(B_store, Sig_store, n, k)
    plot_convergence(B_store, Sig_store)
    plot_burnin_check(B_store)
    plot_posteriors(B_store, Sig_store)

    # 6. IRFs (baseline ordering)
    print("[6] Computing IRFs (h_max=36) ...")
    irf_all = posterior_irfs(B_store, Sig_store, n, p, h_max=36, order=BASE_ORDER)
    for shock in range(n):
        plot_irfs(irf_all, shock, f'02_irf_shock{shock}_{VARNAME[shock]}')

    # 7. Identification sensitivity
    print("\n[7] Identification sensitivity (alternative Cholesky orderings)")
    irf_by_order = {}
    for name, order in ID_ORDERS.items():
        irf_by_order[name] = posterior_irfs(
            B_store, Sig_store, n, p, h_max=24, order=order)
    plot_identification_sensitivity(irf_by_order, shock_idx=1, h_max=24)

    # 8. Prior sensitivity
    print("\n[8] Prior sensitivity (lambda1 in {0.10, 0.20, 0.50})")
    irf_by_lambda = []
    for lam in PRIOR_LAMBDAS:
        print(f"    lambda1 = {lam:.2f} ...")
        sens = fit_bvar(df, p, lam1=lam, n_draw=6000, n_burn=2000, verbose=False)
        irf_by_lambda.append(posterior_irfs(
            sens['B_store'], sens['Sig_store'], n, p, h_max=24, order=BASE_ORDER))
    plot_prior_sensitivity(irf_by_lambda, shock_idx=1, h_max=24)

    # 9. Stationarity robustness (first differences)
    print("\n[9] Stationarity robustness — BVAR on first differences")
    df_diff = make_differenced(df)
    diff_result = fit_bvar(df_diff, p, lam1=0.20, n_draw=6000, n_burn=2000, verbose=False)
    irf_diff = posterior_irfs(
        diff_result['B_store'], diff_result['Sig_store'], n, p, h_max=24, order=BASE_ORDER)
    plot_robustness_comparison(irf_all, irf_diff, shock_idx=1, h_max=24)
    export_posterior_tables(diff_result, adf_df, lag_df, tag='first_diff')

    # 10. Classical VAR benchmark
    print(f"\n[10] Classical VAR({p}) benchmark")
    cls_result = VAR(df).fit(p, trend='c')
    cls_irf    = cls_result.irf(36).orth_irfs
    print(cls_result.summary())
    for shock in range(n):
        plot_irf_comparison(irf_all, cls_irf, shock, h_max=24)

    # 11. Forecast
    print("[11] Posterior predictive forecast ...")
    fc = posterior_forecast(result['Y'], B_store, Sig_store, p, h=12)
    plot_forecast(df, fc, p, h=12)

    # 12. Fit
    plot_historical_fit(df, B_mean, p)

    # Summary
    print()
    print(bar)
    print("  RESULTS SUMMARY")
    print(bar)
    print(f"  {'Equation':<22}  {'Mean':>8}  {'Std':>7}  {'5th%':>7}  "
          f"{'95th%':>7}  {'ESS':>7}  Geweke")
    print("  " + "-" * 70)
    for i, lb in enumerate(VARLAB):
        ch = B_store[:, 1 + i, i]
        g  = geweke(ch)
        print(f"  {lb:<22}  {ch.mean():>8.4f}  {ch.std():>7.4f}  "
              f"{np.percentile(ch,5):>7.4f}  {np.percentile(ch,95):>7.4f}  "
              f"{ess(ch):>7.0f}  {g:>6.3f} {ok_mark(g)}")
    print(f"\n  Outputs -> '{OUT}/'")
    print(bar)


if __name__ == '__main__':
    main()
