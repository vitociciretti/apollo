# Transformer joint generator - scorecard

Decoder-only transformer, 4 layers, d_model 64, 4 heads, context 256. Joint 8-ticker Student-t heads, cross-sectional dependence via bootstrap of in-sample 8-dim standardized-residual rows. Trained on 3558 common days (2012-05-21 .. 2026-07-16), best val NLL 7.4704 (sum over 8 tickers). Sample length 4032 days.

## Per-ticker: real vs transformer sample

| ticker | annvol real | annvol TF | kurt real | kurt TF | ACF|r|(1) real | ACF|r|(1) TF | ACF|r|(21) real | ACF|r|(21) TF | GPH d real | GPH d TF |
|---|---|---|---|---|---|---|---|---|---|---|
| NVDA | 44.7% | 37.7% | 10.7 | 5.2 | 0.190 | 0.045 | 0.092 | 0.041 | 0.371 | 0.271 |
| AAPL | 28.3% | 20.1% | 9.6 | 5.0 | 0.206 | 0.111 | 0.073 | 0.038 | 0.395 | 0.334 |
| MSFT | 26.3% | 19.8% | 11.3 | 4.4 | 0.232 | 0.044 | 0.068 | 0.025 | 0.354 | 0.255 |
| GOOGL | 27.6% | 25.2% | 9.6 | 7.3 | 0.138 | 0.009 | 0.050 | 0.060 | 0.283 | 0.310 |
| AMZN | 32.1% | 25.5% | 8.9 | 5.1 | 0.174 | 0.077 | 0.063 | 0.067 | 0.354 | 0.349 |
| META | 39.8% | 29.2% | 23.8 | 7.2 | 0.110 | 0.046 | 0.085 | 0.044 | 0.305 | 0.355 |
| NFLX | 45.8% | 27.9% | 32.4 | 6.8 | 0.129 | 0.068 | 0.034 | 0.044 | 0.223 | 0.324 |
| TSLA | 56.8% | 44.0% | 7.5 | 5.9 | 0.126 | 0.054 | 0.060 | 0.056 | 0.461 | 0.304 |

## Cross-sectional dependence: corr(NVDA, AAPL)

| measure | real | transformer sample |
|---|---|---|
| full-sample corr | 0.484 | 0.266 |
| crisis corr (top-decile joint vol days) | 0.640 | 0.213 |

## NVDA baseline: univariate LSTM (from docs/technical_documentation.tex)

Real-NVDA column there uses the full 2010-2026 sample; the transformer's real column above uses the 8-ticker common sample (post META IPO), so real values differ slightly.

| metric | NVDA real (docs) | LSTM sample | transformer sample |
|---|---|---|---|
| annualized vol | 45.2% | 53.4% | 37.7% |
| kurtosis | 9.8 | 10.1 | 5.2 |
| ACF\|r\|(1) | 0.171 | 0.131 | 0.045 |
| ACF\|r\|(21) | 0.092 | 0.089 | 0.041 |
| GPH d | 0.397 | 0.516 | 0.271 |
