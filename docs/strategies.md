# Strategy Reference

This system keeps a single universe of ~40 liquid coins (`SYMBOLS` in `.env`) while letting the ***active set*** rotate via lanes/signals that reflect the math below. The Rust core + Nemo agree on the formulas because `strategy.rs` loads `data/strategy.toml` and regenerates `data/nemo_prompt.txt`, so the same weights/filters drive both deterministic scoring and the AI prompt.

## Asset Definitions
Let \(P_{i,t}\) be asset \(i\)’s latest price and \(r_{i,t} = \ln(P_{i,t} / P_{i,t-1})\) the log return. Define the live indicators already computed in `features.rs`: trend \(T\), volatility \(V\), RSI \(R\), buy ratio \(B\), imbalance \(I\), MACD histogram \(M\), ATR, z-score \(Z\), quant signals \(Q_m,Q_r,Q_v\), CVaR allocation \(CWeight\), etc. These feed the lanes/signals below.

## Base Score
Normalized score \(N(x,a,b) = \frac{\text{clamp}((x-a)/(b-a),0,1)\times 2-1}{ }\) maps each indicator to \([-1,+1]\). The 7D score (matching `scoring.rs`) is:
\[
S = 0.30 N(T,-5,5) 
  +0.22 \frac{N(M,-1,1)+N(R,20,80)}{2}
  +0.10 (-N(ATR,0,P\times0.05))
  +0.20 N(V,0.5,2)
  +0.10 (-N(Z,-2.5,2.5))
  +0.05 \frac{N(I,-1,1)+N(B,0.3,0.7)}{2}
\]
The AI prompt uses this to explain why a coin scored high/low.

## Lane & Filter Rotation
Every \(K\) ticks (suggest 12), recompute lanes:

1. **Lane 1 (Momentum):** require \(V\geq1.1\), \(R<80\), \(T\geq1\). This keeps coins with clean trend + liquidity.
2. **Lane 2 (Compression):** require \(BBwidth<0.06, Q_v<0.04, |P-E_{50}|/E_{50}<0.03, ATR/P<0.03, I\geq-0.15\). Captures low-volatility setups ready to expand.
3. **Lane 3 (Trend continuation):** require \(P>E_{50}, Q_r>0, 40<R<70, Risk<0.7, T\geq0, \mu\geq-0.005\).

Global filters (applied before lanes):
* Liquidity \(\geq0.15\), spread \(\leq1.0\%\), vol ratio \(\geq0.1\).
* Coins failing their lane are dropped until they meet the thresholds again, so the active set rotates naturally.

## Strategy Menu
Choose one or compose an ensemble; each uses the same base universe + filter rules.

1. **Volatility Parity (risk parity)**  
   - Estimate rolling volatility \(\sigma_{i,t}\) using \(L\) past returns, e.g., \(L=21\).  
   - Raw weight \(\tilde w_{i,t} = \frac{1/\sigma_{i,t}}{\sum_{j} 1/\sigma_{j,t}}\).  
   - Cap \(w_{i,t} = \min(\tilde w_{i,t}, w_{\max})\), renormalize to sum to 1 (and pad cash if all coins capped).  
   - Use \(w_{\max}=0.12\) and rebalance every \(K=12\) ticks; skip rebalance unless \(|w_{i,t} - w_{i,t-1}| > \delta\) with \(\delta=0.015\).

2. **Momentum Ranking**  
   - Momentum \(m_{i,t} = \sum_{k=1}^{H} r_{i,t-k}\) with horizon \(H=21\).  
   - Let \(S_t\) be the top \(K=10\) assets by \(m_{i,t}\).  
   - Option A: equal-weight \(w_i = 1/K\) for \(i\in S_t\).  
   - Option B (momentum-weighted): \(w_i = \frac{\max(m_{i,t},0)}{\sum_{j\in S_t} \max(m_{j,t},0)}\).  
   - Combine with trend filter \(T_{i,t} = \mathbb{1}[MA_{fast}(P)>MA_{slow}(P)]\) and set \(w_i=0\) if \(T_{i,t}=0\). If no coin passes \(T_{i,t}\), route weight to cash.

3. **Risk-adjusted momentum**  
   - Compute \(q_{i,t} = m_{i,t} / (\sigma_{i,t} + \epsilon)\) with \(\epsilon=1e{-4}\).  
   - Rank by \(q_{i,t}\) and allocate to the top \(K\) using either equal weight or softmax (below).

4. **Softmax ensemble (default AI-friendly)**  
   - Score \(s_{i,t} = \alpha m_{i,t} + \beta T_{i,t} + \gamma / (\sigma_{i,t} + \epsilon)\). Choose \(\alpha=1.0,\beta=0.5,\gamma=0.2,\epsilon=1e{-4}\).  
   - Convert to weights via softmax with temperature \(\tau=1.0\):  
     \(w_{i,t} = \frac{\exp(s_{i,t}/\tau)}{\sum_j \exp(s_{j,t}/\tau)}\).  
   - Apply \(w_{\max}=0.15\) caps and renormalize; require \(|w_{i,t} - w_{i,t-1}|>\delta\) with \(\delta=0.01\) before trading.

5. **Minimum variance portfolio**  
   - Estimate covariance matrix \(\Sigma_t\) using the last 42 returns.  
   - Solve \(\min_{w} w^\top\Sigma_t w\) subject to \(\sum_i w_i=1, w_i\geq0, w_i\leq w_{\max}\).  
   - Approximate via CVaR optimizer or offload to CVaR sidecar (`cvar_optimizer` already runs); treat its `CWeight` output as a mask on the risk-adjusted weights above.

6. **Transaction-cost aware utility**  
   - Let target weights be \(w^\text{target}_i\).  
   - Turnover penalty \(TC_t = \sum_i c_i |w_{i,t} - w_{i,t-1}|\) with \(c_i\) derived from each asset’s fee/slippage (can reuse `config.rs`).  
   - Optimal weights maximize \(\mu_t^\top w - \lambda w^\top \Sigma_t w - \kappa TC_t\) under the same constraints; reduce \(\lambda\) or \(\kappa\) to tune aggressiveness.

## Implementation sketch

1. Continue to store lane/filter parameters in `strategy.toml` (see current sections).  
2. In `strategy.rs`, add new sections for general strategy settings:
   ```toml
   [strategy]
   type = "softmax"
   H = 21
   K = 10
   w_max = 0.15
   delta = 0.01
   tau = 1.0
   alpha = 1.0
   beta = 0.5
   gamma = 0.2
   ```
3. `trading_loop.rs` (around `strategy` use) should:
   - Recompute \(m_{i,t}\) and \(\sigma_{i,t}\) for all symbols fetched by `features.rs`.  
   - Create mask of coins passing lane/global filters (`coin_filter.rs`).  
   - Based on `strategy.type`, compute weights per formulas above.  
   - Before sending orders, enforce cap and trade only if \(|w_{i,t} - w_{i,t-1}|>\delta\).  
   - Provide weights + reasons to Nemo via `ai_bridge` prompt (maybe extend `nemo_prompt.txt` template with the selected formulas and parameters).

4. Document the spec in `docs/strategies.md` (this file) and optionally in README so AI/Dev teams know which formulas power the decisions.

## Next steps
1. Choose the configuration you want (softmax ensemble is the most flexible).  
2. I can generate the updated `strategy.toml` snippet + Nemotron prompt text with those parameters.  
3. If desired, I can write the supporting Rust helper code (e.g., new functions in `strategy.rs` and `trading_loop.rs`) to compute \(m\), \(\sigma\), softmax weights, lane filtering, and the turnover check.  
4. Let me know if you want a helper to export the same weights to `data/nemo_state.json` so the WS service can be aware too.
