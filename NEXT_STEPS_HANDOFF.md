# Future Research Handoff

## Four Points for Tonight

- The main result is not "text cannot matter"; it is that by `T-7`, Kalshi appears to have already incorporated the information tested in this project.
- The next research frontier is earlier in the cycle: `T-30`, `T-20`, and `T-14`, where macro data, Fed speeches, and rate futures may still be moving market-implied probabilities.
- A strong extension is cross-market disagreement: compare Kalshi probabilities with Fed funds futures, SOFR futures, economist consensus, macro surprises, and speaker-specific Fed communication signals.
- The project can evolve from a pure alpha test into a practical macro risk framework for hedging and sizing exposure in rates, banks, real estate, credit, FX, and duration-sensitive assets.

## Core Interpretation

This project finds strong evidence that Kalshi FOMC rate-decision markets are highly efficient by the final week before an announcement. The key empirical result is that `T-7` and `T-1` implied expectations are almost identical, with `corr(T-7, T-1) = 0.9993`. In the tested sample, simple NLP signals, fine-tuned transformer sentiment, and realized surprise residuals did not generate statistically reliable equity or rates alpha.

That result should be interpreted narrowly. It does not prove that prediction markets are always perfectly efficient, and it does not rule out every possible trading or hedging application. It says that late-window FOMC expectations are difficult to beat using the text and surprise methods tested here.

## Earlier-Horizon Efficiency

The most natural next study is to move earlier in the meeting cycle. Instead of starting at `T-7`, test whether Kalshi prices are calibrated at `T-30`, `T-20`, `T-14`, and `T-10`.

The academic question:

```text
How quickly do FOMC prediction markets incorporate public macroeconomic information?
```

The empirical target could be either final settlement value or future mark-to-market movement:

```text
contract_return_to_settlement = payout - executable_price
contract_return_to_T-7 = price_T-7 - price_T-20
```

The key is to evaluate any signal after bid-ask spreads, fees, liquidity, contract rules, and realistic execution assumptions. A signal that works before transaction costs is not enough.

## Cross-Market Disagreement

A stronger extension is not simply "Kalshi versus NLP." It is Kalshi versus the broader macro consensus.

Future models could compare Kalshi-implied probabilities with:

- Fed funds futures
- SOFR futures
- OIS or front-end Treasury rates
- Economist survey consensus
- CPI, payrolls, PCE, and inflation surprise measures
- Federal Reserve speech and statement tone
- Dot-plot dispersion and dissent history
- Volatility and liquidity measures

The main signal would be the gap between a model-implied probability and the Kalshi-implied probability. Academically, this tests whether institutional rates markets and retail-accessible prediction markets process information at the same speed.

## Speaker-Conditional Fed Communication

Another promising layer is speaker identity. The current project mostly treats Fed communication as an aggregate text signal. Future work could ask whether prediction markets correctly weight different Fed speakers.

Prior research on FOMC voting-right rotation shows that Reserve Bank presidents behave differently when they are voting versus non-voting members, and that Treasury markets react differently to their speeches. That suggests communication should be conditioned on institutional role, not treated as one homogeneous Fed text stream.

Future variables could include:

- Chair versus governor versus Reserve Bank president
- Voting versus non-voting Reserve Bank president
- Historical hawk/dove speaker profile
- Regional unemployment, inflation, and growth conditions
- Pre-FOMC versus post-FOMC timing
- Pre-Beige Book versus post-Beige Book timing
- High-disagreement periods, such as dissents or dot-plot dispersion
- Hedging or uncertainty intensity in the language

The research question:

```text
Do prediction markets correctly weight Fed communication by speaker credibility, voting status, and information content?
```

This preserves the core contribution of the current project while making the text signal more economically precise.

## Hedging and Asset Allocation Applications

Even if Kalshi prices are efficient, they can still be useful. Efficient prices are not useless; they are market-implied probability distributions.

Potential applications:

- Equity exposure: use FOMC uncertainty to size positions in banks, real estate, financials, and broad equity beta.
- Rates exposure: use expected policy moves and surprise risk for Treasury ETFs, SOFR futures, Fed funds futures, and yield-curve trades.
- Credit exposure: use policy uncertainty as a stress input for investment-grade and high-yield credit.
- FX and gold: use rate-path probabilities as inputs for USD and real-rate-sensitive assets.
- Personal risk management: reduce concentrated exposure around high-uncertainty FOMC events.
- Institutional hedging: translate prediction-market probabilities into scenario weights for portfolio stress tests.

The practical extension is to treat Kalshi not only as a trading venue, but as a live macro probability surface.

## Trading Kalshi Contracts Themselves

The current findings suggest that by `T-7`, a simple directional strategy is unlikely to have meaningful edge. However, earlier horizons may still be worth testing.

The correct framing is not "is Kalshi right?" The correct framing is:

```text
Is the executable contract price lower or higher than the best estimate of true probability, after costs and risk?
```

Potential areas for future study include:

- Early-window miscalibration before major macro releases are fully processed
- Short-lived underreaction after CPI, payrolls, PCE, or major Fed speeches
- Low-liquidity or wide-spread conditions
- Tail-probability contracts where the modal outcome is efficiently priced but extreme outcomes may not be
- Cross-market basis between Kalshi probabilities and rates-implied probabilities

These ideas should be treated as research hypotheses, not conclusions. The burden of proof is an out-of-sample backtest with realistic costs.

## Suggested Next Paper Framing

A clean title for the next project could be:

```text
Who Moves Macro Prediction Markets? Speaker Identity, Voting Status, and Cross-Market Disagreement in FOMC Expectations
```

The contribution would be to move from aggregate Fed tone to speaker-conditional information content. The project would test whether prediction markets incorporate information from Fed speakers at different speeds depending on voting status, regional conditions, hedging language, and disagreement regimes.

That would build directly on the current result: Kalshi is efficient by the final week, but the open question is how and when that efficiency emerges.
