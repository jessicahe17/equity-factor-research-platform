# Research Protocol

## Objective

The first goal of this project is to test whether cross-sectional equity momentum contains useful information about future U.S. stock returns.

The initial research question is:

> Does a conventional 12–2 momentum signal predict one-month-ahead stock returns, and is the relationship robust across time and reasonable portfolio construction choices?

The purpose is not to search for the parameter combination that produces the best backtest. The main focus is whether the signal shows persistent and defensible predictive structure.

## Universe and Sample

The initial universe will consist of approximately the largest 1,000 eligible U.S. common stocks listed on NYSE, AMEX, and Nasdaq.

The universe must be reconstructed using information available at each historical date. Using only stocks that survive to the present would introduce survivorship bias.

The intended main sample period is January 2001 through December 2025, with additional earlier observations used when needed to construct momentum signals.

## Data

CRSP through WRDS is the preferred data source because it provides point-in-time security information, returns, market capitalization data, and delisting information.

If CRSP is unavailable, a fallback source will be chosen with particular attention to survivorship, delisting, corporate actions, and historical universe construction.

Provider-specific raw data should eventually be converted into a standardized internal equity dataset so the rest of the research pipeline is not tied to one data vendor.

## Momentum Signal

The first factor will be conventional 12–2 momentum.

For stock $i$ held during month $t$,

$$
MOM_{i,t} = \prod_{j=2}^{12}(1+r_{i,t-j})-1.
$$

The most recent month is excluded to separate intermediate-horizon momentum from short-term reversal effects.

The 12–2 definition is fixed before examining project results. Alternative lookback windows may later be studied as robustness checks rather than used to optimize historical performance.

## Timing and Eligibility

At the end of month $t-1$, we will:

1. determine the eligible stock universe using information available at that time;
2. calculate momentum signals;
3. rank stocks;
4. form portfolios that are held during month $t$.

The return being predicted must therefore occur strictly after the information used to construct the signal.

Stocks without enough history to calculate the full momentum window will be excluded for that month. Missing observations inside the required signal window will not automatically be treated as zero or ignored.

If a stock delists during a holding period, the associated economic loss must remain part of the portfolio return.

## Factor Evaluation

The primary signal-level metric will be monthly Spearman Rank Information Coefficient:

$$
IC_t = Corr_{Spearman}(MOM_{i,t},R_{i,t+1}).
$$

We will examine the average IC, its stability through time, the fraction of positive months, and appropriate statistical significance measures.

A positive IC is evidence of predictive cross-sectional structure, not proof by itself that the factor is genuine or economically useful.

Stocks will also be sorted into momentum quintiles. We will compare future returns across the five groups and examine the $Q_5-Q_1$ long-short spread.

The baseline portfolio sort will use equal weights. Value-weighted portfolios and decile sorts will later serve as robustness checks.

## Research Principles

Signal quality and trading performance will be treated as separate questions.

A factor may contain predictive information while producing weak net performance because of turnover, transaction costs, risk exposures, or implementation constraints.

The project will therefore explicitly guard against:

* look-ahead bias;
* survivorship and delisting bias;
* incorrect signal-return alignment;
* unrealistic liquidity or execution assumptions;
* transaction-cost omission;
* data snooping and parameter overfitting;
* multiple testing;
* misleading benchmark or risk exposures.

The overall research process is:

**Question → Hypothesis → Data → Methodology → Experiment → Validation → Robustness → Interpretation → Limitations**

The central rule is:

> Specify the research design first, test it second, and interpret the results without tuning the methodology just to improve the backtest.