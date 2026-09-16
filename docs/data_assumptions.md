# Data Assumptions and Construction

## Data Source

The final reported empirical analysis uses monthly U.S. equity data from **Compustat**.

The production pipeline is built from three Compustat extracts:

* monthly security-level market data;
* historical exchange classifications;
* security-level delisting information.

The raw Compustat files are proprietary and are therefore stored outside the repository and are not distributed with the project.

The platform was designed to keep provider-specific ingestion separate from the downstream research pipeline. Earlier in development, a Sharadar-based provider was implemented and validated using sample data. That provider remains in the repository as a prototype and testing path, but the empirical results reported in this project are based on Compustat.

The standardized downstream panel is designed to provide the information required for point-in-time universe construction, momentum signal formation, portfolio sorting, and predictive analysis without relying on future observations.


## Security Identifiers and Universe Eligibility

Each security is identified using the Compustat combination of `gvkey` and `iid`. The platform constructs a stable `security_id` from these fields while preserving leading zeros.

Ticker symbols are treated as descriptive metadata rather than permanent security identifiers.

### Historical Exchange Eligibility

The research universe is restricted to ordinary U.S. common stocks listed on:

* NYSE;
* AMEX; or
* Nasdaq.

Historical exchange membership is determined using Compustat security-history records. The eligible exchange codes are:

* `11` — NYSE
* `12` — AMEX
* `14` — Nasdaq

An exchange classification is applied to a security-month only when the observation date falls within that record's effective interval:

```text
efffrom <= observation date <= effthru
```

A missing `effthru` is treated as open-ended.

The pipeline does not bridge gaps between historical exchange records, infer classifications from future observations, or fill missing historical classifications using later security metadata.

### Point-in-Time Universe Construction

The baseline research universe consists of the Top 1,000 eligible securities by lagged market capitalization each month.

Membership for outcome month $t$ is determined using information from $t-1$:

* lagged base eligibility; and
* lagged market capitalization.

Among eligible securities with nonmissing lagged market capitalization, securities are ranked by lagged market capitalization and the largest 1,000 are selected.

Importantly, a security is not removed from the outcome-month universe merely because its current-month eligibility changes, it delists, or it otherwise disappears during month $t$. Universe membership is based on information available before the outcome month rather than on information revealed during or after it.


## Monthly Market Data and Returns

The Compustat monthly security file provides the core price, return, and shares data used in the empirical panel.

### Monthly Returns

Compustat field `trt1m` is interpreted as the security's monthly return expressed in percentage points. It is converted to decimal form before entering the standardized panel:

```text
return_ex_delist = trt1m / 100
```

Missing monthly returns remain missing and are not replaced with zero.

Momentum construction also requires calendar-month continuity. A security must have the required sequence of monthly observations over the formation window; returns are not allowed to bridge gaps in the monthly history.

### Prices and Shares

Monthly price is taken from `prccm`.

Shares outstanding are taken from `cshom`, which is interpreted as the actual number of shares rather than a value requiring an additional scaling factor.

Market capitalization is therefore constructed as:

```text
market_cap = prccm * cshom
```

The resulting market capitalization is used for both universe ranking and portfolio weighting.

### Timing of Market Capitalization

Current-month market capitalization is not used to determine the month $t$ research universe or value-weighted portfolio weights.

Instead, the pipeline uses **lagged market capitalization** from $t-1$. This ensures that portfolio formation relies only on information available before the outcome month and avoids using contemporaneous price or shares information from month $t$.

## Delisting Treatment

The Compustat security file provides delisting-related information through fields including `dldtei` and `dlrsni`.

The pipeline distinguishes between performance-related delistings and other terminal security records.

### Performance-Related Delistings

For delisting reason codes:

* `02`
* `03`

the analysis applies a −30% delisting return.

This assumption is incorporated into the construction of `total_return` so that securities experiencing performance-related delistings are not treated as though they simply disappear with no terminal loss.

### Other Terminal Records

For non-performance-related terminal records where no appropriate delisting return is available, the delisting return remains missing rather than being automatically replaced with zero.

The pipeline therefore avoids treating an unknown terminal return as a flat return.

### Validation

Full-panel validation identified 3,077 performance-related delisting events. All were assigned the intended −30% delisting return, and the validation checks found no inconsistencies in the application of the rule.

Delisting treatment is applied explicitly as part of return construction rather than by removing affected securities from the outcome-month universe.

## Momentum Signal Construction and Missing Data

The baseline factor is conventional 12–2 momentum.

For outcome month $t$, the momentum signal is constructed from cumulative returns over:

```text
t-12 through t-2
```

Month $t-1$ is intentionally skipped, and the outcome return is month $t$.

The signal is therefore already aligned to the outcome month. No additional shift is applied after signal construction.

### Warm-Up Period

The reported empirical sample runs from January 2001 through December 2025.

Monthly data from January through December 2000 are retained as a warm-up period so that early outcome months can accumulate the return history required for signal formation.

### Missing Returns

Missing monthly returns are not replaced with zero.

A momentum signal is formed only when the required return history is available with proper calendar-month continuity. The implementation does not compound across gaps in a security's monthly return history.

This treatment is intended to distinguish genuinely observed zero returns from missing information and to avoid creating artificial return histories.

### Alternative Formation Window

A 6–2 momentum specification is used as a pre-specified robustness check.

For outcome month $t$, this signal uses returns from:

```text
t-6 through t-2
```

The same timing, missing-data, and calendar-continuity rules apply as in the baseline 12–2 specification.

## Full Compustat Validation

The final Compustat pipeline was validated on the complete 2000–2025 monthly security panel before the empirical analysis was finalized.

Key validation checks included:

* no duplicate security-month observations in the raw monthly panel;
* correct preservation of stable security identifiers;
* historical exchange classifications applied only within their effective date ranges;
* point-in-time universe construction using lagged eligibility and lagged market capitalization;
* exactly 1,000 securities in the baseline universe for every reported month;
* consistent momentum-signal eligibility and signal availability;
* no zero-filling of missing monthly returns;
* validated application of the performance-related delisting-return rule; and
* consistency between independently computed analysis coverage and the observations used in monthly Rank IC calculations.

Across the 300 reported months from January 2001 through December 2025, the baseline universe contained exactly 1,000 securities per month.

The baseline 12–2 momentum signal was available for approximately 98.1% of baseline universe observations. Missing signals were investigated and were found to arise primarily from insufficient return history, especially for recent listings, rather than from a construction error.

Extreme momentum observations were also manually reconstructed during validation. The largest observed signal was confirmed directly from the underlying monthly returns and was therefore retained rather than winsorized or removed.

These checks were used to validate data construction and temporal alignment before interpreting the empirical momentum results.

## Prototype Provider Validation

Before the final Compustat pipeline was implemented, the provider architecture was prototyped and validated using a small real Sharadar sample containing AAPL, MSFT, and JPM during 2025.

The sample confirmed:

* correct daily-to-monthly aggregation;
* correct adjusted-price monthly returns;
* correct conversion of market capitalization units;
* successful mapping of Sharadar permanent identifiers;
* correct metadata normalization;
* preservation of expected missing values;
* lagged market-cap universe formation;
* no duplicate security-month observations; and
* no impossible price, return, or market-cap values in the validation sample.

The sample contained no delisting events, so Sharadar-specific delisting behavior was not validated using this small real-data sample. Rare cases were instead covered by synthetic unit and integration tests during the prototype stage.

The Sharadar provider remains in the repository as an alternative ingestion and testing path, but it is not the source of the empirical results reported in the final study. Those results are based on the Compustat pipeline described above.