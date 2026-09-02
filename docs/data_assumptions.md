# Data Assumptions and Limitations

## Data Source

The platform is designed to remain provider-independent.

CRSP through WRDS was the preferred data source, but access was unavailable during development. The initial implementation therefore uses Sharadar as the primary fallback provider.

Sharadar provides daily prices, security metadata, corporate actions, and point-in-time historical market capitalization for active and delisted U.S. securities.

## Security Universe

The intended research universe consists of U.S. ordinary common stocks listed on NYSE, AMEX, or Nasdaq.

For the Sharadar implementation, security metadata are restricted to the fundamentals universe of primary-class common stocks. Exchange names are standardized so that `NYSEMKT` maps to `AMEX`.

Sharadar's permanent `permaticker` identifier is mapped to the platform's `security_id`. Ticker symbols are treated as metadata rather than permanent security identifiers.

Historical exchange classifications are not currently reconstructed separately for every security-month. This differs from CRSP's historical exchange-code structure and is a provider-specific limitation.

## Monthly Observations

The standardized panel contains one row per security and calendar month.

Daily observations are aggregated using the final available trading observation in each month, but the resulting observation is labeled with the calendar month-end date.

Monthly returns are calculated from Sharadar's adjusted closing prices, which account for stock splits, cash dividends, and spinoffs.

A return is calculated only when the immediately preceding calendar month is observed. Returns are not allowed to skip across missing calendar months.

## Market Capitalization

Sharadar supplies historical point-in-time market capitalization in USD millions. These values are converted to actual USD before entering the standardized panel.

Because the Sharadar implementation uses provider-supplied market capitalization, `shares_outstanding` is currently left missing rather than reconstructed synthetically from price and market capitalization.

Missing share counts are therefore an expected provider limitation and not automatically treated as an integrity failure.

Universe membership uses lagged market capitalization to avoid using contemporaneous information when forming the portfolio universe.

## Delistings

Sharadar includes active and delisted securities and provides corporate-action information about delisting events.

However, Sharadar does not provide a direct equivalent of CRSP's delisting-return (`DLRET`) field.

When a delisting event is known but the associated terminal delisting return is unavailable, `delisting_return` remains missing rather than being silently replaced with zero.

Any later treatment or imputation of missing terminal delisting returns must be an explicit research-methodology decision and should be subjected to robustness analysis.

## Validation Sample

The provider pipeline was validated using a small real Sharadar sample containing AAPL, MSFT, and JPM during 2025.

The sample confirmed:

* correct daily-to-monthly aggregation;
* correct adjusted-price monthly returns;
* correct conversion of market capitalization units;
* successful mapping of Sharadar permanent identifiers;
* correct metadata normalization;
* preservation of expected missing values;
* lagged market-cap universe formation;
* no duplicate security-month observations;
* no impossible price, return, or market-cap values in the validation sample.

The sample contained no delisting events, so rare delisting behavior is covered by synthetic unit and integration tests rather than the small real-data validation sample.

The small sample validates the data pipeline and provider assumptions.