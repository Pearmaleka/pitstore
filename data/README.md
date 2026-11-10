# Data

## `shiller_sp500_monthly.csv`

Monthly S&P Composite data from 1871: price, dividends, earnings, CPI and the long-term
interest rate, as compiled by Robert Shiller and republished by
[datasets/s-and-p-500](https://github.com/datasets/s-and-p-500) under the Open Data Commons
Public Domain Dedication and Licence (commit `87a2284`, SHA-256
`b797a3c8de4586af837b86a156b05f49be11ffab99e13171dae763b2a11bdf79`).

Notes that matter for point-in-time work:

- `Date` is the first day of the month the row describes. The experiments use the last day
  of the month as the event time.
- `SP500` is the monthly average of daily closes.
- `Dividend` and `Earnings` are trailing twelve-month figures, interpolated monthly from
  quarterly data, so a month's value is only final once its quarter has been reported.
- Values not yet reported are stored as `0.0`. In this copy, dividends and earnings end in
  June 2023 and the CPI and the long rate in September 2023; later rows only have prices.

Rebuild: `curl -o data/shiller_sp500_monthly.csv https://raw.githubusercontent.com/datasets/s-and-p-500/87a2284276f79d75041c71c8f6d9f2aa1c965e84/data/data.csv`
