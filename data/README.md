# Data Access

Raw and intermediate data files are not committed to git because they are large and include source data subject to API/provider terms.

Parquet and CSV data exports are available through the project Google Drive folder:

[https://drive.google.com/drive/folders/1QfwrOqGvLdVPa1l9H3pE6_VBvI8pyCF0?usp=sharing](https://drive.google.com/drive/folders/1QfwrOqGvLdVPa1l9H3pE6_VBvI8pyCF0?usp=sharing)

Equity return data can be reconstructed by running the notebooks, which pull daily prices through `yfinance`.

Snowflake source organization:

- Database: `PREDMARKET`
- Schemas: `KALSHI`, `POLYMARKET`
