# Security Master Tokenizer & Embedding

A comprehensive dataset and tooling for building financial security identifiers for LLM tokenization.

## Data Structure

```
data/
├── isins_companies.csv      # Main security master (160K rows, 9,133 unique ISINs)
├── isins_test_cases.csv     # Test cases for ISIN recognition
└── vocab/
    ├── isins.txt          # Unique ISINs (9,133)
    ├── tickers.txt         # Ticker symbols (159,999)
    ├── sedols.txt         # SEDOL codes (8,807)
    ├── mics.txt           # ISO 10383 MIC codes (46)
    ├── currencies.txt     # Currencies (41)
    ├── exchanges.txt      # Exchange codes (82)
    └── symbols_full.csv   # Combined mapping (39,373)
```

## Quick Start

### 1. Generate Security Master

```bash
# Full generation (uses FinanceDatabase)
python scripts/security_master.py

# Test with limit
python scripts/security_master.py --limit 1000
```

### 2. Extract Vocabulary

```bash
python scripts/extract_vocab.py
```

### 3. Generate Test Cases

```bash
python scripts/create_vocab_tests.py
python scripts/create_vocab_tests.py --limit 50
```

## Output Files

### isins_companies.csv (Main Master)

| Column | Description | Coverage |
|--------|-------------|----------|
| isin | ISIN | 100% |
| company | Company name | 100% |
| ticker | Ticker symbol | 100% |
| exchange_code | Exchange code | 100% |
| currency | Currency | 100% |
| summary | Description | 100% |
| mic | ISO 10383 MIC | 81.8% |
| sedol | SEDOL | 24.1% |

### vocab/*.txt

| File | Count | Example |
|------|-------|---------|
| isins.txt | 9,133 | US0231351067 |
| tickers.txt | 159,999 | AMZN, AAPL |
| sedols.txt | 8,807 | 200001 |
| mics.txt | 46 | XNAS, XNYS |
| currencies.txt | 41 | USD, EUR |
| exchanges.txt | 82 | NMS, LSE |

### vocab/vocab_test_cases.csv

| Column | Description |
|--------|-------------|
| input | Test query |
| id_type | Type (isin/ticker/sedol/mic/currency/exchange) |
| expected_value | Expected identifier |
| position | front/middle/end |
| related_isin | Related ISIN |
| related_ticker | Related ticker |
| related_mic | Related MIC |

## Sample Data

### Amazon (US0231351067)

| ticker | company | mic | sedol |
|--------|---------|-----|-------|
| AMZN | Amazon.com, Inc. | XNAS | 200001 |
| AMZ.DE | Amazon.com, Inc. | XFRA | 200001 |
| AMZN.MI | Amazon.com, Inc. | XMIL | 200001 |
| AMZO34.SA | Amazon.com, Inc. | XBVMF | 200001 |

### Microsoft (US5949181045)

| ticker | company | mic | sedol |
|--------|---------|-----|-------|
| MSFT | Microsoft Corporation | XNAS | B2823C4 |
| MSF.DE | Microsoft Corporation | XFRA | B2823C4 |
| MSFT.MI | Microsoft Corporation | XMIL | B2823C4 |

## Scripts

| Script | Description |
|--------|-------------|
| security_master.py | Generate isins_companies.csv |
| extract_vocab.py | Extract vocab files |
| create_vocab_tests.py | Generate test cases |

## Requirements

- Python 3.11+
- financedatabase
- pandas
- pyarrow

Install:
```bash
pip install financedatabase pandas pyarrow
```

## Notes

- Names are cleaned (removed "DL-", "ORD" suffixes)
- Multi-listed securities have multiple rows (one per exchange)
- Source: FinanceDatabase + Std_Security_Code