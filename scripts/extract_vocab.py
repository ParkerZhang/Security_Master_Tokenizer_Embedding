#!/usr/bin/env python3
"""
Extract unique identifiers from security master for LLM vocabulary building.

Outputs:
- isins.txt - All unique ISINs
- tickers.txt - All unique tickers
- sedols.txt - All unique SEDOLs
- cusips.txt - All unique CUSIPs
- mics.txt - All unique ISO MIC codes
- currencies.txt - All unique currencies
- exchanges.txt - All exchange codes (original)
- symbols_full.csv - Combined identifier mapping

Usage:
    python extract_vocab.py
    python extract_vocab.py --input data/isins_companies.csv
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Extract vocabulary from security master')
    parser.add_argument('--input', default='data/isins_companies.csv', help='Input file')
    parser.add_argument('--output-dir', default='data/vocab', help='Output directory')
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Total rows: {len(df)}")
    print(f"  Unique ISINs: {df['isin'].nunique()}")
    
    # ISINs
    isins = df['isin'].dropna().unique()
    print(f"\nISINs: {len(isins)}")
    with open(f'{args.output_dir}/isins.txt', 'w') as f:
        for isin in sorted(isins):
            f.write(isin + '\n')
    
    # Tickers
    tickers = df['ticker'].dropna().unique()
    print(f"Tickers: {len(tickers)}")
    with open(f'{args.output_dir}/tickers.txt', 'w') as f:
        for t in sorted(tickers):
            f.write(t + '\n')
    
    # SEDOLs
    sedols = df['sedol'].dropna().unique()
    print(f"SEDOLs: {len(sedols)}")
    with open(f'{args.output_dir}/sedols.txt', 'w') as f:
        for s in sorted(sedols):
            f.write(s + '\n')
    
    # CUSIPs
    cusips = df['cusip'].dropna().unique() if 'cusip' in df.columns else []
    print(f"CUSIPs: {len(cusips)}")
    if len(cusips) > 0:
        with open(f'{args.output_dir}/cusips.txt', 'w') as f:
            for c in sorted(cusips):
                f.write(c + '\n')
    
    # MICs
    mics = df['mic'].dropna().unique()
    print(f"MICs: {len(mics)}")
    with open(f'{args.output_dir}/mics.txt', 'w') as f:
        for m in sorted(mics):
            f.write(m + '\n')
    
    # Currencies
    currencies = df['currency'].dropna().unique()
    print(f"Currencies: {len(currencies)}")
    with open(f'{args.output_dir}/currencies.txt', 'w') as f:
        for c in sorted(currencies):
            f.write(c + '\n')
    
    # Exchange codes (original)
    exchanges = df['exchange_code'].dropna().unique()
    print(f"Exchange codes: {len(exchanges)}")
    with open(f'{args.output_dir}/exchanges.txt', 'w') as f:
        for e in sorted(exchanges):
            f.write(e + '\n')
    
    # Combined CSV
    result = df[['isin', 'ticker', 'mic', 'sedol', 'exchange_code', 'currency']].copy()
    result = result.drop_duplicates(subset=['isin', 'ticker'])
    result = result.dropna(subset=['isin', 'ticker'])
    result.to_csv(f'{args.output_dir}/symbols_full.csv', index=False)
    print(f"\nSaved {len(result)} rows to {args.output_dir}/symbols_full.csv")
    
    print(f"\n=== VOCAB FILES ===")
    print(f"  data/vocab/isins.txt       - {len(isins)} ISINs")
    print(f"  data/vocab/tickers.txt      - {len(tickers)} tickers")
    print(f"  data/vocab/sedols.txt     - {len(sedols)} SEDOLs")
    print(f"  data/vocab/mics.txt      - {len(mics)} MICs")
    print(f"  data/vocab/currencies   - {len(currencies)} currencies")
    print(f"  data/vocab/exchanges     - {len(exchanges)} exchanges")


if __name__ == '__main__':
    main()