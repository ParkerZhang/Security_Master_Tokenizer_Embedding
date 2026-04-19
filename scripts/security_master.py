#!/usr/bin/env python3
"""
Security Master Data Generator

Generates isins_companies.csv with:
- ISIN, Ticker, Exchange, Currency, Company Name, Summary
- MIC (ISO 10383), SEDOL

Usage:
    python security_master.py
    python security_master.py --limit 1000  # limit for testing
"""

import argparse
import pandas as pd
import financedatabase as fd


# ISO 10383 MIC mapping (FinanceDatabase codes -> ISO MIC)
MIC_MAPPING = {
    'NMS': 'XNAS', 'NYQ': 'XNYS', 'LSE': 'XLON', 'FRA': 'XFRA',
    'BER': 'XBER', 'DUS': 'XDUS', 'MUN': 'XMUN', 'STU': 'XSAT',
    'HAM': 'XHAM', 'HAN': 'XHAN', 'HKG': 'XHKG', 'SAO': 'XBVMF',
    'MEX': 'XMEX', 'MIL': 'XMIL', 'PAR': 'XPAR', 'AMS': 'XAMS',
    'TOR': 'XTSE', 'VAN': 'XTSX', 'OSL': 'XOSL', 'ASX': 'XASX',
    'VIE': 'XWBO', 'EBS': 'XWBO', 'BUE': 'XBUE', 'IOB': 'XIOB',
    'JKT': 'XIDX', 'KLS': 'XKLS', 'SET': 'XSET', 'TLO': 'XTLO',
    'CAI': 'XCAI', 'PNK': 'PNK', 'GER': 'XFRA', 'STO': 'XSTO',
    'JPX': 'XTJA', 'CNQ': 'XSSE', 'SGO': 'XSGO', 'JNB': 'XJNB',
    'BSE': 'XNSE', 'SES': 'XSES', 'NCM': 'XNGM', 'MCE': 'XMCE',
    'ASE': 'XASE', 'NZE': 'XNZE', 'NSE': 'XNSE', 'LIS': 'XLIS',
    'TWO': 'XTAI', 'TAI': 'XTAI', 'CN': 'XSSE', 'IST': 'XIST',
    'SHZ': 'XSHE', 'CPH': 'XCPH', 'BRU': 'XBRU', 'SHH': 'XHKG',
}


def clean_name(name):
    """Clean company name by removing common suffixes"""
    if pd.isna(name):
        return ''
    name = str(name)
    suffixes = ['DL-,01', 'DL-001', 'ORD', 'ORD ', 'NCD', 'CP ', 'Registered']
    for suffix in suffixes:
        name = name.replace(suffix, '')
    return name.strip()[:60]


def download_sedol():
    """Download SEDOL mapping"""
    try:
        url = 'https://github.com/Wenzhi-Ding/Std_Security_Code/raw/main/isin/sedol.pq'
        sedol_df = pd.read_parquet(url).drop_duplicates(subset=['isin'], keep='first')
        return dict(zip(sedol_df['isin'], sedol_df['sedol']))
    except Exception as e:
        print(f"Warning: Could not download SEDOL: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Generate security master data')
    parser.add_argument('--limit', type=int, default=None, help='Limit securities')
    parser.add_argument('--output', default='data/isins_companies.csv', help='Output file')
    args = parser.parse_args()
    
    print("Fetching data from FinanceDatabase...")
    eq = fd.Equities()
    df = eq.select()
    df = df.reset_index()  # symbol becomes column 'symbol'
    
    if args.limit:
        df = df.head(args.limit)
        print(f"  Limited to {args.limit}")
    
    print(f"  Total: {len(df)}")
    
    # Build company name mapping (best name per ISIN)
    print("Cleaning company names...")
    name_map = {}
    for isin, group in df.groupby('isin', dropna=True):
        names = group['name'].dropna().unique()
        best = ''
        for n in names:
            n = str(n)
            if 'DL-' not in n and 'ORD' not in n:
                best = n
                break
        if not best and len(names) > 0:
            best = str(names[0])
        name_map[isin] = clean_name(best)
    
    df['company'] = df['isin'].map(name_map)
    
    # Select columns
    result = df[['isin', 'company', 'symbol', 'exchange', 'currency', 'summary']].copy()
    result = result.rename(columns={'symbol': 'ticker', 'exchange': 'exchange_code'})
    result = result.drop_duplicates(subset=['isin', 'ticker'])
    
    # Add MIC
    print("Adding ISO MIC...")
    result['mic'] = result['exchange_code'].map(MIC_MAPPING)
    
    # Add SEDOL
    print("Adding SEDOL...")
    sedol_map = download_sedol()
    result['sedol'] = result['isin'].map(sedol_map)
    
    # Save
    result.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")
    print(f"  Rows: {len(result)}")
    print(f"  Unique ISINs: {result['isin'].nunique()}")
    print(f"  With MIC: {result['mic'].notna().sum()}")
    print(f"  With SEDOL: {result['sedol'].notna().sum()}")


if __name__ == '__main__':
    main()