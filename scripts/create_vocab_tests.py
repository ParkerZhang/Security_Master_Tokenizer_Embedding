#!/usr/bin/env python3
"""
Generate test cases for all security identifiers.

Creates queries that contain identifiers in different positions:
- Front: "AAPL stock info"
- Middle: "Get price for AAPL on NASDAQ"
- End: "Search for NASDAQ stock"

Usage:
    python create_vocab_tests.py
    python create_vocab_tests.py --limit 500
"""

import argparse
import pandas as pd
import random
import os


# Query templates
TEMPLATES = {
    'isin': [
        ("front", "What is {}?"),
        ("front", "Find {} security"),
        ("front", "Lookup {}"),
        ("middle", "Search database for {}"),
        ("middle", "Get info on {} ISIN"),
        ("end", "Look up security with ISIN {}"),
        ("end", "Find info for {}"),
    ],
    'ticker': [
        ("front", "{} price"),
        ("front", "{} stock"),
        ("front", "Quote {}"),
        ("middle", "Get {} on {}"),
        ("middle", "{} traded on {}"),
        ("end", "Look up {}"),
        ("end", "Find {} info"),
    ],
    'sedol': [
        ("front", "{} SEDOL"),
        ("middle", "Find SEDOL {}"),
        ("end", "Look up security {}"),  # sedol at end
        ("end", "Find info for SEDOL {}"),
    ],
    'mic': [
        ("front", "{} exchange"),
        ("middle", "Trade on {}"),
        ("middle", "List exchange {}"),
        ("end", "Stocks on {}"),
        ("end", "Find stocks for {}"),
    ],
    'currency': [
        ("front", "{} value"),
        ("middle", "in {}"),
        ("middle", "traded {}"),
        ("end", "Denominated in {}"),
    ],
    'exchange': [
        ("front", "{} market"),
        ("middle", "trade on {}"),
        ("end", "Find stocks on {}"),
    ],
}


def create_tests(df, id_type, limit=100):
    """Create test cases for a specific identifier type"""
    rows = []
    identifiers = df[id_type].dropna().unique()
    
    if len(identifiers) == 0:
        return rows
    
    sample_ids = random.sample(list(identifiers), min(limit, len(identifiers)))
    
    for identifier in sample_ids:
        # Get related data for context
        subset = df[df[id_type] == identifier]
        if len(subset) == 0:
            continue
        
        # Get related identifiers for context
        related_isin = subset['isin'].iloc[0] if 'isin' in subset.columns else None
        related_ticker = subset['ticker'].iloc[0] if 'ticker' in subset.columns else None
        related_mic = subset['mic'].iloc[0] if 'mic' in subset.columns and pd.notna(subset['mic'].iloc[0]) else None
        related_exchange = subset['exchange_code'].iloc[0] if 'exchange_code' in subset.columns else None
        related_currency = subset['currency'].iloc[0] if 'currency' in subset.columns else None
        
        templates = TEMPLATES.get(id_type, [])
        
        for pos, template in templates:
            test_input = template
            
            # Replace placeholders with actual values
            placeholders = template.count('{}')
            
            if id_type == 'isin':
                test_input = template.format(identifier)
            elif id_type == 'ticker':
                if '{}' in template and placeholders == 1:
                    test_input = template.format(identifier)
                elif placeholders == 2:
                    # Use related MIC or exchange for context
                    context = related_mic or related_exchange or ''
                    if context:
                        test_input = template.format(identifier, context)
            elif id_type == 'sedol':
                test_input = template.format(identifier)
            elif id_type == 'mic':
                if '{}' in template:
                    test_input = template.format(identifier)
            elif id_type == 'currency':
                test_input = template.format(identifier)
            elif id_type == 'exchange':
                test_input = template.format(identifier)
            
            rows.append({
                'input': test_input,
                'id_type': id_type,
                'expected_value': identifier,
                'position': pos,
                'related_isin': related_isin,
                'related_ticker': related_ticker,
                'related_mic': related_mic,
            })
    
    return rows


def main():
    parser = argparse.ArgumentParser(description='Generate vocab test cases')
    parser.add_argument('--limit', type=int, default=100, help='Test cases per type')
    parser.add_argument('--output', default='data/vocab/vocab_test_cases.csv')
    args = parser.parse_args()
    
    # Load symbols full to get related data
    print("Loading symbols_full.csv...")
    df = pd.read_csv('data/vocab/symbols_full.csv')
    print(f"  Rows: {len(df)}")
    
    print(f"\nCreating test cases (limit={args.limit})...")
    
    all_tests = []
    
    # ISIN tests
    print("  ISIN tests...")
    all_tests.extend(create_tests(df, 'isin', args.limit))
    
    # Ticker tests
    print("  Ticker tests...")
    all_tests.extend(create_tests(df, 'ticker', args.limit))
    
    # MIC tests
    print("  MIC tests...")
    all_tests.extend(create_tests(df, 'mic', args.limit))
    
    # SEDOL tests
    print("  SEDOL tests...")
    all_tests.extend(create_tests(df, 'sedol', args.limit))
    
    # Currency tests
    print("  Currency tests...")
    currencies = pd.read_csv('data/vocab/currencies.txt', header=None)[0].tolist()
    for currency in currencies[:args.limit]:
        pos = random.choice(['front', 'middle', 'end'])
        template = random.choice([t[1] for t in TEMPLATES['currency'] if t[0] == pos])
        all_tests.append({
            'input': template.format(currency),
            'id_type': 'currency',
            'expected_value': currency,
            'position': pos,
            'related_isin': '',
            'related_ticker': '',
            'related_mic': '',
        })
    
    # Exchange tests
    print("  Exchange tests...")
    exchanges = pd.read_csv('data/vocab/exchanges.txt', header=None)[0].tolist()
    for exchange in exchanges[:args.limit]:
        pos = random.choice(['front', 'middle', 'end'])
        template = random.choice([t[1] for t in TEMPLATES['exchange'] if t[0] == pos])
        all_tests.append({
            'input': template.format(exchange),
            'id_type': 'exchange',
            'expected_value': exchange,
            'position': pos,
            'related_isin': '',
            'related_ticker': '',
            'related_mic': '',
        })
    
    # Shuffle and save
    result = pd.DataFrame(all_tests)
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)
    
    result.to_csv(args.output, index=False)
    
    print(f"\nSaved to {args.output}")
    print(f"  Total: {len(result)}")
    print(f"\n  By type:")
    for id_type in result['id_type'].unique():
        count = len(result[result['id_type'] == id_type])
        print(f"    {id_type}: {count}")
    print(f"\n  By position:")
    for pos in result['position'].unique():
        count = len(result[result['position'] == pos])
        print(f"    {pos}: {count}")


if __name__ == '__main__':
    main()