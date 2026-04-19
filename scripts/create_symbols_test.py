import pandas as pd
import random

df = pd.read_csv('/home/src/Security_Master_Tokenizer_Embedding/data/symbol_list.csv')

# Templates for test cases
templates_front = [
    "{}",
    "{} price",
    "Quote {}",
]

templates_middle = [
    "What's the price of {}?",
    "Show data on {} traded on {}",
    "{} {} current price",
    "Find {} on {} exchange",
    "Stock {} ({}) trading",
    "Get quote for {} in {}",
]

templates_end = [
    "Look up security {}",
    "Find info for {}",
    "Search for {} on market",
]

test_cases = []
used = set()

# Get unique ISINs
unique_isins = df['isin'].dropna().unique()[:400]

for isin in unique_isins:
    subset = df[df['isin'] == isin]
    if len(subset) < 1:
        continue
    
    row = subset.iloc[0]
    ticker = row['ticker']
    mic = row['mic'] if pd.notna(row['mic']) else ''
    sedol = row['sedol'] if pd.notna(row['sedol']) else ''
    exchange = row['exchange_code']
    
    if pd.isna(ticker):
        continue
    
    # Front position (ticker at start)
    if random.random() < 0.3:
        input_text = random.choice(templates_front).format(ticker)
        test_cases.append({
            'input': input_text,
            'expected_isin': isin,
            'expected_ticker': ticker,
            'expected_mic': mic,
            'expected_exchange': exchange,
            'expected_sedol': sedol,
            'position': 'front'
        })
    
    # Middle position (ticker in middle)
    if mic:
        input_text = random.choice(templates_middle).format(ticker, mic)
    else:
        input_text = random.choice(templates_middle).format(ticker, exchange)
    test_cases.append({
        'input': input_text,
        'expected_isin': isin,
        'expected_ticker': ticker,
        'expected_mic': mic,
        'expected_exchange': exchange,
        'expected_sedol': sedol,
        'position': 'middle'
    })
    
    # End position
    if sedol:
        input_text = random.choice(templates_end).format(sedol)
        test_cases.append({
            'input': input_text,
            'expected_isin': isin,
            'expected_ticker': ticker,
            'expected_mic': mic,
            'expected_exchange': exchange,
            'expected_sedol': sedol,
            'position': 'end'
        })

result = pd.DataFrame(test_cases)
result = result.sample(frac=1, random_state=42).reset_index(drop=True)

print(f'Total: {len(result)}')
print('\\nPosition dist:')
print(result['position'].value_counts())

print('\\nSample test cases:')
print(result.sample(10)[['input', 'expected_ticker', 'expected_mic', 'position']].to_string(index=False))

result.to_csv('/home/src/Security_Master_Tokenizer_Embedding/data/symbols_test_cases.csv', index=False)
print('\\nSaved to data/symbols_test_cases.csv')