import pandas as pd
import random

templates_front = [
    "What is the current price of {}?",
    "Can you get details for {}?",
    "Show me information on security {}",
    "Find {} in our database",
    "Lookup {} symbol",
    "Get quote for {}",
    "Search for ISIN {}",
    "What company is associated with {}?",
]

templates_middle = [
    "I need data on {} for my portfolio",
    "Please check {} and related securities",
    "Compare {} with market indices",
    "Is {} a good investment?",
    "Get metrics for {} in the tech sector",
    "What's the history of {}?",
    "Analyze {} performance",
    "Show me {} financial data",
]

templates_end = [
    "Please look up security with ISIN {}",
    "Can you find information on this security {}",
    "Search the database for {}",
    "Get details for the instrument {}",
    "Find data for {} I'm interested in",
    "Look up {} in the system",
    "Get information on the bond {}",
    "Check the metrics for {}",
]

context_templates = [
    "The ISIN {} represents a stock listed on NYSE",
    "{} is a bond from US markets",
    "Security {} traded in European exchanges",
    "The instrument {} from Asian markets",
    "{} listed on Tokyo Stock Exchange",
    "European security {} from Germany",
    "UK listed {} on LSE",
    "Canadian stock {} on TSX",
]

df = pd.read_csv('/home/src/Security_Master_Tokenizer_Embedding/data/isins_1000.csv')
isins = df['isin'].dropna().unique()[:300]

test_cases = []

for isin in isins:
    test_cases.append({'ISIN': isin, 'Input': random.choice(templates_front).format(isin)})
    test_cases.append({'ISIN': isin, 'Input': random.choice(templates_middle).format(isin)})
    test_cases.append({'ISIN': isin, 'Input': random.choice(templates_end).format(isin)})
    test_cases.append({'ISIN': isin, 'Input': random.choice(context_templates).format(isin)})

result = pd.DataFrame(test_cases)
result = result.sample(frac=1, random_state=42).reset_index(drop=True)

print(f'Total test cases: {len(result)}')
print(result.sample(10, random_state=42).to_string(index=False))

result.to_csv('/home/src/Security_Master_Tokenizer_Embedding/data/isins_test_cases.csv', index=False)
print(f'\nSaved to data/isins_test_cases.csv')