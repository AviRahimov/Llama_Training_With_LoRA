#!/usr/bin/env python3

import pandas as pd
import numpy as np

# Load and examine the data more closely
df = pd.read_csv('/home/vi/Llama_Training_With_LoRA/combined_human_conversations.csv')

print("🔍 DETAILED DATA ANALYSIS")
print("=" * 50)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n📊 Data Quality Analysis:")
print(f"Total rows: {len(df)}")
print(f"Unique conversation IDs: {df['conversation_id'].nunique()}")

# Check for missing values
print(f"\nMissing values:")
for col in df.columns:
    missing = df[col].isnull().sum()
    print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")

# Check data types
print(f"\nData types:")
print(df.dtypes)

# Examine some problematic cases
print(f"\n🔍 Examining problematic data:")

# Check for extremely long texts
df['question_len'] = df['Question'].astype(str).str.len()
df['answer_len'] = df['Answer'].astype(str).str.len()

print(f"\nText length statistics:")
print(f"Question lengths: min={df['question_len'].min()}, max={df['question_len'].max()}, mean={df['question_len'].mean():.1f}")
print(f"Answer lengths: min={df['answer_len'].min()}, max={df['answer_len'].max()}, mean={df['answer_len'].mean():.1f}")

# Find rows with extremely long text
long_questions = df[df['question_len'] > 1000]
long_answers = df[df['answer_len'] > 1000]

print(f"\nExtremely long texts:")
print(f"Questions > 1000 chars: {len(long_questions)}")
print(f"Answers > 1000 chars: {len(long_answers)}")

if len(long_questions) > 0:
    print(f"\nSample long question (first 200 chars):")
    print(repr(long_questions.iloc[0]['Question'][:200]))

if len(long_answers) > 0:
    print(f"\nSample long answer (first 200 chars):")
    print(repr(long_answers.iloc[0]['Answer'][:200]))

# Check for unusual characters or encoding issues
print(f"\n🔍 Checking for unusual characters:")

# Look for non-standard characters
def check_unusual_chars(text):
    if pd.isna(text):
        return False
    text = str(text)
    # Check for very long lines or unusual patterns
    return len(text) > 5000 or '\x00' in text or len(text.split('\n')) > 100

unusual_questions = df[df['Question'].apply(check_unusual_chars)]
unusual_answers = df[df['Answer'].apply(check_unusual_chars)]

print(f"Questions with unusual chars: {len(unusual_questions)}")
print(f"Answers with unusual chars: {len(unusual_answers)}")

# Sample some normal conversations
print(f"\n📝 Sample conversations:")
for i in range(3):
    print(f"\nConversation {i+1}:")
    row = df.iloc[i]
    print(f"  ID: {row['conversation_id']}")
    print(f"  Question: {str(row['Question'])[:100]}...")
    print(f"  Answer: {str(row['Answer'])[:100]}...")

print(f"\n" + "=" * 50)
