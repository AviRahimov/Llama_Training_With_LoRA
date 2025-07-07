#!/usr/bin/env python3
"""
Conversation Quality Classifier

This script analyzes conversations from combined_human_conversations.csv 
and classifies each conversation as either:
- "normal": Coherent human conversation
- "random": Random keyboard typing or gibberish

Uses multiple approaches:
1. Pre-trained language model for text coherence
2. Statistical features for randomness detection
3. Linguistic patterns analysis
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML imports
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import nltk
from nltk.corpus import words

# Download required NLTK data
try:
    nltk.data.find('corpora/words')
except LookupError:
    print("Downloading NLTK words corpus...")
    nltk.download('words', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

class ConversationClassifier:
    def __init__(self):
        """Initialize the conversation classifier with pre-trained models."""
        print("🤖 Initializing Conversation Quality Classifier...")
        
        # Load pre-trained models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Use a model trained for text quality/coherence detection
        # RoBERTa is good at understanding text quality
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        print(f"Loading model: {self.model_name}")
        
        try:
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1
            )
            print("✅ Sentiment classifier loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load {self.model_name}, using default: {e}")
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                device=0 if self.device == "cuda" else -1
            )
        
        # Load English words for vocabulary check
        self.english_words = set(words.words())
        print(f"📖 Loaded {len(self.english_words)} English words")
        
        # Initialize TF-IDF for anomaly detection
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        print("✅ Classifier initialization complete!")
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract statistical features from text that indicate randomness."""
        if not text or len(text.strip()) == 0:
            return self._empty_features()
        
        text = text.lower().strip()
        
        # Basic statistics
        char_count = len(text)
        word_count = len(text.split())
        
        if word_count == 0:
            return self._empty_features()
        
        # Character-level features
        letter_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        special_char_count = sum(1 for c in text if c in string.punctuation)
        space_count = text.count(' ')
        
        # Ratios
        letter_ratio = letter_count / char_count if char_count > 0 else 0
        digit_ratio = digit_count / char_count if char_count > 0 else 0
        special_ratio = special_char_count / char_count if char_count > 0 else 0
        
        # Word-level features
        words = text.split()
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Vocabulary features
        clean_words = [re.sub(r'[^\w]', '', w) for w in words if w.strip()]
        valid_words = [w for w in clean_words if w in self.english_words]
        vocab_ratio = len(valid_words) / len(clean_words) if clean_words else 0
        
        # Repetition features
        char_repetition = self._calculate_repetition(text)
        word_repetition = self._calculate_word_repetition(words)
        
        # Randomness indicators
        consecutive_chars = self._count_consecutive_chars(text)
        keyboard_patterns = self._detect_keyboard_patterns(text)
        
        # Sentence structure
        sentence_count = len([s for s in text.split('.') if s.strip()])
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else word_count
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'letter_ratio': letter_ratio,
            'digit_ratio': digit_ratio,
            'special_ratio': special_ratio,
            'avg_word_length': avg_word_length,
            'vocab_ratio': vocab_ratio,
            'char_repetition': char_repetition,
            'word_repetition': word_repetition,
            'consecutive_chars': consecutive_chars,
            'keyboard_patterns': keyboard_patterns,
            'avg_sentence_length': avg_sentence_length,
            'sentence_count': sentence_count
        }
    
    def _empty_features(self) -> Dict[str, float]:
        """Return zero features for empty text."""
        return {
            'char_count': 0, 'word_count': 0, 'letter_ratio': 0,
            'digit_ratio': 0, 'special_ratio': 0, 'avg_word_length': 0,
            'vocab_ratio': 0, 'char_repetition': 0, 'word_repetition': 0,
            'consecutive_chars': 0, 'keyboard_patterns': 0,
            'avg_sentence_length': 0, 'sentence_count': 0
        }
    
    def _calculate_repetition(self, text: str) -> float:
        """Calculate character repetition score."""
        if len(text) <= 1:
            return 0
        
        char_counts = Counter(text)
        max_repetition = max(char_counts.values())
        return max_repetition / len(text)
    
    def _calculate_word_repetition(self, words: List[str]) -> float:
        """Calculate word repetition score."""
        if len(words) <= 1:
            return 0
        
        word_counts = Counter(words)
        max_repetition = max(word_counts.values())
        return max_repetition / len(words)
    
    def _count_consecutive_chars(self, text: str) -> float:
        """Count consecutive identical characters."""
        if len(text) <= 1:
            return 0
        
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        return max_consecutive / len(text)
    
    def _detect_keyboard_patterns(self, text: str) -> float:
        """Detect common keyboard patterns and random character sequences."""
        # Common keyboard patterns
        patterns = [
            'qwerty', 'asdf', 'zxcv', 'qwertyuiop', 'asdfghjkl', 'zxcvbnm',
            'abcdef', '123456', 'abcd', '1234', 'aaaaa', 'bbbbb', 'dfgh', 'fghj',
            'hjkl', 'tyui', 'yuio', 'uiop', 'sdfg', 'dfghjk', 'cvbnm', 'mnbv'
        ]
        
        text_lower = text.lower()
        pattern_score = 0
        
        # Check for exact pattern matches
        for pattern in patterns:
            if pattern in text_lower:
                pattern_score += len(pattern) / len(text) if len(text) > 0 else 0
        
        # Check for random character sequences (3+ chars with low vowel content)
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        random_sequences = 0
        
        for word in words:
            if len(word) >= 4:
                vowels = sum(1 for c in word if c in 'aeiou')
                consonant_clusters = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]{3,}', word))
                
                # Signs of random typing
                if vowels / len(word) < 0.2 or consonant_clusters > 0:
                    random_sequences += 1
        
        # Add random sequence penalty
        if words:
            random_ratio = random_sequences / len(words)
            pattern_score += random_ratio * 0.5
        
        return min(pattern_score, 1.0)  # Cap at 1.0
    
    def classify_text_quality(self, text: str) -> Dict[str, float]:
        """Use pre-trained model to assess text quality."""
        if not text or len(text.strip()) == 0:
            return {'confidence': 0, 'is_coherent': 0}
        
        try:
            # Use sentiment analysis as a proxy for text coherence
            # Coherent text typically produces confident sentiment predictions
            result = self.sentiment_classifier(text[:512])  # Limit to model's max length
            
            confidence = result[0]['score']
            
            # Higher confidence in sentiment often indicates more coherent text
            is_coherent = 1 if confidence > 0.7 else 0
            
            return {
                'confidence': confidence,
                'is_coherent': is_coherent
            }
        except Exception as e:
            print(f"Error in text quality classification: {e}")
            return {'confidence': 0, 'is_coherent': 0}
    
    def process_conversation(self, conversation_data: pd.DataFrame) -> Dict[str, any]:
        """Process a single conversation and extract all features."""
        # Combine all text from the conversation
        all_text = ""
        
        for _, row in conversation_data.iterrows():
            question = str(row['Question']) if pd.notna(row['Question']) else ""
            answer = str(row['Answer']) if pd.notna(row['Answer']) else ""
            
            # Split by ~ and add all parts
            question_parts = [part.strip() for part in question.split('~') if part.strip()]
            answer_parts = [part.strip() for part in answer.split('~') if part.strip()]
            
            all_text += " ".join(question_parts + answer_parts) + " "
        
        all_text = all_text.strip()
        
        # Extract features
        text_features = self.extract_text_features(all_text)
        quality_features = self.classify_text_quality(all_text)
        
        # Combine features
        features = {**text_features, **quality_features}
        features['full_text'] = all_text
        
        return features
    
    def classify_conversation(self, features: Dict[str, any]) -> Tuple[str, float]:
        """Classify a conversation as normal or random based on features."""
        # Enhanced rule-based classification with multiple criteria
        
        # Strong indicators of random/gibberish text
        random_indicators = []
        random_weights = []  # Weight each indicator by importance
        
        # 1. Very low vocabulary ratio (not real words) - CRITICAL INDICATOR
        if features['vocab_ratio'] < 0.5:  # Made more sensitive
            random_indicators.append('low_vocab')
            # Extra weight for very low vocab ratio
            weight = 3.0 if features['vocab_ratio'] < 0.2 else 2.0
            random_weights.append(weight)
        
        # 2. High character repetition
        if features['char_repetition'] > 0.25:  # Made more sensitive
            random_indicators.append('high_repetition')
            random_weights.append(1.5)
        
        # 3. High consecutive characters
        if features['consecutive_chars'] > 0.15:  # Made more sensitive
            random_indicators.append('consecutive_chars')
            random_weights.append(1.5)
        
        # 4. Keyboard patterns detected
        if features['keyboard_patterns'] > 0.005:  # Made more sensitive
            random_indicators.append('keyboard_patterns')
            random_weights.append(2.0)
        
        # 5. Very short words or very long words
        if features['avg_word_length'] < 2.5 or features['avg_word_length'] > 12:
            random_indicators.append('unusual_word_length')
            random_weights.append(1.0)
        
        # 6. Low model confidence (incoherent text)
        if features['confidence'] < 0.6:  # Made more sensitive
            random_indicators.append('low_confidence')
            random_weights.append(1.5)
        
        # 7. Very high special character ratio
        if features['special_ratio'] > 0.2:  # Made more sensitive
            random_indicators.append('high_special_chars')
            random_weights.append(1.0)
        
        # 8. Too few words for meaningful conversation
        if features['word_count'] < 3:
            random_indicators.append('too_few_words')
            random_weights.append(2.0)
        
        # 9. NEW: High ratio of very short meaningless words
        words = features['full_text'].split() if 'full_text' in features else []
        short_words = [w for w in words if len(w) <= 2]
        short_word_ratio = len(short_words) / len(words) if words else 0
        if short_word_ratio > 0.4:
            random_indicators.append('too_many_short_words')
            random_weights.append(1.5)
        
        # 10. NEW: Check for nonsense patterns (many non-words)
        import re
        if 'full_text' in features:
            words_clean = [re.sub(r'[^\w]', '', w).lower() for w in words if w.strip()]
            nonsense_words = []
            for word in words_clean:
                if len(word) > 2:  # Only check words longer than 2 chars
                    # Check if word has unusual character patterns
                    vowels = sum(1 for c in word if c in 'aeiou')
                    consonants = len(word) - vowels
                    # Very few vowels relative to length suggests nonsense
                    if len(word) > 4 and vowels / len(word) < 0.2:
                        nonsense_words.append(word)
            
            nonsense_ratio = len(nonsense_words) / len(words_clean) if words_clean else 0
            if nonsense_ratio > 0.3:
                random_indicators.append('nonsense_patterns')
                random_weights.append(2.5)
        
        # Calculate weighted random score
        if random_indicators:
            total_weight = sum(random_weights)
            max_possible_weight = 10.0  # Approximate max weight for normalization
            random_score = min(total_weight / max_possible_weight, 1.0)
        else:
            random_score = 0.0
        
        # More sensitive classification threshold
        if random_score >= 0.25:  # Lowered threshold for better detection
            return "random", random_score
        else:
            return "normal", 1 - random_score
    
    def analyze_dataset(self, csv_path: str) -> pd.DataFrame:
        """Analyze the entire dataset and return results."""
        print(f"📊 Loading dataset from: {csv_path}")
        
        # Load the CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
        
        # Group by conversation_id
        conversation_groups = df.groupby('conversation_id')
        print(f"Found {len(conversation_groups)} unique conversations")
        
        results = []
        
        print("🔍 Analyzing conversations...")
        for i, (conv_id, conv_data) in enumerate(conversation_groups):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(conversation_groups)} conversations processed")
            
            # Process the conversation
            features = self.process_conversation(conv_data)
            classification, confidence = self.classify_conversation(features)
            
            # Store results
            result = {
                'conversation_id': conv_id,
                'classification': classification,
                'confidence': confidence,
                'word_count': features['word_count'],
                'vocab_ratio': features['vocab_ratio'],
                'char_repetition': features['char_repetition'],
                'keyboard_patterns': features['keyboard_patterns'],
                'model_confidence': features['confidence'],
                'sample_text': features['full_text'][:200] + "..." if len(features['full_text']) > 200 else features['full_text']
            }
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        print(f"✅ Analysis complete!")
        
        return results_df

def main():
    """Main function to run the conversation classifier."""
    print("🚀 Starting Conversation Quality Classification")
    print("=" * 60)
    
    # Initialize classifier
    classifier = ConversationClassifier()
    
    # Analyze the dataset
    csv_path = "/home/vi/Llama_Training_With_LoRA/combined_human_conversations.csv"
    results_df = classifier.analyze_dataset(csv_path)
    
    # Display results summary
    print("\n📊 CLASSIFICATION RESULTS SUMMARY")
    print("=" * 60)
    
    classification_counts = results_df['classification'].value_counts()
    print(f"Total conversations analyzed: {len(results_df)}")
    print(f"Normal conversations: {classification_counts.get('normal', 0)} ({classification_counts.get('normal', 0)/len(results_df)*100:.1f}%)")
    print(f"Random/gibberish conversations: {classification_counts.get('random', 0)} ({classification_counts.get('random', 0)/len(results_df)*100:.1f}%)")
    
    # Show some examples
    print(f"\n🔍 EXAMPLES OF CLASSIFIED CONVERSATIONS")
    print("-" * 60)
    
    # Examples of random conversations
    random_examples = results_df[results_df['classification'] == 'random'].head(3)
    if not random_examples.empty:
        print("Random/Gibberish Examples:")
        for _, row in random_examples.iterrows():
            print(f"  ID: {row['conversation_id']}")
            print(f"  Text: {row['sample_text']}")
            print(f"  Confidence: {row['confidence']:.3f}")
            print()
    
    # Examples of normal conversations
    normal_examples = results_df[results_df['classification'] == 'normal'].head(3)
    if not normal_examples.empty:
        print("Normal Conversation Examples:")
        for _, row in normal_examples.iterrows():
            print(f"  ID: {row['conversation_id']}")
            print(f"  Text: {row['sample_text']}")
            print(f"  Confidence: {row['confidence']:.3f}")
            print()
    
    # Save results
    output_path = "/home/vi/Llama_Training_With_LoRA/conversation_classification_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"💾 Results saved to: {output_path}")
    
    # Additional statistics
    print(f"\n📈 DETAILED STATISTICS")
    print("-" * 60)
    print("Average metrics by classification:")
    print(results_df.groupby('classification')[['word_count', 'vocab_ratio', 'char_repetition', 'model_confidence']].mean().round(3))
    
    return results_df

if __name__ == "__main__":
    results = main()
