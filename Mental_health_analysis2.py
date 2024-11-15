from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime, timedelta


class EnhancedMentalHealthAnalyzer:
    def __init__(self):
        self.create_sample_data()
        self.vectorizer = TfidfVectorizer(
            max_features=1500,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams for better context
        )
        self.classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42
        )
        self.emotion_intensity = {
            'high': 3,
            'medium': 2,
            'low': 1
        }

    def create_sample_data(self):
        """Create an enhanced sample dataset with more nuanced categories"""
        # Extended dataset with intensity levels
        texts = [
            ("I feel absolutely incredible today, everything is perfect!", "joy", "high"),
            ("Feeling a bit down but managing okay", "sadness", "low"),
            ("My anxiety is through the roof, can't handle this", "anxiety", "high"),
            ("Making steady progress on my goals", "joy", "medium"),
            ("Haven't slept properly in days, everything feels overwhelming", "anxiety", "high"),
            ("Really grateful for my support system", "joy", "medium"),
            ("Just feeling a bit confused about life lately", "confusion", "low"),
            ("Deep sense of peace after meditation", "joy", "medium"),
            ("Constant worry about the future", "anxiety", "medium"),
            ("Had a breakthrough in therapy today!", "joy", "high"),
            ("No motivation to do anything anymore", "sadness", "high"),
            ("Small wins adding up, feeling proud", "joy", "medium"),
            ("Work stress is getting to me", "anxiety", "medium"),
            ("Accomplished something difficult today", "joy", "high"),
            ("Can't seem to shake this sadness", "sadness", "high"),
            ("Starting a new exciting project", "joy", "high"),
            ("Feeling very isolated and alone", "sadness", "high"),
            ("Making positive changes in my life", "joy", "medium"),
            ("Experiencing frequent panic attacks", "anxiety", "high"),
            ("Finding joy in simple moments", "joy", "low")
        ]

        # Create DataFrame with additional features
        self.data = pd.DataFrame(texts, columns=['text', 'emotion', 'intensity'])

        # Add timestamp for temporal analysis
        dates = pd.date_range(end=datetime.now(), periods=len(texts), freq='D')
        self.data['timestamp'] = dates

        # Add word count feature
        self.data['word_count'] = self.data['text'].apply(lambda x: len(x.split()))

    def extract_features(self, text):
        """Extract advanced linguistic features from text"""
        features = {
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text),
            'contains_negation': any(word in text.lower() for word in ['no', 'not', "n't", 'never'])
        }
        return features

    def analyze_text(self, text):
        """Enhanced emotional content analysis"""
        emotion_patterns = {
            'anxiety': {
                'keywords': ['anxious', 'worried', 'nervous', 'stress', 'panic'],
                'intensity_modifiers': ['very', 'extremely', 'somewhat', 'slightly']
            },
            'sadness': {
                'keywords': ['sad', 'lonely', 'unmotivated', 'tired', 'disconnected'],
                'intensity_modifiers': ['deeply', 'overwhelmingly', 'a bit', 'mildly']
            },
            'joy': {
                'keywords': ['happy', 'joy', 'grateful', 'excited', 'confident'],
                'intensity_modifiers': ['very', 'incredibly', 'somewhat', 'fairly']
            },
            'confusion': {
                'keywords': ['confused', 'unsure', 'uncertain', 'lost', 'wondering'],
                'intensity_modifiers': ['completely', 'totally', 'slightly', 'partially']
            }
        }

        text = text.lower()
        analysis_results = []

        for emotion, patterns in emotion_patterns.items():
            # Check for emotion keywords
            emotion_present = any(keyword in text for keyword in patterns['keywords'])
            if emotion_present:
                # Determine intensity
                intensity = 'medium'  # default
                for modifier in patterns['intensity_modifiers']:
                    if modifier in text:
                        if modifier in ['very', 'extremely', 'deeply', 'overwhelmingly', 'completely', 'totally']:
                            intensity = 'high'
                        elif modifier in ['somewhat', 'slightly', 'a bit', 'mildly', 'partially']:
                            intensity = 'low'
                analysis_results.append({
                    'emotion': emotion,
                    'intensity': intensity
                })

        return analysis_results if analysis_results else [{'emotion': 'neutral', 'intensity': 'low'}]

    def train_model(self):
        """Train an enhanced sentiment classifier"""
        # Prepare features
        X = self.vectorizer.fit_transform(self.data['text'])
        y = self.data['emotion']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train classifier
        self.classifier.fit(X_train, y_train)

        # Generate detailed performance metrics
        y_pred = self.classifier.predict(X_test)

        return {
            'train_score': self.classifier.score(X_train, y_train),
            'test_score': self.classifier.score(X_test, y_test),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test
        }

    def predict_emotion(self, text):
        """Enhanced emotion prediction"""
        text_vectorized = self.vectorizer.transform([text])
        prediction = self.classifier.predict(text_vectorized)[0]
        probabilities = self.classifier.predict_proba(text_vectorized)[0]

        # Get detailed analysis
        detailed_analysis = self.analyze_text(text)
        linguistic_features = self.extract_features(text)

        return {
            'predicted_emotion': prediction,
            'confidence': max(probabilities),
            'detailed_analysis': detailed_analysis,
            'linguistic_features': linguistic_features,
            'all_probabilities': dict(zip(self.classifier.classes_, probabilities))
        }

    def visualize_results(self):
        """Create enhanced visualizations"""
        fig = plt.figure(figsize=(20, 12))

        # 1. Emotion Distribution
        ax1 = plt.subplot(221)
        emotion_counts = Counter(self.data['emotion'])
        # sns.barplot(x=list(emotion_counts.keys()),
        #             y=list(emotion_counts.values()),
        #             palette='viridis',
        #             ax=ax1)
        sns.barplot(x=list(emotion_counts.keys()), y=list(emotion_counts.values()), hue=list(emotion_counts.keys()),
                    ax=ax1)

        ax1.set_title('Distribution of Emotions')
        ax1.set_ylabel('Count')
        plt.xticks(rotation=45)

        # 2. Emotion Intensity Heatmap
        ax2 = plt.subplot(222)
        intensity_matrix = pd.crosstab(self.data['emotion'], self.data['intensity'])
        sns.heatmap(intensity_matrix, annot=True, cmap='YlOrRd', ax=ax2)
        ax2.set_title('Emotion Intensity Distribution')

        # 3. Word Count Distribution by Emotion
        ax3 = plt.subplot(223)
        sns.boxplot(x='emotion', y='word_count', data=self.data, palette='Set3', ax=ax3)
        ax3.set_title('Word Count Distribution by Emotion')
        plt.xticks(rotation=45)

        # 4. Temporal Emotion Trends
        ax4 = plt.subplot(224)
        # emotion_counts_over_time = self.data.set_index('timestamp')['emotion'].rolling('3D').value_counts()
        # Group by emotion and get rolling counts
        # self.data['rolling_emotion'] = self.data.set_index('timestamp')['emotion'].rolling('3D').apply(
        #     lambda x: Counter(x).most_common(1)[0][0] if len(x) > 0 else None, raw=False)
        from collections import Counter

        # Apply rolling window with a custom function to count the most common emotion
        self.data['rolling_emotion'] = self.data.set_index('timestamp')['emotion'].rolling('3D').apply(
            lambda x: Counter(x).most_common(1)[0][0] if len(x) > 0 else None, raw=False)
        for emotion in self.data['emotion'].unique():
            if emotion in emotion_counts_over_time.index.get_level_values(1):
                emotion_data = emotion_counts_over_time.xs(emotion, level=1)
                ax4.plot(emotion_data.index, emotion_data.values, label=emotion, marker='o')
        ax4.set_title('Temporal Emotion Trends')
        ax4.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        return fig
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Create and train the enhanced analyzer
analyzer = EnhancedMentalHealthAnalyzer()
training_results = analyzer.train_model()

# Print detailed model performance
print("\nModel Performance:")
print(f"Training Accuracy: {training_results['train_score']:.2f}")
print(f"Testing Accuracy: {training_results['test_score']:.2f}")
print("\nClassification Report:")
print(training_results['classification_report'])

# Example predictions with enhanced analysis
test_texts = [
    "I'm feeling extremely anxious and overwhelmed about my upcoming presentation",
    "Today was absolutely amazing, everything went perfectly!",
    "Feeling somewhat down but trying to stay positive",
]

print("\nDetailed Predictions:")
for text in test_texts:
    result = analyzer.predict_emotion(text)
    print(f"\nText: {text}")
    print(f"Predicted Emotion: {result['predicted_emotion']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("Emotion Probabilities:", result['all_probabilities'])
    print("Detailed Analysis:", result['detailed_analysis'])
    print("Linguistic Features:", result['linguistic_features'])

# Create and show enhanced visualizations
analyzer.visualize_results()
plt.show()