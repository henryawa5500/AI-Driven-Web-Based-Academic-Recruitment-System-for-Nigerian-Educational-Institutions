import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap
import lime
import lime.lime_tabular
import re
import json

class BiasDetector:
    def __init__(self):
        """Initialize the BiasDetector."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.explainer = None
        self.feature_names = None
    
    def extract_demographic_features(self, text):
        """
        Extract potential demographic features from text.
        Note: This is a simplified implementation. In practice, you would need
        more sophisticated methods and careful handling of sensitive information.
        """
        features = {}
        
        # Gender indicators (simplified example)
        male_indicators = ['he', 'him', 'his', 'mr.', 'mr', 'sir']
        female_indicators = ['she', 'her', 'hers', 'ms.', 'ms', 'mrs.', 'mrs', 'madam']
        
        text_lower = text.lower()
        features['male_indicators'] = sum(text_lower.count(indicator) for indicator in male_indicators)
        features['female_indicators'] = sum(text_lower.count(indicator) for indicator in female_indicators)
        
        # Name patterns (simplified example)
        # Note: This is just for demonstration. In practice, you would need
        # a more sophisticated approach and proper handling of names.
        name_patterns = {
            'western_name': r'\b(?:john|michael|david|james|robert|mary|elizabeth|sarah|jennifer|lisa)\b',
            'nigerian_name': r'\b(?:chukwu|chukwudi|chukwuma|chukwunonso|chukwuebuka|'
                            r'chukwuka|chukwudi|chukwuma|chukwunonso|chukwuebuka|'
                            r'chukwuka|chukwudi|chukwuma|chukwunonso|chukwuebuka)\b'
        }
        
        for pattern_name, pattern in name_patterns.items():
            features[pattern_name] = len(re.findall(pattern, text_lower, re.IGNORECASE))
        
        return features
    
    def prepare_features(self, resumes_df, jobs_df, matches_df):
        """
        Prepare features for bias detection.
        
        Args:
            resumes_df: DataFrame of parsed resumes
            jobs_df: DataFrame of parsed jobs
            matches_df: DataFrame of job-resume matches with scores
        
        Returns:
            DataFrame: Features for bias detection
        """
        features = []
        
        for _, match in matches_df.iterrows():
            resume_id = match['resume_id']
            job_id = match['job_id']
            score = match['similarity_score']
            
            resume_row = resumes_df[resumes_df['id'] == resume_id].iloc[0]
            job_row = jobs_df[jobs_df['id'] == job_id].iloc[0]
            
            # Extract features from resume
            resume_features = self.extract_demographic_features(
                self._combine_resume_text(resume_row)
            )
            
            # Extract features from job
            job_features = self.extract_demographic_features(
                self._combine_job_text(job_row)
            )
            
            # Combine features
            combined_features = {
                **resume_features,
                **job_features,
                'similarity_score': score
            }
            
            features.append(combined_features)
        
        return pd.DataFrame(features)
    
    def _combine_resume_text(self, resume_row):
        """Combine resume text fields for analysis."""
        text_parts = [
            resume_row.get('name', ''),
            resume_row.get('education', ''),
            resume_row.get('skills', ''),
            resume_row.get('experience', ''),
            resume_row.get('publications', '')
        ]
        return ' '.join(str(part) for part in text_parts)
    
    def _combine_job_text(self, job_row):
        """Combine job text fields for analysis."""
        text_parts = [
            job_row.get('title', ''),
            job_row.get('institution', ''),
            job_row.get('qualifications', ''),
            job_row.get('responsibilities', ''),
            job_row.get('experience', '')
        ]
        return ' '.join(str(part) for part in text_parts)
    
    def train_bias_model(self, features_df, target_column='similarity_score', threshold=0.7):
        """
        Train a model to detect potential biases.
        
        Args:
            features_df: DataFrame of features
            target_column: Column to use as target
            threshold: Threshold for high/low similarity classification
        """
        # Prepare target variable
        y = (features_df[target_column] >= threshold).astype(int)
        X = features_df.drop(columns=[target_column])
        
        # Store feature names for explanation
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Initialize explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=self.feature_names,
            class_names=['low_similarity', 'high_similarity'],
            mode='classification'
        )
        
        # Print model performance
        y_pred = self.model.predict(X_test)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
    
    def explain_prediction(self, features_df, instance_idx):
        """
        Explain a specific prediction using LIME.
        
        Args:
            features_df: DataFrame of features
            instance_idx: Index of the instance to explain
        
        Returns:
            dict: Explanation of the prediction
        """
        if self.explainer is None:
            raise ValueError("Model must be trained before generating explanations")
        
        instance = features_df.iloc[instance_idx]
        X = instance.drop('similarity_score')
        
        # Generate explanation
        exp = self.explainer.explain_instance(
            X.values,
            self.model.predict_proba,
            num_features=len(self.feature_names)
        )
        
        # Format explanation
        explanation = {
            'prediction': self.model.predict([X])[0],
            'probability': self.model.predict_proba([X])[0].tolist(),
            'features': {}
        }
        
        for feature, weight in exp.as_list():
            explanation['features'][feature] = weight
        
        return explanation
    
    def detect_bias_patterns(self, features_df):
        """
        Detect potential bias patterns in the matching process.
        
        Args:
            features_df: DataFrame of features
        
        Returns:
            dict: Bias analysis results
        """
        # Calculate feature importance using SHAP
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(features_df.drop(columns=['similarity_score']))
        
        # Analyze feature importance
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            importance = np.abs(shap_values[1][:, i]).mean()
            feature_importance[feature] = importance
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Identify potential bias indicators
        bias_indicators = []
        for feature, importance in sorted_features:
            if importance > 0.1:  # Threshold for significant impact
                # Check correlation with similarity score
                correlation = features_df[feature].corr(features_df['similarity_score'])
                if abs(correlation) > 0.3:  # Threshold for strong correlation
                    bias_indicators.append({
                        'feature': feature,
                        'importance': importance,
                        'correlation': correlation
                    })
        
        return {
            'feature_importance': dict(sorted_features),
            'bias_indicators': bias_indicators
        }

if __name__ == "__main__":
    # Example usage
    detector = BiasDetector()
    
    # Load data
    resumes_df = pd.read_csv('data/processed/parsed_resumes.csv')
    jobs_df = pd.read_csv('data/processed/parsed_job_listings.csv')
    matches_df = pd.read_csv('data/processed/matches.csv')  # Assuming this exists
    
    # Prepare features
    features_df = detector.prepare_features(resumes_df, jobs_df, matches_df)
    
    # Train model
    detector.train_bias_model(features_df)
    
    # Detect bias patterns
    bias_analysis = detector.detect_bias_patterns(features_df)
    print("\nBias Analysis Results:")
    print("Feature Importance:")
    for feature, importance in bias_analysis['feature_importance'].items():
        print(f"{feature}: {importance:.4f}")
    
    print("\nPotential Bias Indicators:")
    for indicator in bias_analysis['bias_indicators']:
        print(f"Feature: {indicator['feature']}")
        print(f"Importance: {indicator['importance']:.4f}")
        print(f"Correlation: {indicator['correlation']:.4f}")
        print() 