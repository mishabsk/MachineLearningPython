# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# ================================================================================
# LLM PREFERENCE PREDICTION - OPTIMIZED COMPETITION SOLUTION
# ================================================================================
# Install required libraries:
# pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost
# ================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import re
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ================================================================================
# ------------------ OPTIMIZED LLM PREFERENCE PREDICTOR CLASS --------------------
# ================================================================================

class OptimizedLLMPredictor:
    def __init__(self):
        """Initialize the predictor with optimized settings"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
        print("🚀 OptimizedLLMPredictor initialized!")
        print("=" * 60)
    
    # ============================================================================
    # ----------------------- FAST FEATURE ENGINEERING ---------------------------
    # ============================================================================
    
    def extract_fast_features(self, df, is_train=True):
        """Extract optimized features with focus on speed and performance"""
        
        print("🔧 Starting feature engineering...")
        features = df[['id']].copy()
        
        # ========================================================================
        # ------------------- Core Text Statistics (Vectorized) ------------------
        # ========================================================================
        
        for resp in ['response_a', 'response_b']:
            text_col = df[resp].astype(str)
            
            # Basic length metrics
            features[f'{resp}_len'] = text_col.str.len()
            features[f'{resp}_words'] = text_col.str.split().str.len()
            features[f'{resp}_sentences'] = text_col.str.count(r'[.!?]+')
            
            # Advanced metrics
            features[f'{resp}_avg_word_len'] = features[f'{resp}_len'] / (features[f'{resp}_words'] + 1)
            features[f'{resp}_punct_ratio'] = text_col.str.count(r'[^\w\s]') / (features[f'{resp}_len'] + 1)
            features[f'{resp}_upper_ratio'] = text_col.str.count(r'[A-Z]') / (features[f'{resp}_len'] + 1)
            
            # Structure indicators
            features[f'{resp}_newlines'] = text_col.str.count(r'\n')
            features[f'{resp}_code_blocks'] = text_col.str.count(r'```')
            features[f'{resp}_bullets'] = text_col.str.count(r'^\s*[-*•]\s', flags=re.MULTILINE)
            features[f'{resp}_numbers'] = text_col.str.count(r'^\s*\d+\.\s', flags=re.MULTILINE)
            
            # Quality indicators
            features[f'{resp}_questions'] = text_col.str.count(r'\?')
            features[f'{resp}_exclamations'] = text_col.str.count(r'!')
            
        print("✅ Basic text features extracted")
        
        # =====================================================================
        # ----------------------- Prompt Analysis -----------------------------
        # =====================================================================
        
        prompt_text = df['prompt'].astype(str)
        features['prompt_len'] = prompt_text.str.len()
        features['prompt_words'] = prompt_text.str.split().str.len()
        features['prompt_questions'] = prompt_text.str.count(r'\?')
        
        print("✅ Prompt features extracted")
        
        # ====================================================================
        # ----------- Comparative Features (Key for Performance) -------------
        # ====================================================================
        
        # Length comparisons
        features['len_ratio_a_b'] = features['response_a_len'] / (features['response_b_len'] + 1)
        features['len_diff_a_b'] = features['response_a_len'] - features['response_b_len']
        features['word_ratio_a_b'] = features['response_a_words'] / (features['response_b_words'] + 1)
        features['word_diff_a_b'] = features['response_a_words'] - features['response_b_words']
        
        # Quality comparisons  
        features['struct_diff_a_b'] = (features['response_a_bullets'] + features['response_a_numbers']) - \
                                      (features['response_b_bullets'] + features['response_b_numbers'])
        
        features['engagement_diff_a_b'] = (features['response_a_questions'] + features['response_a_exclamations']) - \
                                          (features['response_b_questions'] + features['response_b_exclamations'])
        
        print("✅ Comparative features extracted")
        
        # ===================================================================
        # ----- Model Performance Features (Consistent for Train/Test) ------
        # ===================================================================
        
        if is_train and 'model_a' in df.columns:
            # Fast model encoding
            for model_col in ['model_a', 'model_b']:
                if model_col not in self.label_encoders:
                    self.label_encoders[model_col] = LabelEncoder()
                    features[f'{model_col}_id'] = self.label_encoders[model_col].fit_transform(df[model_col])
                else:
                    features[f'{model_col}_id'] = self.label_encoders[model_col].transform(df[model_col])
            
            # Quick model stats
            model_wins = df.groupby('model_a')['winner_model_a'].mean()
            model_wins_b = df.groupby('model_b')['winner_model_b'].mean()
            
            features['model_a_win_rate'] = df['model_a'].map(model_wins).fillna(0.33)
            features['model_b_win_rate'] = df['model_b'].map(model_wins_b).fillna(0.33)
            
            # Store model stats for test prediction
            self.model_a_stats = model_wins.to_dict()
            self.model_b_stats = model_wins_b.to_dict()
            
            print("✅ Model features extracted")
        
        elif not is_train:
            # For test data, create dummy model features to maintain consistency
            features['model_a_id'] = 0  # Default encoding for unknown models
            features['model_b_id'] = 0  # Default encoding for unknown models
            features['model_a_win_rate'] = 0.33  # Default win rate
            features['model_b_win_rate'] = 0.33  # Default win rate
            
            print("✅ Dummy model features added for test consistency")
        
        # ==============================================================
        # ------------ Text Similarity (Fast Version) ------------------
        # ==============================================================
        
        def fast_word_overlap(row):
            words_a = set(str(row['response_a']).lower().split())
            words_b = set(str(row['response_b']).lower().split())
            if len(words_a) == 0 or len(words_b) == 0:
                return 0
            return len(words_a & words_b) / len(words_a | words_b)
        
        features['word_overlap'] = df.apply(fast_word_overlap, axis=1)
        
        print("✅ Similarity features extracted")
        
        # Fill missing values and return
        features = features.fillna(0)
        self.feature_names = [col for col in features.columns if col != 'id']
        
        print(f"🔥 Feature engineering completed! Total features: {len(self.feature_names)}")
        print("=" * 60)
        
        return features
    
    # ============================================================================
    # -------------------- OPTIMIZED MODEL TRAINING ------------------------------
    # ============================================================================
    
    def train_optimized_ensemble(self, X, y):
        """Train optimized ensemble with focus on speed and performance"""
        
        print("🤖 Training optimized ensemble models...")
        print("=" * 60)
        
        # Convert to multiclass format
        y_multiclass = np.argmax(y.values, axis=1)
        
        # ========================================
        # Model 1: LightGBM (Primary Model)
        # ========================================
        print("🚀 Training LightGBM (Primary Model)...")
        self.models['lgb'] = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.08,
            max_depth=6,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective='multiclass',
            num_class=3,
            verbose=-1,
            force_col_wise=True  # Optimization for speed
        )
        self.models['lgb'].fit(X, y_multiclass)
        print("✅ LightGBM training completed!")
        
        # ========================================
        # Model 2: XGBoost (Secondary Model)
        # ========================================
        print("🚀 Training XGBoost...")
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=600,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective='multi:softprob',
            eval_metric='mlogloss',
            verbosity=0,
            tree_method='hist'  # Faster training
        )
        self.models['xgb'].fit(X, y_multiclass)
        print("✅ XGBoost training completed!")
        
        # ========================================
        # Model 3: CatBoost (Robust Model)
        # ========================================
        print("🚀 Training CatBoost...")
        self.models['catboost'] = CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=False,
            loss_function='MultiClass',
            task_type='CPU'  # Explicit CPU usage
        )
        self.models['catboost'].fit(X, y_multiclass)
        print("✅ CatBoost training completed!")
        
        # ========================================  
        # Ensemble Weights (Optimized)
        # ========================================
        self.ensemble_weights = {
            'lgb': 0.5,      # Primary model
            'xgb': 0.35,     # Strong secondary
            'catboost': 0.15 # Robustness
        }
        
        print("🎯 Ensemble training completed!")
        print("=" * 60)
    
    # ============================================================================
    # --------------------- PREDICTION AND VALIDATION ----------------------------
    # ============================================================================
    
    def predict_optimized(self, X):
        """Make optimized ensemble predictions with feature validation"""
        
        print("🔮 Making ensemble predictions...")
        
        # Validate feature consistency
        if hasattr(self, 'feature_names'):
            if X.shape[1] != len(self.feature_names):
                print(f"⚠️ Feature mismatch detected!")
                print(f"Expected: {len(self.feature_names)} features")
                print(f"Received: {X.shape[1]} features")
                
                # Ensure X has the same columns as training
                if isinstance(X, pd.DataFrame):
                    missing_features = set(self.feature_names) - set(X.columns)
                    extra_features = set(X.columns) - set(self.feature_names)
                    
                    if missing_features:
                        print(f"Adding missing features: {missing_features}")
                        for feature in missing_features:
                            X[feature] = 0  # Default value for missing features
                    
                    if extra_features:
                        print(f"Removing extra features: {extra_features}")
                        X = X.drop(columns=list(extra_features))
                    
                    # Reorder columns to match training
                    X = X[self.feature_names]
                
        print(f"✅ Feature validation completed. Shape: {X.shape}")
        
        predictions = np.zeros((X.shape[0], 3))
        
        for name, model in self.models.items():
            pred = model.predict_proba(X)
            predictions += self.ensemble_weights[name] * pred
            print(f"✅ {name} predictions added (weight: {self.ensemble_weights[name]})")
        
        print("🎯 Ensemble predictions completed!")
        print("=" * 60)
        return predictions
    
    def quick_validation(self, X, y, n_splits=3):
        """Quick cross-validation for performance check"""
        
        print("📊 Running quick validation...")
        print("=" * 60)
        
        y_multiclass = np.argmax(y.values, axis=1)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Only validate primary model for speed
        model = self.models['lgb']
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_multiclass)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y_multiclass[train_idx], y_multiclass[val_idx]
            
            # Quick training
            fold_model = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6, 
                random_state=42, verbose=-1, objective='multiclass', num_class=3
            )
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Prediction and scoring
            pred_proba = fold_model.predict_proba(X_fold_val)
            y_val_onehot = np.eye(3)[y_fold_val]
            score = log_loss(y_val_onehot, pred_proba)
            scores.append(score)
            
            print(f"📈 Fold {fold+1} Log Loss: {score:.4f}")
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"🎯 Average CV Score: {avg_score:.4f} (+/- {std_score:.4f})")
        print("=" * 60)
        
        return avg_score

# ================================================================================
# ------------------------ MAIN EXECUTION PIPELINE -------------------------------
# ================================================================================

def main():
    """Optimized main pipeline for maximum performance"""
    
    print("🏁 STARTING LLM PREFERENCE PREDICTION PIPELINE")
    print("=" * 80)
    
    # ============================================================================
    # ------------------------------ DATA LOADING --------------------------------
    # ============================================================================
    
    print("📂 Loading competition data...")
    train_df = pd.read_csv('/kaggle/input/llm-classification-finetuning/train.csv')
    test_df = pd.read_csv('/kaggle/input/llm-classification-finetuning/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/llm-classification-finetuning/sample_submission.csv')
    
    print(f"📊 Train shape: {train_df.shape}")
    print(f"📊 Test shape: {test_df.shape}")
    print("=" * 60)
    
    # ============================================================================
    # --------------------- MODEL INITIALIZATION --------------------------------
    # ============================================================================
    
    predictor = OptimizedLLMPredictor()
    
    # ============================================================================
    # ------------------------- FEATURE ENGINEERING ------------------------------
    # ============================================================================
    
    print("⚡ Processing training features...")
    train_features = predictor.extract_fast_features(train_df, is_train=True)
    X_train = train_features.drop(['id'], axis=1)
    y_train = train_df[['winner_model_a', 'winner_model_b', 'winner_tie']].copy()
    
    # Store feature names for consistency
    predictor.feature_names = list(X_train.columns)
    
    print(f"🔢 Final training shape: {X_train.shape}")
    print(f"📋 Feature names stored: {len(predictor.feature_names)} features")
    print("=" * 60)
    
    # ============================================================================
    # -------------------------- MODEL TRAINING ----------------------------------
    # ============================================================================
    
    predictor.train_optimized_ensemble(X_train, y_train)
    
    # ============================================================================
    # ---------------------------- QUICK VALIDATION ------------------------------
    # ============================================================================
    
    try:
        cv_score = predictor.quick_validation(X_train, y_train)
        performance_indicator = "🔥 EXCELLENT" if cv_score < 1.05 else "✅ GOOD" if cv_score < 1.10 else "⚠️ NEEDS IMPROVEMENT"
        print(f"🎯 Model Performance: {performance_indicator}")
    except Exception as e:
        print(f"⚠️ Validation skipped: {str(e)}")
    
    print("=" * 60)
    
    # ============================================================================
    # ----------------------- TEST PREDICTIONS -----------------------------------
    # ============================================================================
    
    print("🔮 Processing test data...")
    test_features = predictor.extract_fast_features(test_df, is_train=False)
    X_test = test_features.drop(['id'], axis=1)
    
    print(f"📊 Test features shape: {X_test.shape}")
    print(f"📋 Expected shape: ({X_test.shape[0]}, {len(predictor.feature_names)})")
    
    # Make predictions with feature validation
    predictions = predictor.predict_optimized(X_test)
    
    # ============================================================================
    # ------------------------ SUBMISSION CREATION ------------------------------
    # ============================================================================
    
    print("📝 Creating optimized submission...")
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'winner_model_a': predictions[:, 0],
        'winner_model_b': predictions[:, 1], 
        'winner_tie': predictions[:, 2]
    })
    
    # Normalize probabilities to ensure they sum to 1
    prob_sums = submission[['winner_model_a', 'winner_model_b', 'winner_tie']].sum(axis=1)
    submission[['winner_model_a', 'winner_model_b', 'winner_tie']] = \
        submission[['winner_model_a', 'winner_model_b', 'winner_tie']].div(prob_sums, axis=0)
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    
    print("✅ Submission saved as 'submission.csv'")
    print("=" * 60)
    
    # ============================================================================
    # --------------------- FINAL VALIDATION AND SUMMARY ------------------------
    # ============================================================================
    
    print("📋 SUBMISSION SUMMARY")
    print("=" * 60)
    print(f"📊 Shape: {submission.shape}")
    print(f"📋 Columns: {list(submission.columns)}")
    
    # Check probability distributions
    prob_stats = submission[['winner_model_a', 'winner_model_b', 'winner_tie']].describe()
    print("\n📈 Probability Distributions:")
    print(prob_stats.round(4))
    
    print("\n🎯 Sample Predictions:")
    print(submission.head().round(4))
    
    # Final validation
    prob_sums_check = submission[['winner_model_a', 'winner_model_b', 'winner_tie']].sum(axis=1)
    print(f"\n✅ Probability sums (should be ~1.0): Min={prob_sums_check.min():.6f}, Max={prob_sums_check.max():.6f}")
    
    print("=" * 80)
    print("🏆 PIPELINE COMPLETED SUCCESSFULLY!")
    print("🚀 Ready for submission to leaderboard!")
    print("=" * 80)
    
    return predictor, submission

# ================================================================================
# ------------------------------- EXECUTION --------------------------------------
# ================================================================================

if __name__ == "__main__":
    predictor, submission = main()
