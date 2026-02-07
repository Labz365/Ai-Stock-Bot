import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

# Columns the model should NOT use as features
drop_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
             'ma_10', 'ma_50', 'volume_ma_10', 'volume_ma_50', 'target']

for ticker in tickers:
    print(f"{'='*50}")
    print(f"--- {ticker} ---")
    print(f"{'='*50}")
    df = pd.read_csv(f'data/{ticker}_features.csv')

    # Separate features and target
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df['target']

    # Time-based split (80% train, 20% test) — no shuffling
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- Train Random Forest ---
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

    # --- Train Gradient Boosting ---
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb_model.predict(X_test))

    # --- Pick the best ---
    if gb_acc > rf_acc:
        best_model = gb_model
        best_name = 'Gradient Boosting'
        best_acc = gb_acc
    else:
        best_model = rf_model
        best_name = 'Random Forest'
        best_acc = rf_acc

    print(f"\nRandom Forest accuracy:    {rf_acc:.4f}")
    print(f"Gradient Boosting accuracy: {gb_acc:.4f}")
    print(f">>> Using: {best_name} ({best_acc:.4f})\n")

    # Evaluate best model
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))

    # Feature importance (top 10)
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    print("Top 10 features:")
    print(importances.sort_values(ascending=False).head(10))

    # Save best model
    joblib.dump(best_model, f'models/{ticker}.pkl')
    print(f"\nSaved models/{ticker}.pkl ({best_name})\n")
