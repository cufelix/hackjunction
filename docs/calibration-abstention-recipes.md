# Calibration, abstention and FP-rate intervals: recipes (research only, nothing built)

Source: subagent web research 2026-11-13. Items marked UNVERIFIED were snippet-only or from memory; check against installed versions before use. Written for Tim's eval plan.

## Calibration
1. Split by time or session group: train, calibrate, test. Never calibrate on training rows.
2. Fit the classifier (XGBoost or LightGBM) on train.
3. `CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid").fit(X_cal, y_cal)`; import `from sklearn.frozen import FrozenEstimator` (replaces `cv="prefit"`).
4. Use `sigmoid` (Platt) under about 1000 calibration rows or with few positives; `isotonic` only above that (scikit-learn docs say isotonic overfits small sets).
5. Report `brier_score_loss` before and after and plot `calibration_curve(y, p, n_bins=10, strategy="quantile")` (signatures from memory, UNVERIFIED).
6. ECE by hand: bin predictions, count-weighted mean of |accuracy - confidence| (Guo et al.).
Sources: scikit-learn.org/stable/modules/calibration.html, arxiv.org/abs/1706.04599.

## Abstention (Chow rule, selective classification)
1. With calibrated bot probability p: act only if max(p, 1-p) >= tau, otherwise "unsure".
2. Sweep tau on validation data; for each tau compute coverage (share decided alone) and selective risk (error among decided). Plot risk vs coverage.
3. Choose tau from a target, for example "human false-positive rate <= 1% at the highest coverage". Geifman and El-Yaniv give a risk bound with probability 1-delta via binomial tails.
Sources: ar5iv.arxiv.org/html/1705.08500, proceedings.mlr.press/v97/geifman19a. Chow 1970 not retrieved.

## Conformal prediction
- `crepes`: `WrapClassifier(model)`, `.fit(X_tr, y_tr)`, `.calibrate(X_cal, y_cal, class_cond=True)`, `.predict_set(X_te, confidence=0.95)`. Class-conditional (Mondrian) matters for a tiny bot class. Keyword differs by version (`mc=` older, `bins=` newer).
- `MAPIE v1`: `SplitConformalClassifier(estimator, confidence_level=0.95, prefit=True).conformalize(X_c, y_c)`, `.predict_set(X)`. Its binary tutorial warns against set prediction for binary tasks (UNVERIFIED).
- Reading sets: {bot} act as bot, {human} act as human, {bot, human} or empty = unsure bucket.
- Guarantee is marginal and needs exchangeability; it breaks under day-to-day shift.
Sources: crepes.readthedocs.io, mapie.readthedocs.io.

## FP rate interval with few negatives
- `statsmodels.stats.proportion.proportion_confint(k, n, alpha=0.05, method="wilson")`; `method="beta"` for Clopper-Pearson.
- k = 0: rule of three, upper bound about 3/n (exact 1 - 0.05^(1/n)). 0 false positives in 100 humans means an upper bound near 3%, not 0%.

## Prior art (snippet-only, UNVERIFIED)
- arXiv 2503.23629 conformal prediction for social-media bots (tweets, not web sessions).
- arXiv 2607.27143 cost-sensitive conformal and human-in-the-loop abstention on imbalanced tabular sets including fraud.
- MDPI Make 8:190 class-conditional conformal anomaly detection.
- Nothing found on conformal prediction for web-session bot detection: say "we found none", not "none exists".

## Pitfalls
1. Calibrating on training rows (GBMs are overconfident there).
2. Random splits leak one visitor across train and calibration; split by day or IP/session group.
3. Shift between days: calibrate on the most recent day and show it holds on a later one.
4. Isotonic with a small calibration set creates steps and ties.
5. ECE is noisy with few positives; show the reliability diagram with per-bin counts plus Brier.
6. Conformal on a tiny bot class gives wide sets; use class-conditional and state the positive count.
7. "0 false positives" is not 0%; always show the interval.

## What to show on screen
1. Reliability diagram before and after calibration, per-bin counts, Brier and ECE.
2. Risk-coverage chart: x "share of sessions decided alone", y "mistakes among those decisions", operating point marked.
3. One plain sentence: "At 85% automated, human mistakes drop to 1.2%. The other 15% go to review."
4. A bot / human / unsure outcome bar for the test set.
5. FP rate with Wilson interval and n, for example "0.8% (95% CI 0.1 to 4.5%, n=120 humans)".
6. A later-day panel showing the plots still hold or honestly where they drift.
