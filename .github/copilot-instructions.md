# Project rules for Copilot (ME131 student project)

We are building a small Streamlit app that shows the results of a university
machine learning project. Follow these rules in every answer.

## Style
- Use simple, short English in all labels, titles and comments (some of us are not native speakers).
- Add a short comment above every function and every block of code that says what it does and why.
- Keep the code simple and easy to read. No clever tricks. No unused code.

## Data rules
- The app only READS files from these folders: results/, data_out/, models/, demo_data/.
- Never type result numbers into the code. Every number comes from a file.
- Never invent data. If a file is missing, show a clear message and stop that page (st.stop()).
- results/run_status.json says which files are real. If a file is not marked true,
  show a yellow warning at the top of the page: "TEST DATA. These are not our results."

## Names (use exactly these)
- Models: "logreg" (Logistic regression), "rf" (Random forest), "svm" (SVM, support vector machine)
- Groups: "accel_only" (Accelerometer only), "gyro_only" (Gyroscope only), "both" (Both sensors, all 561 features),
  "time_only" (Time domain only), "freq_only" (Frequency domain only)
- Activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
- Accuracy is saved as a fraction (0.92). Show it as a percent (92.0%).
- gap_points is already in percentage points.

## Files and columns
- results/results_summary.csv: model, group, n_features, official_acc, kfold_acc_mean, kfold_acc_std, loso_acc_mean, loso_acc_std, gap_points
- results/loso_by_volunteer.csv: model, group, volunteer, n_samples, accuracy
- results/confusion_matrices.csv: model, group, split (kfold or loso), true_activity, predicted_activity, count
- results/per_class_metrics.csv: model, group, split, activity, precision, recall, f1, support
- results/sensor_tiers.csv (may not exist yet; then the page shows "coming soon"): tier, name, group, kfold_acc, loso_acc, power_value, power_unit, power_mode, datasheet_ref, note
- results/run_status.json: {"real": {"<file name>": true or false, "models": true or false}}
- data_out/feature_groups.csv: col_index, name, sensor, domain, in_accel_only, in_gyro_only, in_both, in_time_only, in_freq_only
- models/demo_{model}_{group}.joblib: a scikit-learn Pipeline trained only on that group's columns
- demo_data/X_test_demo.csv.gz (561 columns) and demo_data/meta_test_demo.csv (row_index, volunteer, true_activity)

## Streamlit rules
- Use st.cache_data for reading CSV files and st.cache_resource for loading models.
- Use plotly for charts. Use blue (#5B9BFF) and yellow (#F2B705) as the main colors. Use large text and labeled axes.
- Never crash. Handle missing files and empty selections with a friendly message.
- No internet calls. No API keys. No tracking.
