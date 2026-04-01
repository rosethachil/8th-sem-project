"""
run_pipeline.py
===============
Master orchestrator — runs the full SEED-IV preprocessing pipeline.

Steps:
  1. Structure inspection  (loader.py)
  2. CSV creation          (csv_exporter.py)
  3. Feature analysis      (analyzer.py)
  4. Feature selection     (analyzer.py)
  5. Report generation     (report_generator.py)

Usage:
  cd "C:/Users/Rose J Thachil/Documents/8th sem"
  python run_pipeline.py

All outputs go to:  8th sem/outputs/
"""

import sys
import os
import time

# Make sure pipeline/ is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))

from config import ensure_output_dirs
from loader import inspect_structure
from csv_exporter import create_csv, merge_multimodal
from analyzer import (
    analyze_features,
    plot_label_distribution,
    plot_correlation,
    run_feature_selection,
    summarize_gaps,
)
from report_generator import generate_markdown_report, generate_pdf_report


def main():
    t0 = time.time()

    print("\n" + "=" * 65)
    print("  SEED-IV Preprocessing Pipeline")
    print("  NO model training — pure data understanding & export")
    print("=" * 65)

    ensure_output_dirs()

    # ──────────────────────────────────────────
    # STEP 1: STRUCTURE INSPECTION
    # ──────────────────────────────────────────
    print("\n[STEP 1/5] Structure Inspection")
    print("-" * 40)
    inspect_structure(verbose=True)

    # ──────────────────────────────────────────
    # STEP 2: CSV CREATION
    # ──────────────────────────────────────────
    print("\n[STEP 2/5] CSV Creation (this may take a few minutes...)")
    print("-" * 40)
    df_de, df_psd, df_eye = create_csv(verbose=True)
    df_merged = merge_multimodal(df_de, df_eye)

    if df_de.empty:
        print("\n[ERROR] No EEG data loaded. Check EEG zip path in config.py")
        sys.exit(1)

    # ──────────────────────────────────────────
    # STEP 3: FEATURE ANALYSIS
    # ──────────────────────────────────────────
    print("\n[STEP 3/5] Feature Analysis")
    print("-" * 40)
    stats_df = analyze_features(df_de, modality="EEG (de_LDS)")
    plot_label_distribution(df_de, title="SEED-IV Label Distribution (EEG de_LDS)")
    plot_correlation(df_de, n_features=62, n_samples=5000)   # first 62 features

    # ──────────────────────────────────────────
    # STEP 4: FEATURE SELECTION
    # ──────────────────────────────────────────
    print("\n[STEP 4/5] Feature Selection (no model training)")
    print("-" * 40)
    selection_results = run_feature_selection(df_de, variance_threshold=1e-4, top_k=20)

    gaps = summarize_gaps(df_de, df_eye, df_merged)
    gaps["low_variance_count"] = len(selection_results.get("low_variance_features", []))

    # ──────────────────────────────────────────
    # STEP 5: REPORT GENERATION
    # ──────────────────────────────────────────
    print("\n[STEP 5/5] Generating Documentation")
    print("-" * 40)
    md_text = generate_markdown_report(
        stats_df        = stats_df,
        gaps            = gaps,
        selection_results = selection_results,
        df_eeg_shape    = df_de.shape if not df_de.empty else None,
        df_eye_shape    = df_eye.shape if not df_eye.empty else None,
        df_merged_shape = df_merged.shape if not df_merged.empty else None,
    )
    generate_pdf_report(md_text)

    # ──────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"\n  Elapsed time: {elapsed:.1f}s\n")
    print("  Output files:")
    print("    outputs/eeg_de_lds.csv")
    print("    outputs/eeg_psd_lds.csv")
    print("    outputs/eye_features.csv")
    print("    outputs/merged_multimodal.csv")
    print("    outputs/feature_stats.csv")
    print("    outputs/mutual_info_scores.csv")
    print("    outputs/plots/label_distribution.png")
    print("    outputs/plots/correlation_heatmap.png")
    print("    outputs/plots/mutual_info_top_features.png")
    print("    outputs/report/dataset_report.md")
    print("    outputs/report/dataset_report.pdf")
    print("\n  Ready for ML model training.")
    print("=" * 65)


if __name__ == "__main__":
    main()
