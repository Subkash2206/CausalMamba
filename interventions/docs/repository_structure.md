# Repository Structure

```
interventions/
  README.md                  Project overview and quick start
  requirements.txt           Python dependencies
  manifest.json              Master file index
  manifest.md                Manifest in markdown format

  core/                      Core intervention framework
    __init__.py              Public API exports
    fft.py                   Pure FFT utilities (fft2, ifft2, shifts)
    masks.py                 Frequency mask generation (lowpass, highpass, etc.)
    intervention.py          FrequencyIntervention class

  experiments/               All experiment scripts
    experiment0_identity_validation.py
    experiment1_synthetic_validation.py
    experiment2_real_lowpass.py
    experiment3_layerwise.py
    experiment4_cutoff_sweep.py
    experiment5_robustness.py
    experiment6_dc_boundary.py
    test_hook_replacement.py
    avr_analysis_intervention.py

  configs/                   Configuration files (future use)
  utils/                     Utility scripts (future use)

  docs/                      Documentation
    README.md                Project overview
    methodology.md           Methodological details
    experiments.md           Experiment descriptions
    repository_structure.md  This file

  results/                   All experiment outputs
    experiment0_identity/    (placeholder for identity validation)
    experiment1_synthetic/   (placeholder for synthetic validation)
    experiment2_whole_network/
      results.csv            Stable copy of metrics
      metadata.json          Experiment metadata
      per_layer_avr_*.csv    Per-layer AVR statistics
      metrics_*.csv          Original timestamped metrics
    experiment3_layerwise/
      results.csv            Stable copy of results
      metadata.json          Experiment metadata
      layerwise_results.csv  Stable layer-wise results
      fig_layerwise.png      Layer-wise figure
      figure.png             Stable figure copy
      layerwise_*.csv        Original timestamped results
      layerwise_*.png        Original timestamped figure
    experiment4_cutoff_sweep/
      results.csv            Stable copy of results
      metadata.json          Experiment metadata
      cutoff_sweep.csv       Stable cutoff sweep data
      fig_cutoff_sweep.png   Cutoff sweep figure
      figure.png             Stable figure copy
      cutoff_sweep_*.csv     Original timestamped data
      sweep_plots_*.png      Original timestamped figure
    experiment5_robustness/
      results.csv            Stable copy of results
      metadata.json          Experiment metadata
      layerwise_cutoff050_*.csv Original timestamped data
    experiment6_dc_boundary/
      metadata.json          Combined experiment metadata
      metadata_*.json        Original timestamped metadata

    paper/
      figures/               Publication-quality figures
        fig2_identity_validation.png
        fig3_whole_network.png
        fig4_layerwise_importance.png
        fig5_cutoff_sweep.png
        fig6_avr_vs_importance.png
        fig7_boundary_errors.png
      tables/                Publication-quality tables
        table2_whole_network.csv
        table3_layerwise.csv
        table4_cutoff_sweep.csv
        table6_dc_baseline.csv
        table7_boundary_analysis.csv
      metadata/              Paper metadata