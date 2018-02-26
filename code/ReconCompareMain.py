import ReconCompareAnalysis as rca
import ReconCompareProcessing as rcp


def main():

    # Setup default plotting environment
    rcp.configure_matplotlib()
    # Plots of the reconstruction and reconsrtuction differences time series.
    rca.s01_reconstruction_time_series()
    # Analysis of the raw reconstructions and reconstruction differences with solar cycle phase and Hale cycle phase.
    rca.s02_reconstruction_phase_series_raw()
    # Analysis of the average reconstructions and reconstruction differences with solar cycle and Hale cycle phase.
    rca.s03_reconstruction_phase_series_avg(n_phase_bins=10)
    # Analysis of the raw recon and recon differences with solar cycle phase, splitting the data according to Hale cycle
    # polarity.
    rca.s04_reconstruction_phase_series_avg_polarity_split()
    # Plots of the time series and phase series of the aa geomagnetic storms and ground level enhancements (GLEs), as
    # well as the annual count of storms and GLEs.
    rca.s05_aa_storm_and_gle_time_series()
    # Comparing the distribution of reconstruction residuals in years with and without GLE, with bootstrap testing of
    # the null distribution.
    rca.s06_bootstrap_residual_distributions_with_gle()
    # Comparing the distribution of reconstruction residuals in years with and without great geomagnetic storms (GGMS),
    #  with bootstrap testing of the null distribution.
    rca.s07_bootstrap_residual_distributions_with_ggms()
    # Comparing the distribution of reconstruction residuals in years with and without GGMS, with bootstrap testing of
    # the null distribution.
    # Here this distributions are also split according to Hale cycle phase.
    rca.s08_bootstrap_residual_distributions_with_ggms_polarity_split()
    # A plot for the appendix demonstrating the use of the bootstrap resampling method.
    rca.s09_bootstrap_example_for_appendix()

    return

if __name__ == "__main__":
    main()
