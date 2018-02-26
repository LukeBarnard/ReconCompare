import os
import ReconCompareProcessing as rcp
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import pandas as pd


def s01_reconstruction_time_series():
    """
    A function to produce a time series plot the different reconstructions and the differences between them.
    :return: None
    """
    pcr1, pcr2, ssn, geo = rcp.load_all_data(get_all=False)

    # Print some rudimentary stats about the 4 reconstructions.
    for data, key in zip([pcr1, pcr2, geo, ssn], ['pcr1', 'pcr2', 'geo', 'ssn']):

        print "******************************"
        print "{0:} - mean: {1:3.2f}".format(key, data['hmf'].mean())
        print "{0:} - median: {1:3.2f}".format(key, data['hmf'].median())

    # Find years in PCR1 difference from PCR2
    id_sep = ~np.isclose(pcr1['hmf'], pcr2['hmf'], atol=3e-1)

    fig, ax = plt.subplots(1, 5)
    cols = rcp.get_plot_colors()

    # AX0 - Time series plot of the different reconstructions.
    ax[0].errorbar(pcr1['time'].values, pcr1['hmf'], pcr1['hmf_e'], fmt='s', elinewidth=1.5, color=cols['pcr1'],
                   mec=cols['pcr1'], zorder=1)
    ax[0].plot(pcr1['time'].values[id_sep], pcr1['hmf'][id_sep], 's', color='r', mec='r', zorder=2)
    ax[1].errorbar(pcr2['time'].values, pcr2['hmf'], pcr2['hmf_e'], fmt='D', elinewidth=1.5, color=cols['pcr2'],
                   mec=cols['pcr2'])
    ax[2].errorbar(geo['time'].values, geo['hmf'], geo['hmf_e'], fmt='o', elinewidth=1.5, color=cols['geo'],
                   mec=cols['geo'])
    ax[3].errorbar(ssn['time'].values, ssn['hmf'], ssn['hmf_e'], fmt='^', elinewidth=1.5, color=cols['ssn'],
                   mec=cols['ssn'])

    # AX1 - Distributions of the different reconstructions. Use Gaussian KDE
    ax1_handles = []
    for data, key, label, style in zip([pcr1, pcr2, geo, ssn], ['pcr1', 'pcr2', 'geo', 'ssn'],
                                ['$B_{c1}$', '$B_{c2}$', '$B_{g}$', '$B_{r}$'], ['-', '--', '-.', ':']):
        # KDE estimate the distribution.
        xlo, xhi = ax[0].get_ylim()
        data_span = np.arange(np.fix(xlo), np.ceil(xhi), 0.05)
        kde = sps.gaussian_kde(data['hmf'])
        pdf = kde.pdf(data_span)
        lg, = ax[4].plot(data_span, pdf, style, color=cols[key], label=label)
        ax1_handles.append(lg)

    for a, label in zip(ax[0:4], ['$B_{c1}$', '$B_{c2}$', '$B_{g}$', '$B_{r}$']):
        a.set_ylim(2, 12)
        a.set_yticks(np.arange(3, 13, 2))
        a.set_ylabel("{} (nT)".format(label))
        a.set_xlim(pd.to_datetime("1844-01-01"), pd.to_datetime("1984-01-01"))
        ticks = pd.to_datetime(["{}-01-01".format(i) for i in np.arange(1850, 1990, 20)])
        a.set_xticks(ticks)

    for a in ax[0:3]:
        a.set_xticklabels([])

    # Format the axis and resize the figure
    ax[3].set_xlabel('Year')

    ax[4].set_xlim(2, 12)
    ax[4].set_ylim(0, 0.35)
    ax[4].set_xlabel('Heliospheric Magnetic Field (nT)')
    ax[4].set_ylabel('PDF')
    ax[4].legend(handles=ax1_handles, ncol=2, loc=1)
    ax[4].yaxis.tick_right()
    ax[4].yaxis.set_label_position('right')

    for a, lab in zip(ax[0:4], ["A)", "B)", "C)", "D)"]):
        a.text(0.01, 0.8, lab, transform=a.transAxes, fontsize=14)

    ax[4].text(0.025, 0.95, "E)", transform=ax[4].transAxes, fontsize=14)
    # Position the two axis differently. AX2 is square and on the right hand side.
    fig.set_size_inches(19.5, 6.5)
    ax[3].set_position([0.035, 0.1, 0.615, 0.22])
    ax[2].set_position([0.035, 0.32, 0.615, 0.22])
    ax[1].set_position([0.035, 0.54, 0.615, 0.22])
    ax[0].set_position([0.035, 0.76, 0.615, 0.22])
    ax[4].set_position([0.66, 0.1, 0.3, 0.88])

    proj_dirs = rcp.project_info()
    name = os.path.join(proj_dirs['figs'], 's01_fig1_reconstruction_time_series.png')
    plt.savefig(name)
    plt.close('all')

    # Now look at time series of relative residuals.
    fig, ax = plt.subplots(4, 1)

    # AX0 - Time series plot of the different reconstructions.
    ax[0].errorbar(geo['time'].values, geo['r1'], geo['r1_e'], fmt='o', elinewidth=1.5, color=cols['geo'],
                   mec=cols['geo'])
    ax[1].errorbar(geo['time'].values, geo['r2'], geo['r2_e'], fmt='o', elinewidth=1.5, color=cols['geo'],
                   mec=cols['geo'])
    ax[2].errorbar(ssn['time'].values, ssn['r1'], ssn['r1_e'], fmt='^', elinewidth=1.5, color=cols['ssn'],
                   mec=cols['ssn'])
    ax[3].errorbar(ssn['time'].values, ssn['r2'], ssn['r2_e'], fmt='^', elinewidth=1.5, color=cols['ssn'],
                   mec=cols['ssn'])

    # Set axes to same limits, for easier comparison.
    for a, ylab, lab in zip(ax, ['$F_{g1}$', '$F_{g2}$', '$F_{r1}$', '$F_{r2}$'], ["A)", "B)", "C)", "D)"]):
        a.set_ylim(-1, 1)
        a.set_yticks(np.arange(-0.8, 1.2, 0.4))
        a.set_ylabel("{}".format(ylab))
        a.set_xlim(pd.to_datetime("1844-01-01"), pd.to_datetime("1984-01-01"))
        ticks = pd.to_datetime(["{}-01-01".format(i) for i in np.arange(1850, 1990, 20)])
        a.set_xticks(ticks)
        a.text(0.01, 0.85, lab, transform=a.transAxes, fontsize=14)

    for a in ax[0:3]:
        a.set_xticklabels([])

    # Format the axis and resize the figure
    ax[3].set_xlabel('Year')

    fig.set_size_inches(15, 6.5)
    fig.subplots_adjust(left=0.06, bottom=0.075, right=0.98, top=0.98, hspace=0.0)

    proj_dirs = rcp.project_info()
    name = os.path.join(proj_dirs['figs'], 's01_fig2_reconstruction_rel_diffs_time_series.png')
    plt.savefig(name)
    plt.close('all')
    
    # Now look at distribution of relative residuals.
    fig, ax = plt.subplots(1, 2)

    data_span = np.arange(-1.0, 1.01, 0.01)
    r1_rand = np.random.uniform(0.1, 0.5, size=sum(id_sep))
    r2_rand = np.random.uniform(0.6, 1.0, size=sum(id_sep))
    for key, col, y_rand, in zip(['r1', 'r2'], ['pcr1', 'pcr2'], [r1_rand, r2_rand]):
        
        # KDE estimate the GEO distribution.
        kde = sps.gaussian_kde(geo[key])
        pdf = kde.pdf(data_span)
        ax[0].plot(data_span, pdf, '-', color=cols[col], label='$P(F_{g' + key[1] + '}$)', zorder=1)
        ax[0].plot(geo[key][id_sep], y_rand, 'o', color=cols[col], mec=cols[col], zorder=2)

        # Print out some stats of the distribution
        print "*******************************"
        print "GEO"
        print key
        print "Median: {}".format(geo[key].median())
        data_mode = data_span[np.argmax(pdf)]
        print "Mode: {}".format(data_mode)
        print "Skew: {}".format(geo[key].skew())
        print "p>0: {}".format(kde.integrate_box_1d(data_mode, 1.5))
        print "p<0: {}".format(kde.integrate_box_1d(-1.5, data_mode))
 
        # KDE estimate the SSN distribution.
        kde = sps.gaussian_kde(ssn[key])
        pdf = kde.pdf(data_span)
        ax[1].plot(data_span, pdf, '-', color=cols[col], label='$P(F_{r' + key[1] + '}$)', zorder=1)
        ax[1].plot(ssn[key][id_sep], y_rand, 'o', color=cols[col], mec=cols[col], zorder=2)

        # Print out some stats of the distribution
        print "*******************************"
        print "SSN"
        print key
        print "Median: {}".format(ssn[key].median())
        data_mode = data_span[np.argmax(pdf)]
        print "Mode: {}".format(data_mode)
        print "Skew: {}".format(ssn[key].skew())
        print "p>0: {}".format(kde.integrate_box_1d(data_mode, 1.5))
        print "p<0: {}".format(kde.integrate_box_1d(-1.5, data_mode))

    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position('right')

    # for a, lab in zip(ax, ["A)" "B)"]):
    for a, label in zip(ax, ["A)", "B)"]):
        # Set axis
        a.legend()
        a.set_xlim(-0.75, 0.75)
        a.set_xticks(np.arange(-0.6, 0.9, 0.3))
        a.set_ylim(0, 2.25)
        # Label
        a.text(0.025, 0.925, label, transform=a.transAxes, fontsize=14)
        a.set_xlabel("Fractional difference")
        a.set_ylabel("PDF")

    fig.set_size_inches(10, 5)
    fig.subplots_adjust(left=0.075, bottom=0.1, right=0.925, top=0.98, wspace=0.01)

    proj_dirs = rcp.project_info()
    name = os.path.join(proj_dirs['figs'], 's01_fig3_reconstruction_rel_diffs_distribution.png')
    plt.savefig(name)
    plt.close('all')

    return


def s02_reconstruction_phase_series_raw():
    """
    A function to produce a time series plot the different reconstructions and reconstruction differencees with solar
    cycle phase, with no averaging over solar cycle phase.
    :return:
    """
    pcr1, pcr2, ssn, geo = rcp.load_all_data(get_all=False)

    for key in ['sc_phase', 'hale_phase']:

        # Basic plot of the raw reconstructions with SC phase
        fig, ax, cols = rcp.setup_plot(nrows=1, ncols=1)
        p1, = ax.plot(geo[key], geo['hmf'], '^', color=cols['geo'], label='GEO')
        p2, = ax.plot(ssn[key], ssn['hmf'], 'x', color=cols['ssn'], label='SSN')
        p3, = ax.plot(pcr1[key], pcr1['hmf'], 'o', color=cols['pcr1'], label='PCR1')
        p4, = ax.plot(pcr2[key], pcr2['hmf'], 'o', color=cols['pcr2'], label='PCR2')

        # Find years where pcr1 and pcr2 are different.
        id_sep = ~np.isclose(pcr1['hmf'], pcr2['hmf'], atol=1e-2)
        ax.plot(pcr1[key][id_sep], pcr1['hmf'][id_sep], 's', color='red')
        # Format, save and tidy away plot
        if key == "sc_phase":
            ax.set_xlabel('Schwabe cycle phase, $\phi$, (rads)')
        if key == "hale_phase":
            ax.set_xlabel('Polarity cycle phase, $\phi_{H}$, (rads)')

        ax.set_ylabel('Heliospheric Magnetic Field (nT)')
        ax.set_xlim(0, 2*np.pi)
        ax.legend(handles=[p1, p2, p3, p4], ncol=2, loc=2)
        proj_dirs = rcp.project_info()
        fn = "s02_fig1_reconstruction_{}_series_raw.png".format(key)
        name = os.path.join(proj_dirs['figs'], fn)
        plt.savefig(name)
        plt.close('all')

        # Now look at time series of absolute residuals and relative residuals. Use list of keys in loop to cut down on
        #  repeating plotting code.
        for k1, k2, lab in zip(['d1', 'r1'], ['d2', 'r2'], ['$D', '$R']):
            # Set two row plot, and mark on raw differences with sc phase.
            fig, ax, cols = rcp.setup_plot(nrows=2, ncols=1, figsize=(10, 8))
            p1, = ax[0].plot(geo[key], geo[k1], '^', color=cols['geo'], label=lab + '1_{geo}$')
            p2, = ax[0].plot(ssn[key], ssn[k1], 'x', color=cols['ssn'], label=lab + '1_{ssn}$')
            ax[0].legend(handles=[p1, p2], ncol=2, loc=2)

            p1, = ax[1].plot(geo[key], geo[k2], '^', color=cols['geo'], label=lab + '2_{geo}$')
            p2, = ax[1].plot(ssn[key], ssn[k2], 'x', color=cols['ssn'], label=lab + '2_{ssn}$')
            ax[1].legend(handles=[p1, p2], ncol=2, loc=2)
            # Label up and set lims and format
            if key == "sc_phase":
                ax[1].set_xlabel('Schwabe cycle phase, $\phi$, (rads)')
            if key == "hale_phase":
                ax[1].set_xlabel('Polarity cycle phase, $\phi_{H}$, (rads)')

            if k1 == 'db1':
                ax[0].set_ylabel('Recon - PCR1 (nT)')
                ax[1].set_ylabel('Recon - PCR2 (nT)')
            elif k1 == 'rb1':
                ax[0].set_ylabel('[Recon - PCR1]/Recon')
                ax[1].set_ylabel('[Recon - PCR2]/Recon')

            # Set ylims same for each panel, aids comparison
            y_lims = [a.get_ylim() for a in ax]
            y_lims = [np.min(y_lims), np.max(y_lims)]
            for a in ax:
                a.set_xlim(0, 2 * np.pi)
                a.set_ylim(y_lims)

            proj_dirs = rcp.project_info()
            if k1 == 'd1':
                fn = 's02_fig2_reconstruction_abs_diffs_{}_series_raw.png'.format(key)
                name = os.path.join(proj_dirs['figs'], fn)
            elif k1 == 'r1':
                fn = 's02_fig2_reconstruction_rel_diffs_{}_series_raw.png'.format(key)
                name = os.path.join(proj_dirs['figs'], fn)

            plt.savefig(name)
            plt.close('all')

            return


def s03_reconstruction_phase_series_avg(n_phase_bins=10):
    """
        A function to produce a time series plot the different reconstructions and reconstruction differencees with
        solar cycle phase, which averages over solar cycle phase bins.
        :param n_phase_bins: Number of bins of solar cycle phase. Should be an integer. If not will default to 10, which
                            corresponds to approximately 1 bin per year, for an average solar cycle.
        :return:
        """

    pcr1, pcr2, ssn, geo = rcp.load_all_data(get_all=False)

    # Check the phase bins are sensible
    if not isinstance(n_phase_bins, (int, float)):
        print("Invalid value parsed to n_phase_bins. Defaulting to 10.")
        n_phase_bins = 10
    elif not isinstance(n_phase_bins, int):
        n_phase_bins = int(n_phase_bins)

    # Calculate average of reconstruction parameters and differences over solar cycle phase
    phase_bw = 2*np.pi / n_phase_bins
    phase_bins = np.arange(0, 2*np.pi + phase_bw, phase_bw)

    for ii, key in enumerate(['sc_phase', 'hale_phase']):

        if key == 'sc_phase':
            lg_loc = 1
            geo_avg = rcp.sc_phase_average_recon(phase_bins, geo, ['hmf', 'd1', 'd2', 'r1', 'r2'])
            ssn_avg = rcp.sc_phase_average_recon(phase_bins, ssn, ['hmf', 'd1', 'd2', 'r1', 'r2'])
            pcr1_avg = rcp.sc_phase_average_recon(phase_bins, pcr1, ['hmf'])
            pcr2_avg = rcp.sc_phase_average_recon(phase_bins, pcr2, ['hmf'])
        elif key == 'hale_phase':
            lg_loc = 3
            geo_avg = rcp.hale_phase_average_recon(phase_bins, geo, ['hmf', 'd1', 'd2', 'r1', 'r2'])
            ssn_avg = rcp.hale_phase_average_recon(phase_bins, ssn, ['hmf', 'd1', 'd2', 'r1', 'r2'])
            pcr1_avg = rcp.hale_phase_average_recon(phase_bins, pcr1, ['hmf'])
            pcr2_avg = rcp.hale_phase_average_recon(phase_bins, pcr2, ['hmf'])


        fig, ax, cols = rcp.setup_plot(nrows=3, ncols=1)

        ax[0].errorbar(geo_avg[key], geo_avg['hmf_avg'], yerr=geo_avg['hmf_err'], fmt='-', color=cols['geo'],
                       label='$B_{g}$')
        ax[0].errorbar(ssn_avg[key], ssn_avg['hmf_avg'], yerr=ssn_avg['hmf_err'], fmt='-', color=cols['ssn'],
                       label='$B_{r}$')
        ax[0].errorbar(pcr1_avg[key], pcr1_avg['hmf_avg'], yerr=pcr1_avg['hmf_err'], fmt='-', color=cols['pcr1'],
                       label='$B_{c1}$')
        ax[0].errorbar(pcr2_avg[key], pcr2_avg['hmf_avg'], yerr=pcr2_avg['hmf_err'], fmt='-', color=cols['pcr2'],
                       label='$B_{c2}$')
        ax[0].legend(ncol=2, loc=lg_loc)

        # Get labels for avg, err_l and err_h for this key
        kav = 'r1_avg'
        ker = 'r1_err'

        ax[1].errorbar(geo_avg[key], geo_avg[kav], yerr=geo_avg[ker], color=cols['geo'], label='$F_{g1}$')
        ax[1].errorbar(ssn_avg[key], ssn_avg[kav], yerr=ssn_avg[ker], color=cols['ssn'], label='$F_{r1}$')
        ax[1].hlines(0.0, 0.0, 2*np.pi, linestyles='--', colors='k', linewidths=1.5)
        ax[1].legend(ncol=2, loc=lg_loc)
        # Set transparency and get labels for avg, err_l and err_h for this key
        kav = 'r2_avg'
        ker = 'r2_err'

        ax[2].errorbar(geo_avg[key], geo_avg[kav], yerr=geo_avg[ker], color=cols['geo'], label='$F_{g2}$')
        ax[2].errorbar(ssn_avg[key], ssn_avg[kav], yerr=ssn_avg[ker], color=cols['ssn'], label='$F_{r2}$')
        ax[2].hlines(0.0, 0.0, 2 * np.pi, linestyles='--', colors='k', linewidths=1.5)
        ax[2].legend(ncol=2, loc=lg_loc)

        # Label up and set lims
        if key == "sc_phase":
            ax[2].set_xlabel('Solar cycle phase, $\phi_{s}$, (rads)')
        elif key == "hale_phase":
            ax[2].set_xlabel('Polarity cycle phase, $\phi_{p}$, (rads)')

        ax[0].set_ylabel('HMF (nT)')
        ax[1].set_ylabel('Fractional difference, $F_{1}$')
        ax[2].set_ylabel('Fractional difference, $F_{2}$')

        # All same xlims
        for a, lab in zip(ax, ["A)", "B)", "C)"]):
            a.set_xlim(0, 2*np.pi)
            a.text(0.01, 0.925, lab, transform=a.transAxes, fontsize=14)

        # Remove xticks from top two
        for a in ax[0:2]:
            a.set_xticklabels([])

        ax[0].set_ylim(4.5, 8.5)
        ax[0].set_yticks(np.arange(5, 9, 1))
        # Set ylims equal on difference plots, aids comparison.

        for a in ax[1:]:
            a.set_ylim(-0.25, 0.15)
            a.set_yticks(np.arange(-0.2, 0.1, 0.1))

        fig.set_size_inches(10, 10)
        fig.subplots_adjust(left=0.1, bottom=0.055, right=0.99, top=0.99, hspace=0.05)

        # Save the fig and tidy.
        proj_dirs = rcp.project_info()
        fn = "s03_fig{}_diffs_with_{}.png".format(ii+1, key)
        name = os.path.join(proj_dirs['figs'], fn)

        plt.savefig(name)
        plt.close('all')

        return


def s04_reconstruction_phase_series_avg_polarity_split(n_phase_bins=10):
    """
        A function to produce a time series plot the different reconstructions and reconstruction differencees with
        solar cycle phase, which averages over solar cycle phase bins. This also splits the solar cycles by the polarity
        of the Hale cycle.
        :param n_phase_bins: Number of bins of solar cycle phase. Should be an integer. If not will default to 10, which
                            corresponds to approximately 1 bin per year, for an average solar cycle.
        :return:
        """

    pcr1, pcr2, ssn, geo = rcp.load_all_data(get_all=False)

    # Check the phase bins are sensible
    if not isinstance(n_phase_bins, (int, float)):
        print("Invalid value parsed to n_phase_bins. Defaulting to 10.")
        n_phase_bins = 10
    elif not isinstance(n_phase_bins, int):
        n_phase_bins = int(n_phase_bins)

    # Calculate average of reconstruction parameters and differences over solar cycle phase
    phase_bw = 2*np.pi / n_phase_bins
    phase_bins = np.arange(0, 2*np.pi + phase_bw, phase_bw)

    for key in ['sc_phase', 'hale_phase']:
        if key == 'sc_phase':
            id_even = np.mod(geo['sc_num'], 2) == 0
        elif key == 'hale_phase':
            id_even = np.mod(geo['hale_num'], 2) == 0

        id_odd = np.logical_not(id_even)

        fig, ax_all, cols = rcp.setup_plot(nrows=3, ncols=2)

        if key == 'sc_phase':
            labels = ['Even', 'Odd']
        elif key == 'hale_phase':
            labels = ['qA>0', 'qA<0']

        for polarity, label, ax in zip([id_even, id_odd], labels, [ax_all[:, 0], ax_all[:, 1]]):

            if key == 'sc_phase':
                lg_loc = 1
                geo_avg = rcp.sc_phase_average_recon(phase_bins, geo[polarity], ['hmf', 'd1', 'd2', 'r1', 'r2'])
                ssn_avg = rcp.sc_phase_average_recon(phase_bins, ssn[polarity], ['hmf', 'd1', 'd2', 'r1', 'r2'])
                pcr1_avg = rcp.sc_phase_average_recon(phase_bins, pcr1[polarity], ['hmf'])
                pcr2_avg = rcp.sc_phase_average_recon(phase_bins, pcr2[polarity], ['hmf'])
            elif key == 'hale_phase':
                lg_loc = 1
                geo_avg = rcp.hale_phase_average_recon(phase_bins, geo[polarity], ['hmf', 'd1', 'd2', 'r1', 'r2'])
                ssn_avg = rcp.hale_phase_average_recon(phase_bins, ssn[polarity], ['hmf', 'd1', 'd2', 'r1', 'r2'])
                pcr1_avg = rcp.hale_phase_average_recon(phase_bins, pcr1[polarity], ['hmf'])
                pcr2_avg = rcp.hale_phase_average_recon(phase_bins, pcr2[polarity], ['hmf'])

            ax[0].errorbar(geo_avg[key], geo_avg['hmf_avg'], yerr=geo_avg['hmf_err'], color=cols['geo'],
                           label='$B_{g}$')
            ax[0].errorbar(ssn_avg[key], ssn_avg['hmf_avg'], yerr=ssn_avg['hmf_err'], color=cols['ssn'],
                           label='$B_{r}$')
            ax[0].errorbar(pcr1_avg[key], pcr1_avg['hmf_avg'], yerr=pcr1_avg['hmf_err'], color=cols['pcr1'],
                           label='$B_{c1}$')
            ax[0].errorbar(pcr2_avg[key], pcr2_avg['hmf_avg'], yerr=pcr2_avg['hmf_err'], color=cols['pcr2'],
                           label='$B_{c2}$')
            ax[0].legend(ncol=2, loc=lg_loc)

            # Get labels for avg, err_l and err_h for this key
            kav = 'r1_avg'
            ker = 'r1_err'

            ax[1].errorbar(geo_avg[key], geo_avg[kav], yerr=geo_avg[ker], color=cols['geo'], label='$F_{g1}$')
            ax[1].errorbar(ssn_avg[key], ssn_avg[kav], yerr=ssn_avg[ker], color=cols['ssn'], label='$F_{r1}$')
            ax[1].hlines(0.0, 0.0, 2 * np.pi, linestyles='--', colors='k', linewidths=2)
            ax[1].legend(ncol=2, loc=lg_loc)
            # Set transparency and get labels for avg, err_l and err_h for this key
            kav = 'r2_avg'
            ker = 'r2_err'

            ax[2].errorbar(geo_avg[key], geo_avg[kav], yerr=geo_avg[ker], color=cols['geo'], label='$F_{g2}$')
            ax[2].errorbar(ssn_avg[key], ssn_avg[kav], yerr=ssn_avg[ker], color=cols['ssn'], label='$F_{r2}$')
            ax[2].hlines(0.0, 0.0, 2 * np.pi, linestyles='--', colors='k', linewidths=2)
            ax[2].legend(ncol=2, loc=lg_loc)

            # Label up and set lims
            if key == "sc_phase":
                ax[2].set_xlabel('Solar cycle phase, $\phi_{s}$, (rads)')
            elif key == "hale_phase":
                ax[2].set_xlabel('Polarity cycle phase, $\phi_{p}$, (rads)')

            ax[0].set_ylabel('HMF (nT)')
            ax[1].set_ylabel('Fractional difference, $F_{1}$')
            ax[2].set_ylabel('Fractional difference, $F_{2}$')

            # All same xlims
            for a in ax:
                a.set_xlim(0, 2*np.pi)
                a.text(0.025, 0.85, label, transform=a.transAxes, fontsize=14)

            # Remove xticks from top two
            for a in ax[0:2]:
                a.set_xticklabels([])

            ax[0].set_ylim(4, 9.5)
            ax[0].set_yticks(np.arange(4, 10, 1))
            # Set ylims equal on difference plots, aids comparison.
            y_lims = [-0.25, 0.35]
            for a in ax[1:]:
                a.set_ylim(y_lims)
                # a.set_yticks(np.arange(-0.05,0.3,0.05))

            # If doing the odd ones, put the yaxis ticks on right
            if label in ['Even', 'qA>0']:
                plt_labels = ["A)", "B)", "C)"]
                for a, p_lab in zip(ax, plt_labels):
                    a.text(0.025, 0.925, p_lab, transform=a.transAxes, fontsize=14)
            else:
                plt_labels = ["D)", "E)", "F)"]
                for a, p_lab in zip(ax, plt_labels):
                    a.yaxis.tick_right()
                    a.yaxis.set_label_position("right")
                    a.text(0.025, 0.925, p_lab, transform=a.transAxes, fontsize=14)

        fig.set_size_inches(12.5, 12.5)
        fig.subplots_adjust(left=0.075, bottom=0.055, right=0.925, top=0.99, hspace=0.05, wspace=0.025)

        # Save the fig and tidy.
        proj_dirs = rcp.project_info()
        fn = "s04_fig1_diffs_with_{}_polarity.png".format(key)
        name = os.path.join(proj_dirs['figs'], fn)

        plt.savefig(name)
        plt.close('all')

        return


def s05_aa_storm_and_gle_time_series():
    """
    A function to produce time series and phase series plots of the occurrence of aa geomagnetic storms and GLE events.
    :return: None:
    """
    # Load AA index and AA storms
    aa = rcp.load_aa_data()
    # Compute the top 10% of storms for both peak magnitude and integrated intensity
    for metric in ['peak', 'integral']:
        # Only keep top 10% of storms, according to metric
        aa_storms = rcp.load_aa_storms(100)
        aa_storms = aa_storms[aa_storms[metric] > aa_storms[metric].quantile(0.9)]
        aa_storms.set_index(np.arange(0, len(aa_storms)), inplace=True)

        # Load GLE data
        gle = rcp.load_gle_list()

        fig, ax = plt.subplots(3, 1)
        # AX0 - Time series plot of the different reconstructions.
        ylim = 900
        ax[0].plot(aa['time'], aa['val'], 'k-', label='aa', zorder=0)
        ax[0].plot(aa_storms['time_peak'], aa_storms['peak'], 'ro', markeredgecolor='r', label='aa storm peak', zorder=1)
        ax[0].vlines(gle['time'].values, 0, 100, colors=['c'], label='GLE', zorder=2)
        ax[0].set_xlabel('Year')
        ax[0].set_xlim(aa['time'].min(), aa['time'].max())
        ax[0].set_ylabel('aa (nT)')
        ax[0].set_ylim(0, ylim)
        ax[0].legend(loc=1)

        ax[1].plot(aa['sc_phase'], aa['val'], 'k.', label='aa', zorder=0)
        ax[1].plot(aa_storms['sc_phase'], aa_storms['peak'], 'ro', markeredgecolor='r', label='aa storm peak', zorder=1)
        ax[1].vlines(gle['sc_phase'].values, 0, 100, colors=['c'], label='GLE', zorder=2)
        ax[1].set_xlabel('Solar Cycle Phase, $\phi_{s}$, (rads)')
        ax[1].set_ylabel('aa (nT)')
        ax[1].set_xlim(-0.025, 2.025*np.pi)
        ax[1].set_ylim(0, ylim)
        ax[1].legend(loc=1)

        ax[2].plot(aa['hale_phase'], aa['val'], 'k.', label='aa', zorder=0)
        ax[2].plot(aa_storms['hale_phase'], aa_storms['peak'], 'ro', markeredgecolor='r', label='aa storm peak', zorder=1)
        ax[2].vlines(gle['hale_phase'].values, 0, 100, colors=['c'], label='GLE', zorder=2)
        ax[2].set_xlabel('Polarity Cycle Phase, $\phi_{p}$, (rads)')
        ax[2].set_ylabel('aa (nT)')
        ax[2].set_xlim(-0.025, 2.025 * np.pi)
        ax[2].set_ylim(0, ylim)
        ax[2].legend(loc=1)

        for a, lab in zip(ax, ["A)", "B)", "C)"]):
            a.text(0.01, 0.925, lab, transform=a.transAxes, fontsize=14)

        fig.set_size_inches(12.5, 12.5)
        fig.subplots_adjust(left=0.06, bottom=0.05, right=0.99, top=0.99, hspace=0.15)

        proj_dirs = rcp.project_info()
        name = os.path.join(proj_dirs['figs'], "s05_fig1_aa_storm_{}_and_gle_time_and_phase_series.png".format(metric))
        plt.savefig(name)
        plt.close('all')

        # Now do the same plot the annual/phase bin count of aa storms and gles
        fig, ax = plt.subplots(3, 1)
        # AX0 - Time series plot of the different reconstructions.
        years = np.arange(aa_storms['time'].min().year, aa_storms['time'].max().year+1, 1)
        aa_years = np.array([t.year for t in aa_storms['time']])
        aa_count = np.array([np.sum(aa_years == yr) for yr in years])
        ax[0].plot(years, aa_count, 'r-', label='aa storms', zorder=1)

        years = np.arange(gle['time'].min().year, gle['time'].max().year + 1, 1)
        gle_years = np.array([t.year for t in gle['time']])
        gle_count = np.array([np.sum(gle_years == yr) for yr in years])
        ax[0].plot(years, gle_count, 'c-', label='GLE', zorder=2)

        ax[0].set_xlabel('Year')
        ax[0].set_xlim(aa['time'].min().year, aa['time'].max().year)
        ax[0].set_ylabel('Annual number of events')
        ax[0].set_ylim(0, 15)
        ax[0].legend(loc=1)

        n_phase_bins = 10
        phase_bw = 2 * np.pi / n_phase_bins
        phase_bins = np.arange(0, 2 * np.pi + phase_bw, phase_bw)
        phase_mid = np.mean([phase_bins[:-1], phase_bins[1:]], axis=0)
        aa_count = np.zeros(len(phase_bins)-1)
        gle_count = np.zeros(len(phase_bins) - 1)
        for i in range(0, len(phase_bins)-1):
            id_storms = (aa_storms['sc_phase'] > phase_bins[i]) & (aa_storms['sc_phase'] <= phase_bins[i+1])
            aa_count[i] = np.sum(id_storms)

            id_gle = (gle['sc_phase'] > phase_bins[i]) & (gle['sc_phase'] <= phase_bins[i + 1])
            gle_count[i] = np.sum(id_gle)

        ax[1].plot(phase_mid, aa_count, 'ro-', markeredgecolor='r', label='aa storms', zorder=1)
        ax[1].plot(phase_mid, gle_count, 'cs-', markeredgecolor='c', label='GLE', zorder=1)
        ax[1].set_xlabel('Solar Cycle Phase, $\phi_{s}$, (rads)')
        ax[1].set_ylabel('Number of events')
        ax[1].set_xlim(-0.025, 2.025 * np.pi)
        ax[1].set_ylim(0, 60)
        ax[1].legend(loc=1)

        n_phase_bins = 10
        phase_bw = 2 * np.pi / n_phase_bins
        phase_bins = np.arange(0, 2 * np.pi + phase_bw, phase_bw)
        phase_mid = np.mean([phase_bins[:-1], phase_bins[1:]], axis=0)
        aa_count = np.zeros(len(phase_bins) - 1)
        gle_count = np.zeros(len(phase_bins) - 1)
        for i in range(0, len(phase_bins) - 1):
            id_storms = (aa_storms['hale_phase'] > phase_bins[i]) & (aa_storms['hale_phase'] <= phase_bins[i + 1])
            aa_count[i] = np.sum(id_storms)

            id_gle = (gle['hale_phase'] > phase_bins[i]) & (gle['hale_phase'] <= phase_bins[i + 1])
            gle_count[i] = np.sum(id_gle)

        ax[2].plot(phase_mid, aa_count, 'ro-', markeredgecolor='r', label='aa storms', zorder=1)
        ax[2].plot(phase_mid, gle_count, 'cs-', markeredgecolor='c', label='GLE', zorder=1)
        ax[2].set_xlabel('Polarity Cycle Phase, $\phi_{p}$, (rads)')
        ax[2].set_ylabel('Number of events')
        ax[2].set_xlim(-0.025, 2.025 * np.pi)
        ax[2].set_ylim(0, 60)
        ax[2].legend(loc=1)

        for a, lab in zip(ax, ["A)", "B)", "C)"]):
            a.text(0.01, 0.925, lab, transform=a.transAxes, fontsize=14)

        fig.set_size_inches(12.5, 12.5)
        fig.subplots_adjust(left=0.06, bottom=0.05, right=0.99, top=0.99, hspace=0.15)

        proj_dirs = rcp.project_info()
        name = "s05_fig2_aa_storm_{}_and_gle_count_with_time_and_phase.png".format(metric)
        name = os.path.join(proj_dirs['figs'], name)
        plt.savefig(name)
        plt.close('all')

        return


def s06_bootstrap_residual_distributions_with_gle():
    """
    A function to compare the distribution of the reconstruction residuals conditional on whether GLEs were known to
    have occurred. Also computes bootstrap estimates of the full distribution, to help assess whether the apparent
    differences between the GLE and no-GLE conditions are possibly due to random sampling from the full distribution.
    :return: None:
    """

    pcr1, pcr2, ssn, geo, aa, aa_storms, gle = rcp.load_all_data(get_all=True, match_gle=True)
    # Get boolean arrays of years with and without gles
    year_of_gle = np.array([g.year for g in gle['time']])
    year_of_recon = np.array([r.year for r in ssn['time']])
    # Match the years to count the GLEs
    num_of_gle = np.array([np.sum(year_of_gle == yr) for yr in year_of_recon], dtype=int)
    id_gle = num_of_gle != 0
    id_nogle = ~id_gle

    print "**********************************************"
    print "Residual distributions conditonal on GLE years"
    print "Total number of years: {:02d}".format(len(id_gle))
    print "Years with GLE: {:02d}".format(sum(id_gle))
    print "Years without GLE: {:02d}".format(sum(id_nogle))

    # Now do CDF plots without fitting the data.
    fig, ax, cols = rcp.setup_plot(nrows=2, ncols=2, figsize=(10, 10))

    for a, data, lab in zip([ax[:, 0], ax[:, 1]], [geo, ssn], ['geo', 'ssn']):
        # NO GGMS
        data_support, bs_pdfs, bs_sample = rcp.bootstrap_distributions(data['r1'], sum(id_nogle), 100)
        # columns of bs_sample are sorted, so are already suited for ECDF plotting.
        ecdf = np.arange(1, bs_sample.shape[0] + 1, 1) / np.float(bs_sample.shape[0])
        a[0].step(bs_sample, ecdf, '-', where="pre", color='gray', alpha=0.2, linewidth=1, label='Bootstraps')

        xval = np.sort(data['r1'][id_nogle].values)
        ecdf = np.arange(1, len(xval) + 1, 1) / np.float(len(xval))
        a[0].step(xval, ecdf, '-', where="pre", color=cols[lab], linewidth=3, label="Observed")

        # GMS
        # Get bootstrap distributions for random selection of years.
        data_support, bs_pdfs, bs_sample = rcp.bootstrap_distributions(data['r1'], sum(id_gle), 100)
        # columns of bs_sample are sorted, so are already suited for ECDF plotting.
        ecdf = np.arange(1, bs_sample.shape[0] + 1, 1) / np.float(bs_sample.shape[0])
        a[1].step(bs_sample, ecdf, '-', where="pre", color='gray', alpha=0.2, linewidth=1, label='Bootstraps')

        xval = np.sort(data['r1'][id_gle].values)
        ecdf = np.arange(1, len(xval) + 1, 1) / np.float(len(xval))
        a[1].step(xval, ecdf, '-', where="pre", color=cols[lab], linewidth=3, label="Observed")

        a[0].text(0.025, 0.9, "No GLE", transform=a[0].transAxes, fontsize=14)
        a[1].text(0.025, 0.9, "GLE", transform=a[1].transAxes, fontsize=14)

    for a, lab in zip(ax.ravel(), ["A)", "B)", "C)", "D)"]):
        # Set lims all equal
        a.set_xlim(-0.79, 0.79)
        a.set_ylim(0, 1.0)
        # Add on the legends
        handles, labels = a.get_legend_handles_labels()
        a.legend(handles[-2:], labels[-2:], loc=6, frameon=False)
        a.text(0.025, 0.95, lab, transform=a.transAxes, fontsize=14)

    # Sort out labelling
    ax[1, 0].set_xlabel("Fractional residual, $F_{g1}$")
    ax[1, 1].set_xlabel("Fractional residual, $F_{r1}$")

    for a in ax[:, 0]:
        a.set_ylabel("CDF($F_{g1}$)")

    for a in ax[:, 1]:
        a.set_ylabel("CDF($F_{r1}$)")

    for a in ax[0, :]:
        a.set_xticklabels([])

    for a in ax[:, 1]:
        a.yaxis.tick_right()
        a.yaxis.set_label_position('right')

    # Print2
    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.075, top=0.99, wspace=0.04, hspace=0.04)
    proj_dirs = rcp.project_info()
    name = os.path.join(proj_dirs['figs'], 's06_fig1_gle_cdf.png')
    plt.savefig(name)

    return


def s07_bootstrap_residual_distributions_with_ggms():
    """
    A function to compare the distribution of the reconstruction residuals conditional on whether great geomagnetic
    storms (GGMS) were known to have occurred. Also computes bootstrap estimates of the full distribution, to help
    assess whether the apparent differences between the GGMS and no-GGMS conditions are possibly due to random sampling
    from the full distribution.
    :return: None:
    """
    pcr1, pcr2, ssn, geo, aa, aa_storms, gle = rcp.load_all_data(get_all=True, match_aa=True, match_gle=False)

    for metric in ['peak', 'integral']:
        # Only keep top 10% of storms
        ggms = aa_storms[aa_storms[metric] > aa_storms[metric].quantile(0.9)]
        ggms.set_index(np.arange(0, len(ggms)), inplace=True)

        # Print out storm stats
        if metric == "peak":
            print "**********************************************"
            print "Total number of storms: {:02d}".format(len(aa_storms))
            print "Total number of large storms: {:02d}".format(len(ggms))

        # Get boolean arrays of years with and without gles
        year_of_ggms = np.array([g.year for g in ggms['time']])
        year_of_recon = np.array([r.year for r in ssn['time']])
        # Match the years to count the GLEs
        num_of_ggms = np.array([np.sum(year_of_ggms == yr) for yr in year_of_recon], dtype=int)
        id_ggms = num_of_ggms != 0
        id_noggms = ~id_ggms

        print "**********************************************"
        print metric.upper()
        print "Residual distributions conditonal on GGMS years"
        print "Total number of years: {:02d}".format(len(id_ggms))
        print "Years with GGMS: {:02d}".format(sum(id_ggms))
        print "Years without GGMS: {:02d}".format(sum(id_noggms))

        # Now do CDF plots without fitting the data.
        fig, ax, cols = rcp.setup_plot(nrows=2, ncols=2, figsize=(10, 10))

        for a, data, lab in zip([ax[:, 0], ax[:, 1]], [geo, ssn], ['geo', 'ssn']):
            # NO GGMS
            data_support, bs_pdfs, bs_sample = rcp.bootstrap_distributions(data['r1'], sum(id_noggms), 100)
            # columns of bs_sample are sorted, so are already suited for ECDF plotting.
            ecdf = np.arange(1, bs_sample.shape[0] + 1, 1) / np.float(bs_sample.shape[0])
            a[0].step(bs_sample, ecdf, '-', where="pre", color='gray', alpha=0.2, linewidth=1, label='Bootstraps')

            xval = np.sort(data['r1'][id_noggms].values)
            ecdf = np.arange(1, len(xval) + 1, 1) / np.float(len(xval))
            a[0].step(xval, ecdf, '-', where="pre", color=cols[lab], linewidth=3, label="Observed")

            # GMS
            # Get bootstrap distributions for random selection of years.
            data_support, bs_pdfs, bs_sample = rcp.bootstrap_distributions(data['r1'], sum(id_ggms), 100)
            # columns of bs_sample are sorted, so are already suited for ECDF plotting.
            ecdf = np.arange(1, bs_sample.shape[0] + 1, 1) / np.float(bs_sample.shape[0])
            a[1].step(bs_sample, ecdf, '-', where="pre", color='gray', alpha=0.2, linewidth=1, label='Bootstraps')

            xval = np.sort(data['r1'][id_ggms].values)
            ecdf = np.arange(1, len(xval) + 1, 1) / np.float(len(xval))
            a[1].step(xval, ecdf, '-', where="pre", color=cols[lab], linewidth=3, label="Observed")

            a[0].text(0.025, 0.9, "No GGMS", transform=a[0].transAxes, fontsize=14)
            a[1].text(0.025, 0.9, "GGMS", transform=a[1].transAxes, fontsize=14)

            # Do KS test on the distributions.
            k_stat, p_val = sps.ks_2samp(data['r1'][id_noggms].values, data['r1'][id_ggms].values)
            print "{} - D, p_val: {:3.2f}, {:5.4f}".format(lab.upper(), k_stat, p_val)

        for a, lab in zip(ax.ravel(), ["A)", "B)", "C)", "D)"]):
            # Set lims all equal
            a.set_xlim(-0.79, 0.79)
            a.set_ylim(0, 1.0)
            # Add on the legends
            handles, labels = a.get_legend_handles_labels()
            a.legend(handles[-2:], labels[-2:], loc=6, frameon=False)
            a.text(0.025, 0.95, lab, transform=a.transAxes, fontsize=14)

        # Sort out labelling
        ax[1, 0].set_xlabel("Fractional residual, $F_{g1}$")
        ax[1, 1].set_xlabel("Fractional residual, $F_{r1}$")

        for a in ax[:, 0]:
            a.set_ylabel("CDF($F_{g1}$)")

        for a in ax[:, 1]:
            a.set_ylabel("CDF($F_{r1}$)")

        for a in ax[0, :]:
            a.set_xticklabels([])

        for a in ax[:, 1]:
            a.yaxis.tick_right()
            a.yaxis.set_label_position('right')

        # Print2
        fig.subplots_adjust(left=0.075, right=0.925, bottom=0.075, top=0.99, wspace=0.035, hspace=0.035)
        proj_dirs = rcp.project_info()
        name = os.path.join(proj_dirs['figs'], "s07_fig1_ggms_{}_cdf.png".format(metric))
        plt.savefig(name)

        return


def s08_bootstrap_residual_distributions_with_ggms_polarity_split():
    """
    A function to compare the distribution of the reconstruction residuals conditional on whether GLEs were known to
    have occurred. This also splits the analysis according to the polarity of the Hale cycle phase. This computes
    bootstrap estimates of the full distribution, to help assess whether the apparent differences between the GLE and
    no-GLE conditions are possibly due to random sampling from the full distribution.
    :return: None:
    """
    pcr1, pcr2, ssn_all, geo_all, aa, aa_storms, gle = rcp.load_all_data(get_all=True, match_aa=True, match_gle=False)

    # Only keep top 10% of storms
    for metric in ['peak', 'integral']:
        ggms_all = aa_storms[aa_storms[metric] > aa_storms[metric].quantile(0.9)]
        ggms_all.set_index(np.arange(0, len(ggms_all)), inplace=True)
        DATA = {'low': {'geo': {'noggms': {}, 'ggms': {}},
                        'ssn': {'noggms': {}, 'ggms': {}}},
                'high': {'geo': {'noggms': {}, 'ggms': {}},
                         'ssn': {'noggms': {}, 'ggms': {}}}}

        for phase in ['low', 'high']:

            # Split the data according to polarity phase.
            if phase == 'low':
                id_polarity = (geo_all['hale_phase'] > (np.pi / 2.0)) & (geo_all['hale_phase'] < (3.0 * np.pi / 2.0))
                geo = geo_all.loc[id_polarity]
                ssn = ssn_all.loc[id_polarity]
                id_polarity = (ggms_all['hale_phase'] > (np.pi / 2.0)) & (ggms_all['hale_phase'] < (3.0 * np.pi / 2.0))
                ggms = ggms_all.loc[id_polarity]
            elif phase == 'high':
                id_polarity = (geo_all['hale_phase'] <= (np.pi / 2.0)) | (geo_all['hale_phase'] >= (3.0 * np.pi / 2.0))
                geo = geo_all.loc[id_polarity]
                ssn = ssn_all.loc[id_polarity]
                id_polarity = (ggms_all['hale_phase'] <= (np.pi / 2.0)) | (ggms_all['hale_phase'] >= (3.0 * np.pi / 2.0))
                ggms = ggms_all.loc[id_polarity]

            # Get boolean arrays of years with and without gles
            year_of_ggms = np.array([g.year for g in ggms['time']])
            year_of_recon = np.array([r.year for r in ssn['time']])
            # Match the years to count the GLEs
            num_of_ggms = np.array([np.sum(year_of_ggms == yr) for yr in year_of_recon], dtype=int)
            id_ggms = num_of_ggms != 0
            id_noggms = ~id_ggms

            print "**********************************************"
            print metric.upper()
            print phase.upper()
            print "Residual distributions conditonal on GGMS years"
            print "Total number of years: {:02d}".format(len(id_ggms))
            print "Years with GGMS: {:02d}".format(sum(id_ggms))
            print "Years without GGMS: {:02d}".format(sum(id_noggms))

            # Now do CDF plots without fitting the data.
            fig, ax, cols = rcp.setup_plot(nrows=2, ncols=2, figsize=(10, 10))

            for a, data, lab in zip([ax[:, 0], ax[:, 1]], [geo, ssn], ['geo', 'ssn']):
                # NO GGMS
                data_support, bs_pdfs, bs_sample = rcp.bootstrap_distributions(data['r1'], sum(id_noggms), 100)
                # columns of bs_sample are sorted, so are already suited for ECDF plotting.
                ecdf = np.arange(1, bs_sample.shape[0] + 1, 1) / np.float(bs_sample.shape[0])
                a[0].step(bs_sample, ecdf, '-', where="pre", color='gray', alpha=0.2, linewidth=1, label='Bootstraps')

                xval = np.sort(data['r1'][id_noggms].values)
                ecdf = np.arange(1, len(xval) + 1, 1) / np.float(len(xval))
                a[0].step(xval, ecdf, '-', where="pre", color=cols[lab], linewidth=3, label="Observed")

                # Join the observed values with the bootstrap estimates to compute the observed rank relative to BS.
                xval = xval.reshape((len(xval), 1))
                all_data = np.hstack((xval, bs_sample))
                bs_rank = np.zeros(all_data.shape)
                for i in range(bs_rank.shape[0]):
                    bs_rank[i, :] = sps.rankdata(all_data[i, :])
                DATA[phase][lab]['noggms'] = {'data': xval, 'ecdf': ecdf, 'bs_rank': bs_rank[:, 0]}

                # GMS
                # Get bootstrap distributions for random selection of years.
                data_support, bs_pdfs, bs_sample = rcp.bootstrap_distributions(data['r1'], sum(id_ggms), 100)
                # columns of bs_sample are sorted, so are already suited for ECDF plotting.
                ecdf = np.arange(1, bs_sample.shape[0] + 1, 1) / np.float(bs_sample.shape[0])
                a[1].step(bs_sample, ecdf, '-', where="pre", color='gray', alpha=0.2, linewidth=1, label='Bootstraps')

                xval = np.sort(data['r1'][id_ggms].values)
                ecdf = np.arange(1, len(xval) + 1, 1) / np.float(len(xval))
                a[1].step(xval, ecdf, '-', where="pre", color=cols[lab], linewidth=3, label="Observed")

                xval = xval.reshape((len(xval), 1))
                all_data = np.hstack((xval, bs_sample))
                bs_rank = np.zeros(all_data.shape)
                for i in range(bs_rank.shape[0]):
                    bs_rank[i, :] = sps.rankdata(all_data[i, :])
                DATA[phase][lab]['ggms'] = {'data': xval, 'ecdf': ecdf, 'bs_rank': bs_rank[:, 0]}

                a[0].text(0.025, 0.9, "No GGMS", transform=a[0].transAxes, fontsize=14)
                a[1].text(0.025, 0.9, "GGMS", transform=a[1].transAxes, fontsize=14)

                # Do KS test on the distributions.
                k_stat, p_val = sps.ks_2samp(data['r1'][id_noggms].values, data['r1'][id_ggms].values)
                print "{} - D, p_val: {:3.2f}, {:5.4f}".format(lab.upper(), k_stat, p_val)

            for a, lab in zip(ax.ravel(), ["A)", "B)", "C)", "D)"]):
                # Set lims all equal
                a.set_xlim(-0.79, 0.79)
                a.set_ylim(0, 1.0)
                # Add on the legends
                handles, labels = a.get_legend_handles_labels()
                a.legend(handles[-2:], labels[-2:], loc=6, frameon=False)
                a.text(0.025, 0.95, lab, transform=a.transAxes, fontsize=14)

            # Sort out labelling
            ax[1, 0].set_xlabel("Fractional residual, $F_{g1}$")
            ax[1, 1].set_xlabel("Fractional residual, $F_{r1}$")

            for a in ax[:, 0]:
                a.set_ylabel("CDF($F_{g1}$)")

            for a in ax[:, 1]:
                a.set_ylabel("CDF($F_{r1}$)")

            for a in ax[0, :]:
                a.set_xticklabels([])

            for a in ax[:, 1]:
                a.yaxis.tick_right()
                a.yaxis.set_label_position('right')

            # Print2
            fig.subplots_adjust(left=0.075, right=0.925, bottom=0.075, top=0.99, wspace=0.035, hspace=0.035)
            proj_dirs = rcp.project_info()
            name = os.path.join(proj_dirs['figs'], "s08_fig1_ggms_{}_cdf_{}.png".format(metric, phase))
            plt.savefig(name)
            plt.close('all')

            return


def s09_bootstrap_example_for_appendix():
    """
    Function to produce a plot demonstrating how the bootstrap resampling method is used to help assess the
    differences between a parameters distribution function under different sub-sampling conditions.
    :return:
    """
    # Load the aa and sunspot count data.
    aa = rcp.load_aa_data()
    # Rename the aa data column from val to aa, so that aa data clearly labelled int he joined dataframe
    aa.rename(index=str, columns={'val': 'aa'}, inplace=True)
    aa.drop(['jd', 'sc_num', 'sc_phase', 'sc_state', 'hale_num', 'hale_phase'], axis=1, inplace=True)
    aa.set_index('time', inplace=True, drop=True)
    aa = aa.resample('D').mean()

    # Load in the sunspot data
    ssn = rcp.load_sunspot_count_data()
    ssn.set_index('time', inplace=True, drop=True)

    # Get the inner join of these frames
    data = aa.join(ssn, how='inner')
    data['time'] = data.index

    # Loose all NaN data and re-index.
    data.dropna(axis=0, how='any', inplace=True)
    data.set_index(np.arange(0, data.shape[0]), inplace=True)

    # Compute the ECDFs of all of aa data, and of subset for R>400
    x_all = data['aa'].sort_values()
    F_all = np.arange(1, data.shape[0] + 1, 1) / (data.shape[0] + 1.0)

    id_r_0 = data['R'] > 400
    x_r_0 = data['aa'][id_r_0].sort_values()
    F_r_0 = np.arange(1, x_r_0.size + 1, 1) / (x_r_0.size + 1.0)
    # Get sample size for bootstrap estimates
    n_samp_r_0 = x_r_0.size
    # Make bootstrap estimates of the null distribution
    bs_support, bs_pdf, bs_sample = rcp.bootstrap_distributions(x_all, n_samp_r_0, 100)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(data['R'], data['aa'], 'k.', label='aa')
    ax[0].plot(data['R'][id_r_0], data['aa'][id_r_0], 'r.', label='aa(R>400)')
    ax[0].set_xlabel('Daily Sunspot Number, R')
    ax[0].set_ylabel('Daily mean aa index (nT)')
    ax[0].legend(frameon=False)

    ax[1].step(x_all, F_all, '-', where="pre", color='k', linewidth=2, label='F(aa)', zorder=2)
    ax[1].step(x_r_0, F_r_0, '-', where="pre", color='r', linewidth=2, label='F(aa(R>400))', zorder=1)
    ax[1].step(bs_sample, F_r_0, '-', where="pre", color='gray', alpha=0.2, linewidth=1, label='Bootstraps', zorder=0)
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Daily mean aa index (nT)')
    ax[1].set_ylabel("ECDF")

    # Add on the legends
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles[0:3], labels[0:3], frameon=False)
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position('right')
    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.1, top=0.99, wspace=0.04, hspace=0.04)
    proj_dirs = rcp.project_info()
    name = os.path.join(proj_dirs['figs'], 's09_bootstrap_example_for_appendix.png')
    fig.savefig(name)

    return
