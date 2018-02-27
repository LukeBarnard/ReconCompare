import numpy as np
import pandas as pd
import os
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.stats as sps


def project_info():
    """
    A function to create a dictionary of project directories, pointing to the different data files and folders, stored
    in config.txt
    :return proj_dirs: Dictionary of project directories, with keys 'data','figures','code', and 'results'.
    """
    files = glob.glob('config.txt')

    if len(files) != 1:
        # If too few or too many config files, guess projdirs
        print('Error: Cannot find correct config file with project directories. Check config.txt exists')
    else:
        # Open file and extract
        with open(files[0], "r") as f:
            lines = f.read().splitlines()
            proj_dirs = {l.split(',')[0]: l.split(',')[1] for l in lines}
            # Update the paths with the root directory (except for root!).
            for k in proj_dirs.iterkeys():
                if k != "root":
                    proj_dirs[k] = os.path.join(proj_dirs['root'], proj_dirs[k])

    # Just check the directories exist.
    for val in proj_dirs.itervalues():
        if not os.path.exists(val):
            print('Error, invalid path, check config: '+val)

    return proj_dirs


def get_plot_colors():
    """
    A function to return a dictionary of default settings for plots, inc:
    fsz : Fontsize
    lw  : Linewidth
    col : Colors for different parameters
    :return:
    """
    # Get default plotting colors for each parameter
    cols = {'geo': '#e41a1c', 'ssn': '#377eb8', 'pcr1': '#984ea3', 'pcr2': '#ff7f00',
            'aa': 'black', 'aas': 'black', 'gle': 'black', 'nm': 'black'}
    return cols


def configure_matplotlib():
    """
    Function to configure the default paramters for matplotlib. Changes default fontsizes, mathtext, subplots and legend
    :return: None
    """
    mpl.rc('lines', linewidth=3)
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)
    mpl.rc('figure.subplot', left=0.1, right=0.98, bottom=0.1, top=0.98, wspace=0.1, hspace=0.1)
    mpl.rc('mathtext', default='regular')
    mpl.rc('legend', numpoints=1, fontsize=14, labelspacing=0.1, handletextpad=0.3, frameon=False, framealpha=0.0,
           columnspacing=0.3, scatterpoints=1, handlelength=1)
    mpl.rc('savefig', dpi=300)

    return


def setup_plot(nrows=1, ncols=1, figsize=(10, 5), distrib_plots=False):
    """
    A function to setup up a plot with nrows x ncols panels, of total size figsize. Returns dictionary containing figure
    handle, axes handle and some ancillary information for plotting, like default colors, linewidths and fontsizes.
    :param nrows: Number of rows of panels.
    :param ncols: Number of columns of panels.
    :param figsize: Figure size in inches, as a tuple (width,height).
    :param distrib_plots: A switch to format the plot setup for the distribution plots
    :return: plot_defaults: Dictionary with keys 'fig','ax','meta', where meta is a dictionary containing default
                            color strings, fontsizess and linewidths.
    """
    if not isinstance(distrib_plots, bool):
        print("Error: distrib_plots is not a bool. Defaulting to false")
        distrib_plots = False

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    cols = get_plot_colors()
    if distrib_plots:
        # Special option for setting up the distribution plots.
        ax[1].yaxis.tick_right()
        ax[1].yaxis.set_label_position('right')
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.975, wspace=0.1)

    return fig, ax, cols
    

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    :param condition: 1-D boolean array
    :return: 2D array where the first and second columns are the start and end indices of contiguously True regions.
    """

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()
    # We need to start things after the change in "condition". Therefore, shift the index by 1 to the right.
    idx += 1
    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[len(condition) - 1]:  # edited from SO solution, as was hanging on the -1 for some reason.
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def define_storms(time, geoindex, thresh, direction='above'):
    """
    Function to identify Storms (peaks above or below a threshold) in the geomagnetic index. Should work for indices of
     different types, such as AA or DST (i.e positive or negative exceedences). Based on the Kilpua et al 2015
     definition of storms in AA, which is the exceedence over 100nT.
    :param time: Array of observation times.
    :param geoindex: Array of geomagnetic index values
    :param thresh: Threshold used to define storms
    :param direction: Whether to look for exceedences above or below the threshold
    :return: storms: DataFrame of storms defined in a geomagnetic index, including time, duration, peak value and time
                     of peak.
    """
    if not isinstance(thresh, (int, float)):
        print('Error: thresh not set to int or float. Defaulting to 100')
        thresh = 100

    # Check that thresh is inside range of geoindex, warn if not.
    if (thresh > geoindex.max()) or (thresh < geoindex.min()):
        print('Error: thresh outside range of geoindex. No storms will be found.')

    if direction == 'above':
        condition = geoindex > thresh
    elif direction == 'below':
        condition = geoindex < thresh
    else:
        print('Error: invalid direction parsed to define_storms. Defaulting to above.')
        condition = geoindex > thresh

    start_time = []
    max_val = []
    max_time = []
    duration = []
    integral = []
    len_index = len(geoindex)
    for start, stop in contiguous_regions(condition):
        if stop <= (len_index) - 1:
            start_time.append(time[start])
            if direction == 'above':
                max_val.append(geoindex[start:stop].max())
                max_time.append(time[np.argmax(geoindex[start:stop])])
            elif direction == 'below':
                max_val.append(geoindex[start:stop].min())
                max_time.append(time[np.argmin(geoindex[start:stop])])

            # Compute storm duration (hours)
            duration.append(time[stop] - time[start])
            # Compute integrated intensity
            integral.append(geoindex[start:stop].sum() * (3 * 3600))

    storms = {'time': start_time, 'peak': max_val, 'time_peak': max_time,
              'duration': duration, 'integral': integral}
    return pd.DataFrame(storms)


def calc_solar_cycle_phase(time):
    """
    Function to calculate the solar cycle number, solar cycle phase (between 0-2pi),and solar cycle state (rising=+1,
    falling =-1) for an array of times, provided as julian dates. Dates taken from Wikipedia, could be improved.
    :param time: An array of julian dates
    :return: sc_num     : Solar cycle number
             sc_phase   : Solar cycle phase
             sc_state   : Solar cycle state
             sc_max     : Solar cycle maximum time
             hale_num   : Hale cycle number, assuming polar fields flip 1 year after sunspot maximum.
             hale_phase : Phase of the Hale cycle, assuming polar fields flip 1 year after sunspot maximum.
    """
    # Get start times and max times in julian dates
    start_times_list = ['1744-01-01', '1755-08-01', '1766-03-01', '1775-08-01', '1784-06-01', '1798-06-01',
                        '1810-06-01', '1823-12-01', '1833-10-01', '1843-09-01', '1855-03-01',
                        '1867-02-01', '1878-09-01', '1890-06-01', '1902-09-01', '1913-12-01',
                        '1923-05-01', '1933-09-01', '1944-01-01', '1954-02-01', '1964-10-01',
                        '1976-05-01', '1986-03-01', '1996-06-01', '2008-01-01', '2020-01-01']

    max_times_list = ['1748-01-01', '1761-06-01', '1769-09-01', '1778-05-01', '1788-02-01', '1805-02-01',
                      '1816-05-01', '1829-11-01', '1837-03-01', '1848-02-01', '1860-02-01',
                      '1870-08-01', '1883-12-01', '1894-01-01', '1906-02-01', '1917-08-01',
                      '1928-04-01', '1937-04-01', '1947-05-01', '1958-03-01', '1968-11-01',
                      '1979-12-01', '1989-07-01', '2000-03-01', '2014-04-01', '2024-01-01']

    start_times = pd.to_datetime(start_times_list).to_julian_date()
    max_times = pd.to_datetime(max_times_list).to_julian_date()
    # Calculate cycle lengths
    cycle_length = start_times[1:] - start_times[0:-1]
    # Throw away last cycle start time, so lengths tally.
    start_times = start_times[0:-1]
    max_times = max_times[0:-1]
    # Calculate corresponding solar cycle numbers.
    sc = np.arange(0, len(start_times), dtype='int')
    # Workout sc index of each time.
    idt = np.zeros(time.shape, dtype='int')
    for i, t in enumerate(time):
        idt[i] = np.argwhere(start_times <= t)[-1]

    sc_num = sc[idt]
    sc_phase = 2.0 * np.pi * (time - start_times[idt]) / cycle_length[idt]
    # Get whether rising or falling part of cycle
    sc_state = np.ones(time.shape)
    sc_state[time > max_times[idt]] = -1

    # Also compute the phase of the Hale cycle and state of the Hale cycle.
    # Assume polar fields flip 1-yr after maximum. Ref for this from Lockwood?
    max_times = pd.to_datetime(max_times_list)
    hale_start = max_times + pd.to_timedelta("365.25 days")
    hale_start = hale_start.to_julian_date()
    hale_length = hale_start[1:] - hale_start[:-1]
    hale_start = hale_start[:-1]
    hale = np.arange(0, len(start_times), dtype='int')
    # Workout sc index of each time.
    idt = np.zeros(time.shape, dtype='int')
    for i, t in enumerate(time):
        idt[i] = np.argwhere(hale_start <= t)[-1]

    hale_num = hale[idt]
    hale_phase = 2.0 * np.pi * (time - hale_start[idt]) / hale_length[idt]

    return sc_num, sc_phase, sc_state, hale_num, hale_phase


def select_common_period(df_tuple):
    """
    Function to select the common period from a list of dataframes. The dataframes must have the format produced by the
    load functions such as load_cosmogenic_data and load_aa_data ect.
    :param df_tuple: A tuple of pandas dataframes, all with a common format 'time' column.
    :return df_tuple_out: A tuple containing each of the input dataframes, which have been set to the common overlap
                          period for each of the dataframes. This resets the index of the dataframe from 0:len(df).
    """
    # Get the max and min times of each df
    tmin_list = []
    tmax_list = []
    for df in df_tuple:
        tmin_list.append(df['time'].min())
        tmax_list.append(df['time'].max())

    # Find largest tmin and smallest tmax, to get common overlap period.
    tmin = np.max(tmin_list)
    tmax = np.min(tmax_list)

    # Chopdown each dataframe out, form output data list, to convert to tuple on output.
    df_out = []
    for df in df_tuple:
        # Limit the data to the common span.
        df = df[(df['time'] >= tmin) & (df['time'] <= tmax)]
        # Reset the pandas indices
        df.set_index(np.arange(0, len(df)), inplace=True)
        df_out.append(df)

    return tuple(df_out)


def calc_reconstruction_differences(pcr1, pcr2, ssn, geo):
    """
    Function to calculate the absolute and relative differences between cosmogenic the PCR1 and PCR2 recosntructions of
    heliospheric magnetic field, with the ssn, geo and bst files. This function accepts the matched dataframes of these
    data as produced by load_all_data().
    :param pcr1: DataFrame of PCR1 reconstruction
    :param pcr2: DataFrame of PCR2 reconstruction
    :param ssn: DataFrame of SSN reconstruction
    :param geo: DataFrame of GEO reconstruction
    :return ssn:  DataFrame of SSN reconstruction, with added absolute and relative differences with PCR1 and PCR2
    :return geo:  DataFrame of SSN reconstruction, with added absolute and relative differences with PCR1 and PCR2
    """
    # Absolute differences from PCR1, accounting for errors too (given asymmetrically in the data sets).
    # Add the absolute errors in quadrature for above and below errors?
    # Absolute differences, PCR1
    ssn['d1'] = pcr1['hmf'] - ssn['hmf']
    ssn['d1_e'] = np.sqrt((ssn['hmf_e'] ** 2) + (pcr1['hmf_e'] ** 2))
    ssn['d1_fe'] = ssn['d1_e']/ssn['d1']

    geo['d1'] = pcr1['hmf'] - geo['hmf']
    geo['d1_e'] = np.sqrt((geo['hmf_e'] ** 2) + (pcr1['hmf_e'] ** 2))
    geo['d1_fe'] = geo['d1_e'] / geo['d1']

    # Relative differences, PCR1
    ssn['r1'] = ssn['d1'] / ssn['hmf']
    ssn['r1_fe'] = np.sqrt((ssn['d1_fe'] ** 2) + (ssn['hmf_fe'] ** 2))
    ssn['r1_e'] = ssn['r1_fe']*ssn['r1']

    geo['r1'] = geo['d1'] / geo['hmf']
    geo['r1_fe'] = np.sqrt((geo['d1_fe'] ** 2) + (geo['hmf_fe'] ** 2))
    geo['r1_e'] = geo['r1_fe'] * geo['r1']

    # Absolute differences, PCR2
    ssn['d2'] = pcr2['hmf'] - ssn['hmf']
    ssn['d2_e'] = np.sqrt((ssn['hmf_e'] ** 2) + (pcr2['hmf_e'] ** 2))
    ssn['d2_fe'] = ssn['d1_e'] / ssn['d1']

    geo['d2'] = pcr2['hmf'] - geo['hmf']
    geo['d2_e'] = np.sqrt((geo['hmf_e'] ** 2) + (pcr2['hmf_e'] ** 2))
    geo['d2_fe'] = geo['d1_e'] / geo['d1']

    # Relative differences, PCR2
    ssn['r2'] = ssn['d2'] / ssn['hmf']
    ssn['r2_fe'] = np.sqrt((ssn['d2_fe'] ** 2) + (ssn['hmf_fe'] ** 2))
    ssn['r2_e'] = ssn['r2_fe'] * ssn['r2']

    geo['r2'] = geo['d2'] / geo['hmf']
    geo['r2_fe'] = np.sqrt((geo['d2_fe'] ** 2) + (geo['hmf_fe'] ** 2))
    geo['r2_e'] = geo['r2_fe'] * geo['r2']

    return ssn, geo
    
 
def load_cosmogenic_data():
    """
    Function to load in the near-Earth HMF reconstruction of McCrakcen and Beer 2015, Solar Physics
    doi:10.1007/s11207-015-0777-x
    :return: df1 - DataFrame containing the annual mean PCR1 10Be Reconstruction, including fields:
       time     : Datetime index of observation
       hmf        : BPCR1 HMF reconstruction
       jd       : Julian date of observation
       sc_num   : Solar cycle number
       sc_phase : Solar cycle phase
       sc_state : Rising (+1) or falling (-1) phase of solar cycle
       hale_num   : Hale cycle number
       hale_phase : Hale cycle phase
    :return: df2 - DataFrame containing the annual mean PCR2 10Be Reconstruction, including fields:
       time     : Datetime index of observation
       hmf        : BPCR1 HMF reconstruction
       jd       : Julian date of observation
       sc_num   : Solar cycle number
       sc_phase : Solar cycle phase
       sc_state : Rising (+1) or falling (-1) phase of solar cycle
       hale_num   : Hale cycle number
       hale_phase : Hale cycle phase
    """
    proj_dirs = project_info()
    file_path = proj_dirs['pcr_recon']
    data = pd.read_excel(file_path, header=0, index_col=None)
    col_nam = list(data.columns.values)
    # New column names
    names = ['Yr', 'Yp', 'Be_rat', 'Be_LIS', 'Bpcr1', 'pnm1', 'nm', 'Bpcr2', 'dI']
    new_nam = {col_nam[i]: names[i] for i in range(0, len(names))}
    data = data.rename(columns=new_nam)
    # Keep only years where production date known.
    data = data[~data['Yp'].isnull()]
    # Replace the bad values - some given as "<2.5" - set as NaN
    id_bad = [isinstance(d, basestring) for d in data['Bpcr1']]
    data['Bpcr1'][id_bad] = np.NaN
    data['Bpcr1'].astype(dtype='float')
    # The time intervals in the Cosmic data are not evenly spaced. Interpolate to regular grid. Do for Bpcr1 and Bpcr2
    id_good = ~data['Yp'].isnull() & ~data['Bpcr1'].isnull()
    f1 = spi.interp1d(data['Yp'][id_good], data['Bpcr1'][id_good], kind='cubic')
    f2 = spi.interp1d(data['Yp'][id_good], data['Bpcr2'][id_good], kind='cubic')
    time = np.arange(1775, 1983, 1)
    bpcr1 = f1(time)
    bpcr2 = f2(time)
    df = pd.DataFrame({'time': time, 'bpcr1': bpcr1, 'bpcr2': bpcr2})
    # Convert year to datetime and add in julian date. Set time to middle of calender year.
    t = [pd.datetime(t, 7, 2, 12, 0, 0) for t in df['time']]
    df['time'] = pd.DatetimeIndex(t)
    df['jd'] = pd.DatetimeIndex(t).to_julian_date()
    df.set_index(np.arange(0, len(df)), inplace=True)
    # Add in solar cycle number, phase and rise or fall state
    number, phase, state, hale_num, hale_phase = calc_solar_cycle_phase(df['jd'].values)
    df['sc_num'] = number
    df['sc_phase'] = phase
    df['sc_state'] = state
    df['hale_num'] = hale_num
    df['hale_phase'] = hale_phase

    # Make two copies of the data frame to produce one for PCR1 and PCR2. Do PCR1 first adn PCR2 second.
    df1 = df.copy()
    df1.drop('bpcr2', axis=1, inplace=True)
    df1.rename(columns={"bpcr1": "hmf"}, inplace=True)
    # Add in error limits from the quoted =/-20% error.
    df1['hmf_e'] = 0.2 * df1['hmf']
    df1['hmf_fe'] = 0.2

    df2 = df.copy()
    df2.drop('bpcr1', axis=1, inplace=True)
    df2.rename(columns={"bpcr2": "hmf"}, inplace=True)
    df2['hmf_e'] = 0.2 * df2['hmf']
    df2['hmf_fe'] = 0.2
    return df1, df2


def load_geomagnetic_data():
    """
        Function to load in the best estimate of the geomagnetic HMF reconstruction provided by Owens et al. 2016 JGR
        doi:10.1002/2016JA022529
        :return: DF - DataFrame containing the annual mean geomagnetic HMF reconstruction, including fields:
           time     : Datetime index of observation
           hmf        : Geomagnetic HMF reconstruction
           dhmf_l      : Lower error limit
           dhmf_u      : Upper error limit
           jd       : Julian date of observation
           sc_num   : Solar cycle number
           sc_phase : Solar cycle phase
           sc_state : Rising (+1) or falling (-1) phase of solar cycle
           hale_num   : Hale cycle number
           hale_phase : Hale cycle phase
    """
    proj_dirs = project_info()
    file_path = proj_dirs['geo_recon']
    names = ['time', 'hmf', 'hmf_u', 'hmf_l']
    df = pd.read_csv(file_path, skiprows=17, delim_whitespace=True, header=None, names=names)
    # Add in error increments for error calcs
    df['dhmf_l'] = np.abs(df['hmf_l'] - df['hmf'])
    df['dhmf_u'] = np.abs(df['hmf_u'] - df['hmf'])
    # Average the errors, assume symmetric and gaussian, rescale to 1 sigma.
    df['hmf_e'] = (df.loc[:, ['dhmf_l', 'dhmf_u']].mean(axis=1)) / 1.282
    df['hmf_fe'] = df['hmf_e'] / df['hmf']
    df.drop(['dhmf_l', 'dhmf_u', 'hmf_u', 'hmf_l'], axis=1, inplace=True)
    # Convert year to datetime and add in julian date. Set time to middle of calender year
    t = [pd.datetime(t, 7, 2, 12, 0, 0) for t in df['time']]
    df['time'] = pd.DatetimeIndex(t)
    df['jd'] = pd.DatetimeIndex(t).to_julian_date()
    # Add in solar cycle number, phase and rise or fall state
    number, phase, state, hale_num, hale_phase = calc_solar_cycle_phase(df['jd'].values)
    df['sc_num'] = number
    df['sc_phase'] = phase
    df['sc_state'] = state
    df['hale_num'] = hale_num
    df['hale_phase'] = hale_phase
    return df


def load_sunspot_data():
    """
        Function to load in the best estimate of the sunspot based HMF reconstruction provided by Owens et al. 2016 JGR
        doi:10.1002/2016JA022529
        :return: DF - DataFrame containing the annual mean geomagnetic reconstruction, including fields:
           time     : Datetime index of observation
           hmf        : Sunspot HMF reconstruction
           dhmf_l      : Lower error limit
           dhmf_u      : Upper error limit
           jd       : Julian date of observation
           sc_num   : Solar cycle number
           sc_phase : Solar cycle phase
           sc_state : Rising (+1) or falling (-1) phase of solar cycle
           hale_num   : Hale cycle number
           hale_phase : Hale cycle phase
    """
    proj_dirs = project_info()
    file_path = proj_dirs['ssn_recon']
    names = ['time', 'hmf', 'hmf_u', 'hmf_l']
    df = pd.read_csv(file_path, skiprows=18, delim_whitespace=True, header=None, names=names)
    # Add in error increments for error calcs
    df['dhmf_l'] = np.abs(df['hmf_l'] - df['hmf'])
    df['dhmf_u'] = np.abs(df['hmf_u'] - df['hmf'])
    df['hmf_e'] = (df.loc[:, ['dhmf_l', 'dhmf_u']].mean(axis=1)) / 1.282
    df['hmf_fe'] = df['hmf_e'] / df['hmf']
    df.drop(['dhmf_l', 'dhmf_u', 'hmf_u', 'hmf_l'], axis=1, inplace=True)
    # Convert year to datetime and add in julian date. Set time to middle of calender year
    t = [pd.datetime(t, 7, 2, 12, 0, 0) for t in df['time']]
    df['time'] = pd.DatetimeIndex(t)
    df['jd'] = pd.DatetimeIndex(t).to_julian_date()
    # Add in solar cycle number, phase and rise or fall state
    number, phase, state, hale_num, hale_phase = calc_solar_cycle_phase(df['jd'].values)
    df['sc_num'] = number
    df['sc_phase'] = phase
    df['sc_state'] = state
    df['hale_num'] = hale_num
    df['hale_phase'] = hale_phase
    return df
    
    
def load_sunspot_count_data():
    """
        Function to load in the daily sunspot data from SILSO (version 2.0). Data taken directly from SILSO
        http://www.sidc.be/silso/datafiles
        :return: DF - Dataframe containing the AA index, including fields:
           time     : Datetime index of observation
           R        : Daily sunspot number
           jd       : Julian date of observation
    """
    # Load the data into pandas, use some converters to get the dtypes correct.
    proj_dirs = project_info()
    file_path = proj_dirs['ssn_count_data']
    colnam = ['year', 'month', 'day', 'deciyear', 'R', 'sigma', 'N', 'Q']
    dtype = {'year': 'str', 'month': 'str', 'day': 'str', 'hour': 'str'}
    df = pd.read_csv(file_path, delim_whitespace=True, names=colnam, na_values="-1", dtype=dtype)
    # Times are a mess, calculate datetime for each index and tidy up arrays
    df['time'] = pd.to_datetime(df.year + df.month + df.day, format="%Y%m%d")
    df.drop(['year', 'month', 'day', 'sigma', 'N', 'Q', 'deciyear'], axis=1, inplace=True)
    return df


def load_aa_data():
    """
        Function to load in the 3 hourly AA geomagnetic index. Data taken directly from ISGI
        http://isgi.unistra.fr/indices_aa.php.
        :return: DF - DataFrame containing the AA index , including fields:
           time     : Datetime index of observation
           val        : AA index value
           jd       : Julian date of observation
           sc_num   : Solar cycle number
           sc_phase : Solar cycle phase
           sc_state : Rising (+1) or falling (-1) phase of solar cycle
           hale_num   : Hale cycle number
           hale_phase : Hale cycle phase
    """
    # Load the data into pandas, use some converters to get the dtypes correct.
    proj_dirs = project_info()
    file_path = proj_dirs['aa_data']
    colnam = ['year', 'month', 'day', 'hour', 'val', 'aa_provisional']
    dtype = {'year': 'str', 'month': 'str', 'day': 'str', 'hour': 'str'}
    df = pd.read_csv(file_path, delim_whitespace=True, names=colnam, skiprows=8,
                     na_values='-', dtype=dtype)
    df.drop(['aa_provisional'], axis=1, inplace=True)
    # Times are a mess, calculate datetime for each index and tidy up arrays
    df['time'] = pd.to_datetime(df.year + df.month + df.day + 'T' + df.hour, format="%Y%m%dT%H%M%S")
    df.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)
    df['jd'] = pd.DatetimeIndex(df['time']).to_julian_date()
    # Add in solar cycle number, phase and rise or fall state
    number, phase, state, hale_num, hale_phase = calc_solar_cycle_phase(df['jd'].values)
    df['sc_num'] = number
    df['sc_phase'] = phase
    df['sc_state'] = state
    df['hale_num'] = hale_num
    df['hale_phase'] = hale_phase
    return df


def load_aa_storms(thresh):
    """
        Function to produce a dataframe containing properties of geomagnetic storms defined using a peak above threshold
        method very similar to the Kilpua et al. 2015
        :param thresh: Threshold used to identify storms
        :return: DF - DataFrame containing the AA storms , including fields:
           time     : Datetime index of storm onset
           val        : AA index value
           jd       : Julian date of observation
           sc_num   : Solar cycle number
           sc_phase : Solar cycle phase
           sc_state : Rising (+1) or falling (-1) phase of solar cycle
           hale_num   : Hale cycle number
           hale_phase : Hale cycle phase
    """
    aa = load_aa_data()
    aa_storms = define_storms(aa['time'], aa['val'], thresh, 'above')
    # Calc julian date of each observation.
    aa_storms['jd'] = pd.DatetimeIndex(aa_storms['time']).to_julian_date()
    # Add in solar cycle number, phase and rise or fall state
    number, phase, state, hale_num, hale_phase = calc_solar_cycle_phase(aa_storms['jd'].values)
    aa_storms['sc_num'] = number
    aa_storms['sc_phase'] = phase
    aa_storms['sc_state'] = state
    aa_storms['hale_num'] = hale_num
    aa_storms['hale_phase'] = hale_phase
    return aa_storms


def load_gle_list():
    """
        Function to load in a list of GLEs provided by the NGDC at
        ftp://ftp.ngdc.noaa.gov/STP/SOLAR_DATA/COSMIC_RAYS/ground-level-enhancements/ground-level-enhancements.txt
        :return: DF - DataFrame containing a list of GLE events, including fields:
           time     : Datetime index of GLE onset
           jd       : Julian date of GLE onset
           sc_num   : Solar cycle number of GLE onset
           sc_phase : Solar cycle phase of GLE onset
           sc_state : Rising (+1) or falling (-1) phase of solar cycle
           hale_num   : Hale cycle number
           hale_phase : Hale cycle phase
    """
    # Get names and data types of the GLE data
    proj_dirs = project_info()
    file_path = proj_dirs['gle_data']
    colnam = ['id', 'date', 'datenum', 'baseline']
    dtype = {'id': 'int', 'date': 'str', 'datenum': 'int', 'baseline': 'str'}
    df = pd.read_csv(file_path, sep=',', names=colnam, header=None, dtype=dtype)
    # Convert time and loose useless columns. Calculate julian date of each observation.
    df['time'] = pd.to_datetime(df['date'])
    df.drop(['date', 'datenum', 'baseline'], axis=1, inplace=True)
    df['jd'] = pd.DatetimeIndex(df['time']).to_julian_date()
    # Add in solar cycle number, phase and rise or fall state
    number, phase, state, hale_num, hale_phase = calc_solar_cycle_phase(df['jd'].values)
    df['sc_num'] = number
    df['sc_phase'] = phase
    df['sc_state'] = state
    df['hale_num'] = hale_num
    df['hale_phase'] = hale_phase
    return df


def load_all_data(get_all=False, match_aa=False, match_gle=False, storm_thresh=100):
    """
    Function to load in all of the reconstruction data, AA data, AA storms, GLE list. Only data from the common span of
    the HMF reconstructions is included. A best estimate average of the geo and ssn reconstructions is also returned.
    Can optionally limit the series to the common span with the aa index and/or gle record.
    :param get_all: Bool. Choose whether to only load in the reconstructions (default), or whether to also load in the
                          the AA index, AA storms and GLE database.
    :param match_aa: Bool. Match timespan of parameters to common span including AA index
    :param match_gle: Bool. Match timespan of parameters to common span including GLE list
    :param storm_thresh: AA threshold used to define storms. Defaults to 100, as per Kilpua et al 2015.
    :return: pcr1: DataFrame with PCR1 cosmogenic reconstruction from load_cosmogenic_data.
             pcr2: DataFrame with PCR2 cosmogenic reconstruction from load_cosmogenic_data.
             ssn: DataFrame with sunspot reconstruction from load_sunspot_data.
             geo: DataFrame with geomagnetic reconstruction from load_geomagnetic_data.
             aa: DataFrame with AA index from load_geomagnetic_data.
             aa_storms: DataFrame of geomagnetic storms defined from AA index, using storm_thresh
             gle: DataFrame of GLEs.
    """
    # Check match_aa and match_gle are valid.
    if not isinstance(get_all, bool):
        print('Error: get_all not set to bool. Either True or False. Defaulting to False')
        get_all = False
    if not isinstance(match_gle, bool):
        print('Error: match_gle not set to bool. Either True or False. Defaulting to False')
        match_gle = False
    if not isinstance(storm_thresh, (int, float)):
        print('Error: storm_thresh not set to int or float. Defaulting to 100')
        storm_thresh = 100

    # Load cosmogenic reconstruction
    pcr1, pcr2 = load_cosmogenic_data()

    # Load sunspot reconstruction
    ssn = load_sunspot_data()

    # Load sunspot reconstruction
    geo = load_geomagnetic_data()

    if get_all:
        # Load AA index and AA storms
        aa = load_aa_data()
        aa_storms = load_aa_storms(storm_thresh)

        # Load GLE data
        gle = load_gle_list()

        # Put data on common timebase, depending on whether or not AA  and GLE should be included.
        if match_aa and match_gle:
            # Limit times by both AA and GLE
            (pcr1, pcr2, geo, ssn, aa, aa_storms, gle) = select_common_period((pcr1, pcr2, geo, ssn, aa, aa_storms, gle))
        elif match_aa and ~match_gle:
            # Limit times by AA only
            (pcr1, pcr2, geo, ssn, aa, aa_storms) = select_common_period((pcr1, pcr2, geo, ssn, aa, aa_storms))
        elif ~match_aa and match_gle:
            # Limit times by GLE only
            (pcr1, pcr2, geo, ssn, gle) = select_common_period((pcr1, pcr2, geo, ssn, gle))
    else:
        # Not loading AA, AA storms or gles. Just place recons on common period.
        (pcr1, pcr2, geo, ssn) = select_common_period((pcr1, pcr2, geo, ssn))

    # Now calculate the absolute and relative differences between the GEO and SSN reconstructions
    # with PCR1 and PCR2
    ssn, geo = calc_reconstruction_differences(pcr1, pcr2, ssn, geo)

    # Parse out the relevant DataFrames in a tuple, depending on whether all data requested.
    if get_all:
        out_tuple = tuple([pcr1, pcr2, ssn, geo, aa, aa_storms, gle])
    else:
        out_tuple = tuple([pcr1, pcr2, ssn, geo])

    return out_tuple


def sc_phase_average_recon(phase_bins, df, keys):
    """
    Function to calculate the solar cycle phase average of the reconstructions and differences in a data frame (df)
    :param phase_bins: numpy array giving the edges of the phase bins
    :param df: dataframe with the reconstruction and differences, as output from load_all_data.
    :param keys: List of keys to to columns in df that will be averaged.
    :return: df_phase: Dataframe with phase averaged points, and error bounds.
    """
    # Get blank data frame with appropriate keys for the averaged quantities
    new_keys = [k + '_' + j for k in keys for j in ['avg', 'err']]
    df_avg = pd.DataFrame({k: np.zeros(len(phase_bins)-1) for k in new_keys})
    df_avg['sc_phase'] = np.zeros(len(phase_bins)-1)
    df_avg['n_samples'] = np.zeros(len(phase_bins) - 1)
    # Also make a dictionary to translate between the input keys and output keys for each of avg, el, and eh
    kd = {k: {j: k + '_' + j for j in ['avg', 'err']} for k in keys}

    # Loop over phase bins, calculate average and error of each requested key, populate df_avg
    for i in range(0, len(phase_bins)-1):
        idx = (df['sc_phase'] > phase_bins[i]) & (df['sc_phase'] <= phase_bins[i+1])
        # Get average phase in this bin
        df_avg.loc[i, 'sc_phase'] = df.loc[idx, 'sc_phase'].mean()
        n_points = np.sum(idx)
        df_avg.loc[i, 'n_samples'] = n_points
        for k in keys:
            # Calculate average and error on this key
            df_avg.loc[i, kd[k]['avg']] = df.loc[idx, k].mean()
            df_avg.loc[i, kd[k]['err']] = df.loc[idx, k].std() / np.sqrt(n_points)

    return df_avg


def hale_phase_average_recon(phase_bins, df, keys):
    """
    Function to calculate the solar cycle phase average of the reconstructions and differences in a data frame (df)
    :param phase_bins: numpy array giving the edges of the phase bins
    :param df: dataframe with the reconstruction and differences, as output from load_all_data.
    :param keys: List of keys to to columns in df that will be averaged.
    :return: df_phase: Dataframe with phase averaged points, and error bounds.
    """
    # Get blank data frame with appropriate keys for the averaged quantities
    new_keys = [k + '_' + j for k in keys for j in ['avg', 'err']]
    df_avg = pd.DataFrame({k: np.zeros(len(phase_bins)-1) for k in new_keys})
    df_avg['hale_phase'] = np.zeros(len(phase_bins)-1)
    df_avg['n_samples'] = np.zeros(len(phase_bins) - 1)
    # Also make a dictionary to translate between the input keys and output keys for each of avg, el, and eh
    kd = {k: {j: k + '_' + j for j in ['avg', 'err']} for k in keys}

    # Loop over phase bins, calculate average and error of each requested key, populate df_avg
    for i in range(0, len(phase_bins)-1):
        idx = (df['hale_phase'] > phase_bins[i]) & (df['hale_phase'] <= phase_bins[i+1])
        # Get average phase in this bin
        df_avg.loc[i, 'hale_phase'] = df.loc[idx, 'hale_phase'].mean()
        n_points = np.sum(idx)
        df_avg.loc[i, 'n_samples'] = n_points
        for k in keys:
            # Calculate average and error on this key
            df_avg.loc[i, kd[k]['avg']] = df.loc[idx, k].mean()
            df_avg.loc[i, kd[k]['err']] = df.loc[idx, k].std() / np.sqrt(n_points)

    return df_avg
    

def bootstrap_distributions(data, n_samples, n_iterations):
    """
    Function to bootstrap estimate the sampling distribution of a statistic
    :param data: Float array of data to sample from.
    :param n_samples: Int val of number of samples.
    :param n_iterations: Int val of number of bootstrap iterations
    :return bootstrap_out: DataFrame containing kernel density estimate of the bootstrap distribution
    """
    if not isinstance(n_samples, int):
        print("Error: n_samples should be an integer. Converting to int.")
        n_samples = int(n_samples)

    if n_samples >= len(data):
        print("Error: Requested more samples than exist in data.")

    if not isinstance(n_iterations, int):
        print("Error: n_samples should be an integer. Converting to int.")
        n_iterations = int(n_iterations)

    if n_iterations < 100:
        print("Error: n_iterations = {} . This is very low".format(n_iterations))
        
    # Get sapce to save each bootstrap sample
    boot_sample = np.zeros((n_samples, n_iterations))
    # Also setup space for the kernal density estimates of the bootstraps
    data_support = np.arange(-1.0, 1.0, 0.01)
    boot_pdfs = np.zeros((len(data_support), n_iterations))
    for i in range(n_iterations):
        # Get the bootstrap sample
        boot_sample[:, i] = np.sort(np.random.choice(data, n_samples, replace=False))
        # Do KDE fits
        kde = sps.gaussian_kde(boot_sample[:, i])
        boot_pdfs[:, i] = kde.pdf(data_support)

    return data_support, boot_pdfs, boot_sample

