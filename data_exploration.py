    # (ap)
    #print("saving data...")
    #df.to_csv('raw_data.csv')
    import matplotlib.pyplot as plt
    import numpy as np

    nan_index = df['LOG_CONCENTRATION'].isnull() & df['GROWTH'].isnull()
    df = df.loc[ ~(nan_index) ]
    print('A total of {} drug observations were removed from original dataset (when LCONC or GROWTH is nan).'.format(nan_index.sum()))

    # Save LCONC hist
    plt.figure(); plt.hist(df['LOG_CONCENTRATION'], bins=30)
    plt.xlabel('Bins'); plt.ylabel('Freq')
    plt.title('LCONC histogram (raw) [{:.2f}, {:.2f}]'.format(df['LOG_CONCENTRATION'].min(), df['LOG_CONCENTRATION'].max()))
    plt.grid("on")
    plt.savefig('lconc_hist_raw')

    # Save GRWOTH hist
    plt.figure(); plt.hist(df['GROWTH'], bins=30)
    plt.xlabel('Bins'); plt.ylabel('Freq')
    plt.title('GROWTH histogram (raw) [{:.2f}, {:.2f}]'.format(df['GROWTH'].min(), df['GROWTH'].max()))
    plt.grid('on')
    plt.savefig('growth_hist_raw')

    # Check some outliers
    print('Check for outliers (consider remove and analyze separetly).')
    low_conc_val = -7
    high_conc_val = -3
    low_conc = df.loc[ df['LOG_CONCENTRATION'] <= low_conc_val, 'GROWTH']
    high_conc = df.loc[ df['LOG_CONCENTRATION'] >= high_conc_val, 'GROWTH']
    print("(low conc) LCONC<={}, GROWTH_range=[{:.2f}, {:.2f}]".format(low_conc_val, low_conc.min(), low_conc.max()))
    print("(high conc) LCONC>={}, GROWTH_range=[{:.2f}, {:.2f}]".format(high_conc_val, high_conc.min(), high_conc.max()))
    print("(low conc) LCONC<={}, GROWTH_range<=-0.2, total: {} obs ==> very effective?".format(low_conc_val, (low_conc<=-0.2).sum() ))
    print("(high conc) LCONC>={}, GROWTH_range>=0.2, total: {} obs ==> not effective?".format(high_conc_val, (high_conc>=0.2).sum() ))


    # Save LCONC vs GROWTH (can't plot so many points)
#    plt.figure(); plt.scatter(df['LOG_CONCENTRATION'], df['GROWTH'])
#    plt.xlabel('LCONC'); plt.ylabel('GROWTH')
#    plt.title('GROWTH histogram (raw) [{:.2f}, {:.2f}]'.format(df['GROWTH'].min(), df['GROWTH'].max()))
#    plt.grid('on')
#    plt.savefig('LCONC_vs_GROWTH')

    # Plot distribution of LCONC levels
    LCONC_levels = [-8, -7, -6, -5, -4, -3]
    level_elements = []
    xticklabels = []
    for i in range(len(LCONC_levels)+1):
        print('i={}'.format(i))
        if i==0:
            dd = df.loc[df['LOG_CONCENTRATION'] <= LCONC_levels[i], 'GROWTH']
            xticklabels.append('(,{}]'.format(LCONC_levels[i]))
        elif i==len(LCONC_levels):
            dd = df.loc[df['LOG_CONCENTRATION'] >= LCONC_levels[i-1], 'GROWTH']
            xticklabels.append('({},)'.format(LCONC_levels[i-1]))
        else:
            dd = df.loc[(df['LOG_CONCENTRATION'] >= LCONC_levels[i-1]) & (df['LOG_CONCENTRATION'] <= LCONC_levels[i]), 'GROWTH']
            xticklabels.append('({},{}]'.format(LCONC_levels[i-1], LCONC_levels[i]))
        level_elements.append(len(dd))

    plt.figure()
    pos = np.arange(len(LCONC_levels)+1)
    plt.bar( pos, level_elements, align='center' )
    plt.xticks(pos, xticklabels)
    plt.savefid('hist_lconc_levels')
    # (ap)