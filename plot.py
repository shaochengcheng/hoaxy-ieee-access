import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from matplotlib_venn import venn2
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from os.path import join

from analysis import *
from correlation import *
import logging
import json
from scipy.stats import sem

DATA_DIR = 'data0104'
# fake
C1 = '#1F78B4'
# snopes
C2 = '#FF7F00'
try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse


def ccdf(s):
    """
    Parameters:
        `s`, series, the values of s should be variable to be handled
    Return:
        a new series `s`, index of s will be X axis (number), value of s
        will be Y axis (probability)
    """
    s = s.copy()
    s = s.sort_values(ascending=True, inplace=False)
    s.reset_index(drop=True, inplace=True)
    n = len(s)
    s.drop_duplicates(keep='first', inplace=True)
    X = s.values
    Y = [n - i for i in s.index]
    return pd.Series(data=Y, index=X) / n


def get_site_name(url):
    hostname = urlparse(url).hostname
    if hostname.startswith('www.'):
        hostname = hostname[4:]
    names = hostname.split('.')
    return names[0]


def pd_vs_ftw(fn='t_snopes_mv_final.csv'):
    fn = join(DATA_DIR, fn)
    output = 'Fig1-pd-vs-ftw.pdf'
    df = pd.read_csv(fn, parse_dates=['f_pd', 's_pd', 's_ftw', 'f_ftw'])
    df['site_name'] = [get_site_name(url) for url in df.fake]
    df = df.sort_values('site_name')
    df.reset_index(inplace=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    s = (df.s_ftw - df.s_pd) / np.timedelta64(1, 'h')
    ax.plot(
        s,
        color=C2,
        linestyle='None',
        marker='d',
        label='snopes',
        alpha=0.8,
        markersize=9)
    grouped = df.groupby('site_name')
    colors = plt.cm.tab20(np.linspace(0, 1, len(grouped.groups)))
    i = 0
    for name, group in grouped:
        s = (group.f_ftw - group.f_pd) / np.timedelta64(1, 'h')
        ax.plot(
            s,
            color=colors[i],
            linestyle='None',
            marker='o',
            markersize=9.5,
            label=name,
            alpha=0.8)
        i = i + 1
    plt.xlim(-3, 75)
    plt.subplots_adjust(left=0.125, right=0.985, top=0.75, bottom=0.105)
    plt.legend(
        fontsize=9,
        bbox_to_anchor=(0.0, 1.04, 1.0, 0.42),
        loc=3,
        ncol=3,
        mode="expand",
        borderaxespad=0.)
    plt.yscale('symlog')
    plt.axhline(y=24, ls='--', linewidth=0.8, color='black')
    plt.axhline(y=-24, ls='--', linewidth=0.8, color='black')
    plt.xlabel('Article pair index')
    plt.ylabel('$\lambda_1$ (hour)')
    plt.savefig(output)


def timeline(data_dir=DATA_DIR,
             fn1='t_snopes_mv_fake_tweet.csv',
             fn2='t_snopes_mv_snopes_tweet.csv'):
    output = 'Fig3a-timeline.pdf'
    fn1 = join(data_dir, fn1)
    fn2 = join(data_dir, fn2)
    df1 = pd.read_csv(fn1, parse_dates=['tweet_created_at'])
    df2 = pd.read_csv(fn2, parse_dates=['tweet_created_at'])
    df1.set_index('tweet_created_at', inplace=True)
    df2.set_index('tweet_created_at', inplace=True)
    s1 = df1.groupby(
        pd.Grouper(freq='1D')).tweet_id.nunique().rename('Fake news')
    s2 = df2.groupby(pd.Grouper(freq='1D')).tweet_id.nunique().rename('Snopes')
    df = pd.concat([s1, s2], axis=1)
    # plot
    f, ax = plt.subplots(figsize=(2.4, 2))
    df.plot(ax=ax, logy=True, color=[C1, C2])
    ax.set_xlabel('')
    ax.set_ylabel('Tweets volume')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.subplots_adjust(left=0.235, right=0.94, bottom=0.23, top=0.85)
    plt.legend(
        fontsize=9,
        bbox_to_anchor=(-0.175, 1.04, 1.2, 0.13),
        loc=3,
        ncol=2,
        mode="expand",
        borderaxespad=0.)
    plt.savefig(output)


def timeline2(data_dir=DATA_DIR,
              fn1='t_snopes_mv_fake_tweet.csv',
              fn2='t_snopes_mv_snopes_tweet.csv'):
    output = 'Fig3-timeline.pdf'
    fn1 = join(data_dir, fn1)
    fn2 = join(data_dir, fn2)
    df1 = pd.read_csv(fn1, parse_dates=['tweet_created_at'])
    df2 = pd.read_csv(fn2, parse_dates=['tweet_created_at'])
    df1.set_index('tweet_created_at', inplace=True)
    df2.set_index('tweet_created_at', inplace=True)
    s1 = df1.groupby(
        pd.Grouper(freq='1D')).tweet_id.nunique().rename('Fake news')
    s2 = df2.groupby(pd.Grouper(freq='1D')).tweet_id.nunique().rename('Snopes')
    df = pd.concat([s1, s2], axis=1)
    # plot
    f, ax = plt.subplots(figsize=(6, 4))
    df.plot(ax=ax, logy=True, color=[C1, C2])
    ax.set_xlabel('')
    ax.set_ylabel('Tweets volume')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)


def tsa_sma_hourly(ts, window=24, center=True, drop_na=True):
    sma = ts.rolling(window, center=center).mean()
    if drop_na is True:
        sma = sma[sma.notnull()]
    return sma


def tsa_ccf(s1, s2, max_lag=15, do_sma=False):
    # ts1 keep fixed, shift ts2
    # positive lag, shift ts2 to right,
    #   in time series analysis, ts2 went ahead of ts1
    # negative lag, shift ts2 to left,
    #   in time series analysis, ts2 left behind ts1
    if do_sma is True:
        s1 = tsa_sma_hourly(s2)
        s2 = tsa_sma_hourly(s2)
        y = ccf(s1, s2)
    else:
        y = ccf(s1, s2)
    y_half = len(y) // 2
    x = range(-y_half, y_half + 1)
    full = pd.Series(y, index=x)
    return full.loc[-max_lag:max_lag]


def timeline_ccf(data_dir=DATA_DIR,
                 fn1='t_snopes_mv_snopes_tweet.csv',
                 fn2='t_snopes_mv_fake_tweet.csv',
                 max_lag=7,
                 freq='1D',
                 do_sma=False):
    output = 'Fig3b-timeline-ccf.pdf'
    fn1 = join(data_dir, fn1)
    fn2 = join(data_dir, fn2)
    df1 = pd.read_csv(fn1, parse_dates=['tweet_created_at'])
    df2 = pd.read_csv(fn2, parse_dates=['tweet_created_at'])
    df1.set_index('tweet_created_at', inplace=True)
    df2.set_index('tweet_created_at', inplace=True)
    s1 = df1.groupby(pd.Grouper(freq=freq)).tweet_id.nunique().rename('snopes')
    s2 = df2.groupby(
        pd.Grouper(freq=freq)).tweet_id.nunique().rename('fake news')
    s1, s2 = s1.align(s2, join='inner')
    ccf = tsa_ccf(s1, s2, max_lag=max_lag, do_sma=do_sma)
    mx = ccf.idxmax()
    my = ccf.max()
    # plot
    f, ax = plt.subplots(figsize=(2., 1.6))
    ax.plot(ccf.index, ccf.values, color='k')
    data_to_axis = ax.transData + ax.transAxes.inverted()
    amx, amy = data_to_axis.transform((mx, my))
    ax.axvline(x=mx, ymax=amy - 0.15, linestyle='dotted', color='k')
    ax.text(
        x=mx,
        y=my + 0.06,
        s=r'$({:.2f}, {:.2f})$'.format(mx, my),
        horizontalalignment='center',
        verticalalignment='top')
    ax.set_ylim([0, 0.38])
    if freq == '1H':
        ax.set_xlabel('Lag in hours')
    else:
        ax.set_xlabel('Lag in days')
    ax.set_ylabel(r'Pearson coef. $\rho$')
    print('Lagest point (x, y) is: ({:.2f}, {:.2f})'.format(mx, my))
    plt.tight_layout(pad=0.05)
    plt.savefig(output)


def timeline_ccf2(data_dir=DATA_DIR,
                  fn1='t_snopes_mv_snopes_tweet.csv',
                  fn2='t_snopes_mv_fake_tweet.csv',
                  max_lag=7,
                  freq='1D',
                  do_sma=False):
    output = 'Fig4-timeline-ccf.pdf'
    fn1 = join(data_dir, fn1)
    fn2 = join(data_dir, fn2)
    df1 = pd.read_csv(fn1, parse_dates=['tweet_created_at'])
    df2 = pd.read_csv(fn2, parse_dates=['tweet_created_at'])
    df1.set_index('tweet_created_at', inplace=True)
    df2.set_index('tweet_created_at', inplace=True)
    s1 = df1.groupby(pd.Grouper(freq=freq)).tweet_id.nunique().rename('snopes')
    s2 = df2.groupby(
        pd.Grouper(freq=freq)).tweet_id.nunique().rename('fake news')
    s1, s2 = s1.align(s2, join='inner')
    ccf = tsa_ccf(s1, s2, max_lag=max_lag, do_sma=do_sma)
    mx = ccf.idxmax()
    my = ccf.max()
    # plot
    f, ax = plt.subplots(figsize=(4, 3))
    ax.plot(ccf.index, ccf.values, color='k')
    data_to_axis = ax.transData + ax.transAxes.inverted()
    amx, amy = data_to_axis.transform((mx, my))
    ax.axvline(x=mx, ymax=amy - 0.15, linestyle='dotted', color='k')
    ax.text(
        x=mx,
        y=my + 0.04,
        s=r'$({:.2f}, {:.2f})$'.format(mx, my),
        horizontalalignment='center',
        verticalalignment='top')
    ax.set_ylim([0, 0.38])
    if freq == '1H':
        ax.set_xlabel('Lag in hours')
    else:
        ax.set_xlabel('Lag in days')
    ax.set_ylabel(r'Pearson coef. $\rho$')
    print('Lagest point (x, y) is: ({:.2f}, {:.2f})'.format(mx, my))
    plt.tight_layout()
    plt.savefig(output)


def prepare_plot_frac_over_total(f1, f2, p):
    dfa1 = prepare_df_tweet_timeline(f1, parse_dates=False)
    dfa2 = prepare_df_tweet_timeline(f2, parse_dates=False)
    dft1 = tt_top_users_df(dfa1, p)
    dft2 = tt_top_users_df(dfa2, p)
    sa1 = tt_count_tweet_type(dfa1, quote_mode=None)
    sa1 = sa1[:-1] / sa1['total']
    sa2 = tt_count_tweet_type(dfa2, quote_mode=None)
    sa2 = sa2[:-1] / sa2['total']
    st1 = tt_count_tweet_type(dft1, quote_mode=None)
    st1 = st1[:-1] / st1['total']
    st2 = tt_count_tweet_type(dft2, quote_mode=None)
    st2 = st2[:-1] / st2['total']

    return pd.Panel(
        dict(
            all=pd.DataFrame(
                [(sa1['origin'], sa2['origin']),
                 (sa1['retweet'], sa2['retweet']),
                 (sa1['reply'], sa2['reply'])],
                columns=['fake news', 'snopes'],
                index=['origin', 'retweet', 'reply']),
            top=pd.DataFrame(
                [(st1['origin'], st2['origin']),
                 (st1['retweet'], st2['retweet']),
                 (st1['reply'], st2['reply'])],
                columns=['fake news', 'snopes'],
                index=['origin', 'retweet', 'reply'])))


def tw_type_share1(data_dir=DATA_DIR,
                   fn1='t_snopes_mv_fake_tweet.csv',
                   fn2='t_snopes_mv_snopes_tweet.csv',
                   p=0.1):
    """
    Plot shares of different tweet types of
        snopes vs fake
    --------------------------------------------------------------------
    data_dir, string, directory where you put the data
    f1, string, file path of snopes tweet timeline file
    f2, string, file path of fake tweet timeline file
    p, float, threshold of top active user
    """
    output = 'Fig5a-tweet-share.pdf'
    fn1 = join(data_dir, fn1)
    fn2 = join(data_dir, fn2)
    # x-offset of bar groups
    x_offset = 0
    # number of groups (number of bar groups)
    n_groups = 3
    # size of figure
    f_size = (2.5, 1.8)
    # first bar position for each group
    x_index = np.arange(n_groups) + x_offset
    # parameters of bars
    b_kwargs = dict(width=0.2)

    data = prepare_plot_frac_over_total(fn1, fn2, p)
    data_all = data['all']

    # plots
    fig, ax = plt.subplots(figsize=f_size)
    rects1 = ax.bar(
        x_index, data_all['fake news'], hatch='//', color=C1, **b_kwargs)
    rects2 = ax.bar(
        x_index + b_kwargs['width'], data_all['snopes'], color=C2, **b_kwargs)
    ax.set_xticks(x_index)
    ax.set_xticklabels(['Origin', 'Retweet', 'Reply'])
    ax.set_ylim([0, 0.8])
    plt.subplots_adjust(left=0.16, right=0.98, bottom=0.128, top=0.84)
    ax.legend(
        (rects1[0], rects2[0]), ('Fake news', 'Snopes'),
        fontsize=8.5,
        bbox_to_anchor=(0, 1.04, 1., 0.104),
        loc=3,
        ncol=2,
        mode="expand",
        borderaxespad=0.)
    ax.tick_params(
        axis='both', which='both', bottom='off', top='off', right='off')
    # plt.tight_layout(pad=0.1, h_pad=0)
    plt.savefig(output)


def tw_type_share(data_dir=DATA_DIR,
                  fn1='t_snopes_mv_fake_tweet.csv',
                  fn2='t_snopes_mv_snopes_tweet.csv',
                  p=0.1):
    """
    Plot shares of different tweet types of
        snopes vs fake
    --------------------------------------------------------------------
    data_dir, string, directory where you put the data
    f1, string, file path of snopes tweet timeline file
    f2, string, file path of fake tweet timeline file
    p, float, threshold of top active user
    """
    output = 'Fig6-tweet-share.pdf'
    fn1 = join(data_dir, fn1)
    fn2 = join(data_dir, fn2)
    # x-offset of bar groups
    x_offset = 0
    # number of groups (number of bar groups)
    n_groups = 3
    # size of figure
    f_size = (6, 3)
    # first bar position for each group
    x_index = np.arange(n_groups) + x_offset
    # parameters of bars
    b_kwargs = dict(width=0.2)

    data = prepare_plot_frac_over_total(fn1, fn2, p)
    data_all = data['all']
    data_top = data['top']

    # plots
    fig = plt.figure(figsize=f_size)
    gs = gridspec.GridSpec(2, 2, height_ratios=[.2, 5])
    ax0 = plt.subplot(gs[0, :])
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[1, 1])

    rects1 = ax1.bar(
        x_index, data_all['fake news'], hatch='//', color=C1, **b_kwargs)
    rects2 = ax1.bar(
        x_index + b_kwargs['width'], data_all['snopes'], color=C2, **b_kwargs)
    ax1.set_xlabel('(a) All')
    ax1.set_xticks(x_index)
    ax1.set_xticklabels(['Origin', 'Retweet', 'Reply'])
    ax1.set_ylim([0, 0.8])
    rects1 = ax2.bar(
        x_index, data_top['fake news'], hatch='//', color=C1, **b_kwargs)
    rects2 = ax2.bar(
        x_index + b_kwargs['width'], data_top['snopes'], color=C2, **b_kwargs)
    ax2.set_xlabel('(b) Top active accounts')
    ax2.set_xticks(x_index)
    ax2.set_xticklabels(['Origin', 'Retweet', 'Reply'])
    ax2.set_yticklabels([])
    ax2.set_ylim([0, 0.8])
    ax1.tick_params(
        axis='both', which='both', bottom='off', top='off', right='off')
    ax2.tick_params(
        axis='both', which='both', bottom='off', top='off', right='off')
    ax0.legend(
        (rects1[0], rects2[0]), ('Fake news', 'Snopes'),
        bbox_to_anchor=(0., -3.2, 1., .1),
        borderaxespad=0.,
        ncol=2,
        mode='expand',
        loc=3,
        fontsize=10)
    ax0.tick_params(
        axis='both', which='both', bottom='off', top='off', right='off')
    ax0.axis('off')
    plt.tight_layout(pad=0.1, w_pad=1.2, h_pad=0.2)
    plt.savefig(output)


def overlapped_users_venn(data_dir=DATA_DIR,
                          fn1='t_ftw_snopes_20160930.csv',
                          fn2='t_ftw_fake_20160930.csv'):
    output = join(BASE_DIR, 'plots', 'user_overlapped.pdf')
    fn1 = join(data_dir, fn1)
    fn2 = join(data_dir, fn2)
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    s1 = set(df1['tweet_user_id'])
    s2 = set(df2['tweet_user_id'])
    plt.figure(figsize=(4, 3))
    venn2([s1, s2], set_labels=('Snopes', 'Fake'))
    plt.savefig(output)


def tt_top_stories(fake_df, ass_df, min_n_fake_tweets=1000):
    merged_df = fake_df.merge(ass_df, on='fake', how='inner')
    n_df = merged_df.groupby('snopes').apply(lambda g: g['tweet_id'].nunique())
    top = n_df.loc[n_df >= min_n_fake_tweets].index
    ass_df.set_index('snopes', inplace=True)
    ass_df = ass_df.loc[top]
    return ass_df.reset_index()


def tt_top_stories_df(f1, f2, f3, min_n_fake_tweets=1000):
    """
    Return top rank (by count number of fake tweets) stories pairs.
    --------------------------------------------------------------------
    Parameters:
        f1, string, file path of snopes tweets timeline `t_ftw_snopes`
        f2, string, file path of fake tweets timeline `t_ftw_fake`
        f3, string, file path of association table `t_ass_date_a`
        min_n_fake_tweets, int, threshold to filter top rank stories
    Return:
        DataFrame, containing paired tweet timeline
    """
    df1 = pd.read_csv(f1, parse_dates=['tweet_created_at'])
    df1 = df1[['tweet_created_at', 'tweet_id', 'snopes']]
    df2 = pd.read_csv(f2, parse_dates=['tweet_created_at'])
    df2 = df2[['tweet_created_at', 'tweet_id', 'fake']]
    df3 = pd.read_csv(f3)

    df3 = tt_top_stories(df2, df3, min_n_fake_tweets)
    df1.set_index('snopes', inplace=True)
    df1 = df1.loc[df3['snopes'].unique()]
    df2.set_index('fake', inplace=True)
    df2 = df2.loc[df3['fake'].unique()]

    return df1, df2, df3


def tt_top_stories_one_group(df1, df2, df3, g_index=None, g_name=None):
    df3_snopes_index = df3.set_index('snopes')
    df3 = df3.groupby('snopes')
    if g_name is not None:
        if g_name not in df3['snopes']:
            raise Exception('%s not in top snopes URLs', g_name)
        else:
            df1 = df1.loc[g_name]
            fakes = df3_snopes_index[g_name]
            return df1, [df2.loc[fake] for fake in fakes]
    elif g_index is not None:
        for i, g in enumerate(df3):
            if g_index == i:
                print(g[0])
                print(g[1]['fake'].tolist())
                df1 = df1.loc[g[0]]
                fakes = g[1]['fake']
                return df1, [df2.loc[fake] for fake in fakes]


def story_pair_tweets_volumn(data_dir=DATA_DIR,
                             fn1='t_ftw_snopes_20160930.csv',
                             fn2='t_ftw_fake_20160930.csv',
                             fn3='t_ass_date.csv',
                             min_n_fake_tweets=1000,
                             g_index=3,
                             g_name=None,
                             rule='D',
                             logy=True):
    output = 'Fig2-story-timeline-g3.pdf'
    fn1 = join(data_dir, fn1)
    fn2 = join(data_dir, fn2)
    fn3 = join(data_dir, fn3)
    figsize = (4, 3)

    df1, df2, df3 = tt_top_stories_df(fn1, fn2, fn3, min_n_fake_tweets)
    ts1, tss2 = tt_top_stories_one_group(
        df1, df2, df3, g_index=g_index, g_name=g_name)
    ts1 = ts1.set_index('tweet_created_at').iloc[:, 0]
    ts1 = ts1.resample(rule).count()
    tss2 = [ts2.set_index('tweet_created_at').iloc[:, 0] for ts2 in tss2]
    tss2 = [ts2.resample(rule).count() for ts2 in tss2]
    min_dt = min(ts1.index.min(), min([ts2.index.min() for ts2 in tss2]))
    max_dt = min(ts1.index.max(), max([ts2.index.max() for ts2 in tss2]))
    outter_index = pd.date_range(start=min_dt, end=max_dt, freq=rule)
    ts1 = ts1.reindex(index=outter_index)
    tss2 = [ts2.reindex(index=outter_index) for ts2 in tss2]
    if logy is True:
        ts1[ts1 == 0] = 1
        ts1 = ts1.fillna(1)
    else:
        ts1 = ts1.fillna(0)
    # plot
    f, ax = plt.subplots(figsize=figsize)
    for ts2 in tss2:
        if logy is True:
            ts2[ts2 == 0] = 1
            ts2 = ts2.fillna(1)
        else:
            ts2 = ts2.fillna(0)
        ts2.plot(ax=ax, logy=logy, label='Fake news, URL1', color=C1)
    ts1.plot(ax=ax, logy=logy, label='Snopes, URL2', color=C2)
    ax.set_ylabel('Tweets volume')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)


def delay_dist_hist(fn='t_snopes_match_delay0104.csv'):
    fn = join(DATA_DIR, fn)
    output = join(BASE_DIR, 'plots', 'pd_delay_dist_hist.pdf')
    figsize = (4, 3)

    df = pd.read_csv(fn, parse_dates=['f_pd', 's_pd'])
    pd_delta = df['s_pd'] - df['f_pd']
    s = pd_delta / np.timedelta64(1, 'h')
    print(len(s))
    s = s[s > 0]
    print(len(s))
    fig, ax = plt.subplots(figsize=figsize)
    w = np.ones_like(s) / len(s)
    bins = np.logspace(-1, 5, 7)
    ax.hist(s.tolist(), bins=bins, weights=w, color=C1)
    ax.set_xscale("log")
    plt.xlabel('Delay (hour)')
    plt.ylabel('Percentage')
    plt.tight_layout()
    plt.savefig(output)


def delay_dist_bar(fn='t_snopes_match_delay0104.csv'):
    fn = join(DATA_DIR, fn)
    output = join(BASE_DIR, 'plots', 'pd_delay_dist_bar.pdf')
    figsize = (4, 3)

    df = pd.read_csv(fn, parse_dates=['f_pd', 's_pd'])
    delta_s = df['s_pd'] - df['f_pd']
    delta_s.index = df['s_pd']
    delta_s = delta_s / np.timedelta64(1, 'h')
    delta_s = delta_s[delta_s > 0]
    s_cat = [
        delta_s[delta_s < 24], delta_s[(24 <= delta_s) & (delta_s < 24 * 7)],
        delta_s[(24 * 7 <= delta_s) & (delta_s < 24 * 30)],
        delta_s[(24 * 30 <= delta_s) & (delta_s < 24 * 365)],
        delta_s[delta_s >= 24 * 365]
    ]

    s_bar_names = ['Day', 'Week', 'Month', 'Year', '>Year']
    bar_width = 0.5
    bar_ind = np.arange(5) + 0.3
    bar_v = [x.count() for x in s_cat]
    # bar_mean = [x.mean() for x in s_cat]
    # bar_std = [x.std() for x in s_cat]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(bar_ind, bar_v, bar_width, color='y')
    plt.tick_params(
        axis='both', which='both', bottom='off', top='off', right='off')
    plt.ylabel(r'Number of stories')
    plt.xlabel(r'Delay')
    plt.xticks(bar_ind + 0.2, s_bar_names)
    plt.tight_layout()
    plt.savefig(output)


def volume_dist(fn='t_snopes_match_volume0104.csv'):
    fn = join(DATA_DIR, fn)
    output = join(BASE_DIR, 'plots', 'v_fake_dist.pdf')
    figsize = (4, 3)

    df = pd.read_csv(fn, parse_dates=['f_pd', 's_pd'])
    s = df['f_n']
    fig, ax = plt.subplots(figsize=figsize)
    w = np.ones_like(s) / len(s)
    bins = np.logspace(1, 5, 5)
    # ax.hist(s.tolist(), weights=w, color=C1)
    ax.hist(s.tolist(), bins=bins, weights=w, color=C1)
    ax.set_xscale("log")
    plt.xlabel('Number of Fake tweet')
    plt.ylabel('Percentage')
    plt.tight_layout()
    plt.savefig(output)


# def v_snopes_vs_v_fake_scatter(fn='t_ass_date_a.csv'):
#     fn = join(DATA_DIR, fn)
#     output = join(BASE_DIR, 'plots', 'v_snopes_vs_v_fake_scatter.pdf')
#     figsize = (4, 3)

#     df = pd.read_csv(fn, parse_dates=['f_pd', 's_pd'])
#     df = df.loc[df.f_pd > '2016-05-16 05:54:15']
#     print(len(df))
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.scatter(df['s_n'], df['f_n'])
#     plt.xlabel('Number of tweet sharing snopes')
#     plt.ylabel('Number of tweet sharing fake')
#     plt.tight_layout()
#     plt.savefig(output)


def survival_by_lag_KM(fn='t_snopes_match_volume0104.csv'):
    fn = join(DATA_DIR, fn)
    output = 'Fig4-survival-KM.pdf'
    figsize = (4.5, 1.8)

    df = pd.read_csv(fn, parse_dates=['f_pd', 's_pd'])
    T1 = (df.s_pd - df.f_pd) / np.timedelta64(1, 'h')
    E1 = np.ones(len(T1))
    kmf1 = KaplanMeierFitter()
    kmf1.fit(T1, E1)
    T2 = df.f_n.values
    E2 = np.ones(len(T2))
    kmf2 = KaplanMeierFitter()
    kmf2.fit(T2, E2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    kmf1.plot(ax=ax1)
    # ax1.axvline(x=24, ymax=0.32, linestyle='dotted', color='k')
    # ax1.axhline(y=0.6, xmax=0.05, linestyle='dotted', color='k')
    ax1.text(
        x=15 * 24,
        y=0.32,
        s=r'(7x24, 0.32)',
        horizontalalignment='left',
        verticalalignment='center')
    ax1.set_xlabel(r'Delay, $\lambda_2$ (hour)')
    ax1.set_ylabel(r'$Pr$')
    ax1.legend_.remove()
    kmf2.plot(ax=ax2, label="KM, volume $v$")
    # ax2.axvline(x=800, ymax=0.32, linestyle='dotted', color='k')
    # ax2.axhline(y=0.32, xmax=0.04, linestyle='dotted', color='k')
    ax2.text(
        x=3000,
        y=0.32,
        s=r'(800, 0.32)',
        horizontalalignment='left',
        verticalalignment='center')
    ax2.set_xlabel(r'Tweets volume, $v$')
    ax2.legend_.remove()
    plt.tight_layout(w_pad=1)
    plt.savefig(output)


def survival_by_lag_KM2(fn='t_snopes_match_volume0104.csv'):
    fn = join(DATA_DIR, fn)
    output = 'Fig5-survival-KM.pdf'
    figsize = (7, 3)

    df = pd.read_csv(fn, parse_dates=['f_pd', 's_pd'])
    T1 = (df.s_pd - df.f_pd) / np.timedelta64(1, 'h')
    E1 = np.ones(len(T1))
    kmf1 = KaplanMeierFitter()
    kmf1.fit(T1, E1)
    T2 = df.f_n.values
    E2 = np.ones(len(T2))
    kmf2 = KaplanMeierFitter()
    kmf2.fit(T2, E2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    kmf1.plot(ax=ax1)
    ax1.axvline(x=7 * 24, ymax=0.32, linestyle='dotted', color='k')
    ax1.axhline(y=0.32, xmax=0.05, linestyle='dotted', color='k')
    ax1.text(
        x=15 * 24,
        y=0.32,
        s=r'(7x24, 0.32)',
        horizontalalignment='left',
        verticalalignment='center')
    ax1.set_xlabel(r'Delay, $\lambda_2$ (hour)')
    ax1.set_ylabel(r'$Pr$')
    ax1.legend_.remove()
    kmf2.plot(ax=ax2, label="KM, volume $v$")
    ax2.axvline(x=800, ymax=0.32, linestyle='dotted', color='k')
    ax2.axhline(y=0.32, xmax=0.04, linestyle='dotted', color='k')
    ax2.text(
        x=3000,
        y=0.32,
        s=r'(800, 0.32)',
        horizontalalignment='left',
        verticalalignment='center')
    ax2.set_xlabel(r'Tweets volume, $v$')
    ax2.legend_.remove()
    plt.tight_layout(w_pad=1)
    plt.savefig(output)


def bot_score_dist(fn1='fake_bot_score.json',
                   fn2='snopes_bot_score.json',
                   nbins=20):
    fn1 = join(DATA_DIR, fn1)
    fn2 = join(DATA_DIR, fn2)
    output = 'Fig5b-bot-score.pdf'
    with open(fn1) as f1:
        r1 = json.load(f1)
    with open(fn2) as f2:
        r2 = json.load(f2)
    s1 = []
    s2 = []
    for user_id, b in r1:
        if 'score' in b:
            s1.append(b['score'])
    for user_id, b in r2:
        if 'score' in b:
            s2.append(b['score'])
    logger.info('Number of fake accounts: %s', len(s1))
    logger.info('Number of snopes accounts: %s', len(s2))
    s1 = np.array(s1)
    s2 = np.array(s2)
    logger.info('Fake, ratio larger than 0.5: %s', (s1 > 0.5).sum() / len(s1))
    logger.info('Snopes, ratio larger than 0.5: %s',
                (s2 > 0.5).sum() / len(s2))
    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    bins = np.linspace(0, 1, nbins + 1)
    w1 = np.ones_like(s1) / len(s1)
    w2 = np.ones_like(s2) / len(s2)
    ax.hist(s1, bins, weights=w1, alpha=0.75, label='Fake news', color=C1)
    ax.hist(s2, bins, weights=w2, alpha=0.75, label='Snopes', color=C2)
    plt.subplots_adjust(left=0.2, right=0.98, bottom=0.23, top=0.845)
    ax.legend(
        fontsize=8.5,
        bbox_to_anchor=(-0.06, 1.04, 1.06, 0.104),
        loc=3,
        ncol=2,
        mode="expand",
        borderaxespad=0.)
    # plt.legend(loc='upper right', fontsize=8.5)
    plt.xlabel('Bot score')
    plt.ylabel('Frequency')
    plt.savefig(output)


def bot_score_dist2(fn1='fake_bot_score.json',
                    fn2='snopes_bot_score.json',
                    nbins=20):
    fn1 = join(DATA_DIR, fn1)
    fn2 = join(DATA_DIR, fn2)
    output = 'Fig7-bot-score.pdf'
    with open(fn1) as f1:
        r1 = json.load(f1)
    with open(fn2) as f2:
        r2 = json.load(f2)
    s1 = []
    s2 = []
    for user_id, b in r1:
        if 'score' in b:
            s1.append(b['score'])
    for user_id, b in r2:
        if 'score' in b:
            s2.append(b['score'])
    logger.info('Number of fake accounts: %s', len(s1))
    logger.info('Number of snopes accounts: %s', len(s2))
    s1 = np.array(s1)
    s2 = np.array(s2)
    logger.info('Fake, mean=%s, std=%s', s1.mean(), s1.std())
    logger.info('Snopes, mean=%s, std=%s', s2.mean(), s2.mean())
    logger.info('Fake, ratio larger than 0.5: %s', (s1 > 0.5).sum() / len(s1))
    logger.info('Snopes, ratio larger than 0.5: %s',
                (s2 > 0.5).sum() / len(s2))
    fig, ax = plt.subplots(figsize=(4, 3))
    bins = np.linspace(0, 1, nbins + 1)
    w1 = np.ones_like(s1) / len(s1)
    w2 = np.ones_like(s2) / len(s2)
    ax.hist(s1, bins, weights=w1, alpha=0.75, label='Fake news', color=C1)
    ax.hist(s2, bins, weights=w2, alpha=0.75, label='Snopes', color=C2)
    plt.legend()
    plt.xlabel('Bot score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output)


def diffusion_ccdf(fn1='t_snopes_mv_fake_tweet.csv',
                   fn2='t_snopes_mv_snopes_tweet.csv',
                   output='diffusion-ccdf.pdf'):
    fn1 = join(DATA_DIR, fn1)
    fn2 = join(DATA_DIR, fn2)
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    s11 = df1.groupby('fake').tweet_id.nunique()
    s12 = df1.groupby('fake').tweet_user_id.nunique()
    s21 = df2.groupby('snopes').tweet_id.nunique()
    s22 = df2.groupby('snopes').tweet_user_id.nunique()
    s11 = ccdf(s11)
    s12 = ccdf(s12)
    s21 = ccdf(s21)
    s22 = ccdf(s22)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.4))
    ax1.plot(s11.index, s11.values, label='Claim')
    ax1.plot(s21.index, s21.values, label='Snopes')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('$n_1$')
    ax1.set_ylabel(r'$Pr\left\{N\geq n_1\right\}$')
    ax1.text(
        x=0.5,
        y=-0.48,
        s='(a) Article Popularity by Tweets',
        horizontalalignment='center',
        transform=ax1.transAxes)
    ax1.legend(fontsize=9)
    ax2.plot(s12.index, s12.values, label='Claim')
    ax2.plot(s22.index, s22.values, label='Snopes')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('$n_2$')
    ax2.set_ylabel(r'$Pr\left\{N\geq n_2\right\}$')
    ax2.text(
        x=0.5,
        y=-0.48,
        s='(b) Article Popularity by Users',
        horizontalalignment='center',
        transform=ax2.transAxes)
    ax2.legend(fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1], w_pad=2.4)
    plt.savefig(output)


def case_study1(fn='case_studies_I_article_id_108396.json',
                ofn='case-studies-I.pdf'):
    fn = join(DATA_DIR, fn)
    with open(fn, 'r') as f:
        data = json.load(f, encoding='utf-8')
    df = pd.DataFrame(data)
    df = df.drop_duplicates('id')
    df['user_screen_name'] = df.user.apply(lambda x: x['screen_name'])
    rs = dict()
    rs['ntotal'] = len(df)
    rs['hub_user_screen_name'] = df.groupby('user_screen_name').size().idxmax()
    rs['hub_user_tweets_n'] = df.groupby('user_screen_name').size().max()
    rs['hub_user_tweets_r'] = float(rs['hub_user_tweets_n']) / rs['ntotal']
    df2 = df.loc[(df.user_screen_name == rs['hub_user_screen_name'])
                 & (df.in_reply_to_status_id.isnull())
                 & (df.retweeted_status.isnull())
                 & (df.quoted_status.isnull())]
    rs['hub_user_origins_n'] = len(df2)
    rs['hub_user_origins_r'] = float(
        rs['hub_user_origins_n']) / rs['hub_user_tweets_n']
    print(rs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.15))
    patches, texts, autotexts = ax1.pie(
        [rs['hub_user_tweets_n'], rs['ntotal'] - rs['hub_user_tweets_n']],
        labels=['@' + rs['hub_user_screen_name'], 'Others'],
        explode=(0, 0.1),
        colors=('lightcoral', 'lightskyblue'),
        autopct='%1.1f%%',
        shadow=True,
        radius=1,
        startangle=180 - 7.2,
        labeldistance=1.15,
        textprops=dict(ha='center', family='monospace', fontsize=9),
    )
    df2.loc[:, 'created_at'] = pd.to_datetime(df2.created_at)
    df2.set_index('created_at', inplace=True)
    df2.resample('1D').size().plot(ax=ax2, logy=True, legend=False)
    ax1.text(x=0, y=-1.85, s='(a) Shares of tweets', ha='center', fontsize=9.5)
    ax2.set_xlabel(
        '(b) Timeline of @{}'.format(rs['hub_user_screen_name']), fontsize=9.5)
    ax2.set_ylabel('Volume')
    plt.tight_layout(pad=0.5)
    plt.savefig(ofn)


def case_study3(fn='case_studies_III.json', ofn='case-studies-III.pdf'):
    fn = join(DATA_DIR, fn)
    with open(fn, 'r') as f:
        data = json.load(f, encoding='utf-8')
    df = pd.DataFrame(data)
    df = df.drop_duplicates('id')
    df['user_screen_name'] = df.user.apply(lambda x: x['screen_name'])
    rs = dict()
    rs['ntotal'] = len(df)
    rs['hub_user_screen_name'] = df.groupby('user_screen_name').size().idxmax()
    rs['hub_user_tweets_n'] = df.groupby('user_screen_name').size().max()
    rs['hub_user_tweets_r'] = float(rs['hub_user_tweets_n']) / rs['ntotal']
    df2 = df.loc[(df.user_screen_name == rs['hub_user_screen_name'])
                 & (df.in_reply_to_status_id.notnull())]
    rs['hub_user_replies_n'] = len(df2)
    rs['hub_user_replies_r'] = float(
        rs['hub_user_replies_n']) / rs['hub_user_tweets_n']

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6, 2.6), gridspec_kw=dict(width_ratios=[2, 3]))
    patches, texts, autotexts = ax1.pie(
        [rs['hub_user_tweets_n'], rs['ntotal'] - rs['hub_user_tweets_n']],
        labels=['@' + rs['hub_user_screen_name'], 'Others'],
        explode=(0, 0.1),
        colors=('lightcoral', 'lightskyblue'),
        autopct='%1.1f%%',
        shadow=True,
        radius=1,
        startangle=180 - 7.2,
        labeldistance=1.15,
        textprops=dict(ha='center', family='monospace', fontsize=9),
    )
    wc = WordCloud(
        width=800, height=400, background_color='white',
        colormap='inferno').generate_from_frequencies(
            df2['in_reply_to_screen_name'].value_counts().to_dict())
    ax2.imshow(wc, interpolation="bilinear")
    ax2.set_axis_off()
    ax1.text(x=0, y=-1.6, s='(a) Shares of tweets', ha='center', fontsize=9.5)
    ax2.text(
        x=0.5,
        y=-0.34,
        s='(b) Word cloud of screen names of the replied',
        ha='center',
        fontsize=9.5,
        transform=ax2.transAxes)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(ofn)


def se_pd_vs_ftw(fn='t_snopes_mv_final.csv'):
    fn = join(DATA_DIR, fn)
    df = pd.read_csv(fn, parse_dates=['f_pd', 's_pd', 's_ftw', 'f_ftw'])
    df['dt_snopes'] = (df.s_ftw - df.s_pd) / np.timedelta64(1, 'h')
    df['dt_fake'] = (df.f_ftw - df.f_pd) / np.timedelta64(1, 'h')
    df = df.sort_values('dt_snopes', ascending=True)
    df = df.iloc[2:]
    print('snopes (exclude two exceptions): mean={} hour, sem={}'.format(
          df.dt_snopes.mean(), sem(df.dt_snopes.values)))
    print('fake (exclude two exceptions): mean={} hour, sem={}'.format(
          df.dt_fake.mean(), sem(df.dt_fake.values)))


def se_survival(fn='t_snopes_match_volume0104.csv'):
    fn = join(DATA_DIR, fn)

    df = pd.read_csv(fn, parse_dates=['f_pd', 's_pd'])
    T1 = (df.s_pd - df.f_pd) / np.timedelta64(1, 'h')
    T2 = df.f_n
    print('survival time: mean={}, sem={}'.format(T1.mean(), sem(T1.values)))
    print('survival tweets: mean={}, sem={}'.format(T2.mean(), sem(T2.values)))
