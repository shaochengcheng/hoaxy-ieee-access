from os.path import join

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = 'data0104'
# claim
C1 = '#1F78B4'
# fact_checking
C2 = '#FF7F00'


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


def tt_count_tweet_type(df, quote_mode='any'):
    """
    Return numbers of original, quoted, replied, and retweeted tweet
    --------------------------------------------------------------------
    Parameters:
        df, DataFrame, tweet timeline dataframe
        quote_mode, string, choice of {'any', 'only'}
            'any', count if quoted_user_id is not null
            'only', count if quoted_user_id is not null
                    and not retweet and not reply
    Return:
        pd.Series contains count numbers
    """
    n_df = df.count()
    n_total = n_df['tweet_id']
    n_retweet = n_df['retweeted_user_id']
    n_reply = n_df['in_reply_to_user_id']
    n_origin = len(df.loc[(df['retweeted_user_id'].isnull())
                          & (df['quoted_user_id'].isnull()) &
                          (df['in_reply_to_user_id'].isnull())])
    if quote_mode is None:
        n_origin = n_total - n_retweet - n_reply
        return pd.Series(
            [n_origin, n_retweet, n_reply, n_total],
            index=['origin', 'retweet', 'reply', 'total'])
    if quote_mode == 'any':
        n_quote = n_df['quoted_user_id']
    else:
        n_quote = len(df.loc[(df['quoted_user_id'].notnull())
                             & (df['retweeted_user_id'].isnull()) &
                             (df['in_reply_to_user_id'].isnull())])
    return pd.Series(
        [n_origin, n_retweet, n_quote, n_reply, n_total],
        index=['origin', 'retweet', 'quote', 'reply', 'total'])


def tt_top_users(df, p=0.01):
    """
    Get top activate users from timeline data
    ---------------------------------------------------------------------
    Parameters:
        df, DataFrame
        p, float, percentage of top users
    Return:
        Int64Index
    """
    df = df.groupby(['tweet_user_id']).count()
    df = df.sort_values(by='tweet_id', ascending=False)
    n = int(len(df) * p)
    return df.index[:n]


def tt_top_users_df(df, p=0.01):
    top_users = tt_top_users(df, p)
    df = df.set_index('tweet_user_id')
    df = df.loc[top_users]
    return df.reset_index()


def prepare_plot_frac_over_total(df, p):
    uq_keys = [
        'tweet_created_at', 'tweet_id', 'tweet_user_id', 'retweeted_user_id',
        'quoted_user_id', 'in_reply_to_user_id'
    ]
    df.rename(
        columns=dict(
            tweet_retweeted_status_id='retweeted_user_id',
            tweet_quoted_status_id='quoted_user_id',
            tweet_in_reply_to_status_id='in_reply_to_user_id',
            tweet_raw_id='tweet_id',
            user_raw_id='tweet_user_id'),
        inplace=True)
    dfa1 = df.loc[df.site_label == 0].drop_duplicates(uq_keys)
    dfa2 = df.loc[df.site_label == 1].drop_duplicates(uq_keys)
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

    return dict(
        all=pd.DataFrame(
            [(sa1['origin'], sa2['origin']), (sa1['retweet'], sa2['retweet']),
             (sa1['reply'], sa2['reply'])],
            columns=['fake news', 'snopes'],
            index=['origin', 'retweet', 'reply']),
        top=pd.DataFrame(
            [(st1['origin'], st2['origin']), (st1['retweet'], st2['retweet']),
             (st1['reply'], st2['reply'])],
            columns=['fake news', 'snopes'],
            index=['origin', 'retweet', 'reply']))


def tw_type_share(df, p=0.1, output='generized-types-of-tweets.pdf'):
    """
    Plot shares of different tweet types
    """
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

    data = prepare_plot_frac_over_total(df, p)
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
        axis='both', which='both', bottom=False, top=False, right=False)
    ax2.tick_params(
        axis='both', which='both', bottom=False, top=False, right=False)
    ax0.legend(
        (rects1[0], rects2[0]), ('Claim', 'Fact-checking'),
        bbox_to_anchor=(0., -3.2, 1., .1),
        borderaxespad=0.,
        ncol=2,
        mode='expand',
        loc=3,
        fontsize=10)
    ax0.tick_params(
        axis='both', which='both', bottom=False, top=False, right=False)
    ax0.axis('off')
    plt.tight_layout(pad=0.1, w_pad=1.2, h_pad=0.2)
    plt.savefig(output)


def diffusion_ccdf(df,
                   output='generized-diffusion-ccdf.pdf'):
    gps = df.groupby(['article_id', 'site_label'])
    by_ntweets = gps.tweet_raw_id.nunique().rename('ntweets')
    by_nusers = gps.user_raw_id.nunique().rename('nusers')
    s11 = by_ntweets.loc[:, 0].reset_index(drop=True)
    s12 = by_ntweets.loc[:, 1].reset_index(drop=True)
    s21 = by_nusers.loc[:, 0].reset_index(drop=True)
    s22 = by_nusers.loc[:, 1].reset_index(drop=True)
    s11 = ccdf(s11)
    s12 = ccdf(s12)
    s21 = ccdf(s21)
    s22 = ccdf(s22)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.4))
    ax1.plot(s11.index, s11.values, label='Claim')
    ax1.plot(s21.index, s21.values, label='Fact checking')
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
    ax2.plot(s22.index, s22.values, label='Fact checking')
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
