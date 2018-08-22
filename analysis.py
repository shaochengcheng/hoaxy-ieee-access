import dateutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


def prepare_precision_percentile(fn, drop_duplicates=False):
    df = pd.read_csv(fn)
    if drop_duplicates is True:
        df = df.loc[df['duplicate'].str.lower() == 'no']
    useful_cols = ['score', 'relevant']
    df = df[useful_cols]
    n = len(df)
    scores = df['score'].unique()
    rs = [
        (s, 100 * float(sum(df['score'] <= s)) / n,
         float(sum(df.loc[df['score'] >= s]['relevant'].str.lower() == 'yes')) /
         sum(df['score'] >= s)) for s in scores
    ]
    rs_df = pd.DataFrame(rs, columns=['score', 'percentile', 'precision'])
    rs_df = rs_df.set_index('percentile')
    rs_df = rs_df.sort_index(ascending=True)
    return rs_df


def prepare_precision_rank(fn, drop_duplicates=False):
    df = pd.read_csv(fn)
    if drop_duplicates is True:
        df = df.loc[df['duplicate'].str.lower() == 'no']
    useful_cols = ['score', 'relevant']
    df = df[useful_cols]
    rs = [(df.iloc[i - 1]['score'], i,
           float(sum(df.iloc[:i]['relevant'].str.lower() == 'yes')) / i)
          for i in range(1, len(df) + 1)]
    rs_df = pd.DataFrame(rs, columns=['score', 'rank', 'precision'])
    rs_df = rs_df.set_index('rank')
    rs_df = rs_df.sort_index(ascending=True)
    return rs_df


def q_read_csv_url(csvfile):
    """
    Quick alias to read csvfile of view v_ass_date.
    ---------
    Parameters:
        csvfile, str, filename
    Return:
        df, pd.DataFrame
    """
    df = pd.read_csv(
        csvfile, header=0, parse_dates=['s_pd', 'f_pd', 's_ft', 'f_ft'])
    return df.loc[(df['s_pd'].notnull()) & (df['f_pd'].notnull())]


def group_fake(df, gf='min'):
    """
    Group by snopes, to get min or max f_pd.
    ----------
    Parameters:
        df, pd.DataFrame, raw dataframe
        gf, str, groupby function name, {'min', 'max'}
    Return:
        df, pd.DataFrame
    """
    if gf == 'min':
        return df.groupby(['snopes'], as_index=False).min()
    elif gf == 'max':
        return df.groupby(['snopes'], as_index=False).max()


def subset_after(df, start=dateutil.parser.parse('2016-05-16')):
    """
    Return subset of data, after start of our collection.
    -----------------
    Parameters:
        df, pd.DataFrame, raw dataframe
        start, datetime, begining of collect
    Return:
        pd.DataFrame
    """
    df = df.loc[df['f_pd'] >= start]
    exclude_snopes = 'http://www.snopes.com/106-dead-in' +\
        '-california-music-festival-bombing/'
    return df.loc[df['snopes'] != exclude_snopes]


def ddist_p_v_p(df, unit='h'):
    """
    Delay distribution of delta, where delta is difference between df['s_pd']
    and df['f_pd'].
    -----------
    Parameters:
        df, pd.DataFrame, dataset
        unit, string, unit of datetime
    Return:
        pd.Series
    """
    ds = df['s_pd'] - df['f_pd']
    ds.index = df['s_pd']
    ds = ds / np.timedelta64(1, unit)
    return ds


def ddist_p_v_t(df, unit='h'):
    """
    Delay distribution of delta, where delta is difference between df['x_ft']
    and df['x_pd'].
    -------------------
    Parameters:
        df, pd.DataFrame, dataset (should be subset_after_collect)
        unit, string, unit of datetime
    Return:
        pd.DataFrame
    """
    ds = pd.DataFrame(
        dict(snopes=(df['s_ft'] - df['s_pd']), fake=(df['f_ft'] - df['f_pd'])))
    for col in ds.columns:
        ds[col] = ds[col] / np.timedelta64(1, 'm')
    ds.set_index(df['s_pd'], inplace=True)
    return ds


def build_file(mode, hour, dir='/home/shao', prefix='delta_vs_tweets_'):
    """
    Build file path.
    -----------------
    Parameters:
        mode, string, type of measument 1
        hour, string, type of measument 2
        dir, string, directory that store the data file
        prefix, stirng, prefix of the data file
    Return:
        string, filepath
    """
    if mode == 'whole':
        return os.join(dir, prefix, mode, '.csv')
    else:
        return os.join(dir, prefix, mode, '_' + hour, '.csv')


def dist_d_v_t(csvfile, group_fake=True, delta_max=True):
    """
    Distribution of delay (p_v_p) associated with number of tweets.
    ----------------
    Parameters:
        csvfile, string, filename
    Return:
        pd.DataFrame
    """

    # raw csv file structure
    # snopes,fake,s_pd,delta,n_tw
    def gf(x):
        if delta_max is True:
            r = dict(
                snopes=x['snopes'][0],
                n_fake=x['fake'].count(),
                s_pd=x['s_pd'][0],
                delta=max(x['s_pd']),
                tn_tw=sum(x['n_tw']))
        else:
            r = dict(
                snopes=x['snopes'][0],
                n_fake=x['fake'].count(),
                s_pd=x['s_pd'][0],
                delta=min(x['s_pd']),
                tn_tw=sum(x['n_tw']))
        return r

    df = pd.read_csv(csvfile, parse_dates=['s_pd'])
    df['delta'] = pd.to_timedelta(df['delta'])
    df.groupby(['snopes'], as_index=False).apply(gf)
    return df


def q_read_csv_ts(csvfile):
    """
    Quick alias to read csv file, time series analysis
    --------------------
    Parameters:
        csvfile, string, file name
    Return:
        pd.Dataframe
        where columns are:  tweet_created_at,tweet_id,tweet_user_id,
                            tweet_type,snopes_page_url,fake_page_url
    """
    return pd.read_csv(csvfile, parse_dates=['tweet_created_at'])


def get_tweets_ts(df):
    """
    Get tweets timeline from dataframe.
    ------------------
    Parameters:
        df, pd.DataFrame, raw data set
    Return:
        tuple of pd.Series (ts1, ts2)
        where, ts1 is tweets for fact checking and ts2 is tweets for fake news.
    """
    df.set_index('tweet_created_at', inplace=True)
    ts1 = df['tweet_id'].loc[df['tweet_type'] == 'fact_checking']
    ts2 = df['tweet_id'].loc[df['tweet_type'] == 'fake_news']
    ts1.drop_duplicates(inplace=True)
    ts2.drop_duplicates(inplace=True)
    return ts1, ts2


def get_overlapped_tweets(df):
    """
    Get overlapped tweets that contain both fake and fact_checking urls.
    -------------------
    Parameters:
        df, pd.DataFrame, raw data
    Return:
        overlap, pd.Dataframe
    """
    duplicates = df.loc[df['tweet_id'].duplicated(keep=False)]
    return duplicates.groupby(
        ['tweet_id'],
        as_index=False).apply(lambda x: x['tweet_type'].nunique() > 1)


def prepare_df_tweet_timeline(f,
                              parse_dates=['tweet_created_at'],
                              keep_url=False):
    """
    Prepare DataFrame data of tweets timeline. Data should come from
        t_ftw_fake or t_ftw_snopes
    --------------------------------------------------------------------
    Parameters:
        f, string, csv file path
        parse_dates, same as pd.read_csv
        keep_url, boolean, whether keep url columns, if False, duplication
            will be dropped
    Return:
        DataFrame, tweet timeline data
    """
    df = pd.read_csv(f, parse_dates=parse_dates)
    if keep_url is True:
        return df
    else:
        uq_keys = [
            'tweet_created_at', 'tweet_id', 'tweet_user_id',
            'retweeted_user_id', 'quoted_user_id', 'in_reply_to_user_id'
        ]
        return df[uq_keys].drop_duplicates()


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
                print(g[1]['fake'][0])
                df1 = df1.loc[g[0]]
                fakes = g[1]['fake']
                return df1, [df2.loc[fake] for fake in fakes]


def tt_top_users(df, p=0.01):
    """
    Get top activate users from timeline data
    ---------------------------------------------------------------------
    Parameters:
        df, DataFrame, dumped from t_ftw_snopes or t_ftw_fake
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
    n_origin = len(df.loc[(df['retweeted_user_id'].isnull()) & (
        df['quoted_user_id'].isnull()) & (df['in_reply_to_user_id'].isnull())])
    if quote_mode is None:
        n_origin = n_total - n_retweet - n_reply
        return pd.Series(
            [n_origin, n_retweet, n_reply, n_total],
            index=['origin', 'retweet', 'reply', 'total'])
    if quote_mode == 'any':
        n_quote = n_df['quoted_user_id']
    else:
        n_quote = len(df.loc[(df['quoted_user_id'].notnull()) &
                             (df['retweeted_user_id'].isnull()) &
                             (df['in_reply_to_user_id'].isnull())])
    return pd.Series(
        [n_origin, n_retweet, n_quote, n_reply, n_total],
        index=['origin', 'retweet', 'quote', 'reply', 'total'])


def prepare_plot_frac_over_retweet(f1, f2, p):
    dfa1 = prepare_df_tweet_timeline(f1, parse_dates=False)
    dfa2 = prepare_df_tweet_timeline(f2, parse_dates=False)
    dft1 = tt_top_users_df(dfa1, p)
    dft2 = tt_top_users_df(dfa2, p)
    sa1 = tt_count_tweet_type(dfa1)
    sa1 = sa1[:-1] / sa1['retweet']
    sa2 = tt_count_tweet_type(dfa2)
    sa2 = sa2[:-1] / sa2['retweet']
    st1 = tt_count_tweet_type(dft1)
    st1 = st1[:-1] / st1['retweet']
    st2 = tt_count_tweet_type(dft2)
    st2 = st2[:-1] / st2['retweet']

    return pd.Panel(
        dict(
            origin=pd.DataFrame(
                [[sa1['origin'], st1['origin']], [sa2['origin'], st2['origin']]
                ],
                columns=['all', 'top'],
                index=['fact checking', 'fake news']),
            quote=pd.DataFrame(
                [[sa1['quote'], st1['quote']], [sa2['quote'], st2['quote']]],
                columns=['all', 'top'],
                index=['fact checking', 'fake news']),
            reply=pd.DataFrame(
                [[sa1['reply'], st1['reply']], [sa2['reply'], st2['reply']]],
                columns=['all', 'top'],
                index=['fact checking', 'fake news'])))


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
                [(sa1['origin'], sa2['origin']), (sa1['retweet'], sa2[
                    'retweet']), (sa1['reply'], sa2['reply'])],
                columns=['snopes', 'fake news'],
                index=['origin', 'retweet', 'reply']),
            top=pd.DataFrame(
                [(st1['origin'], st2['origin']), (st1['retweet'], st2[
                    'retweet']), (st1['reply'], st2['reply'])],
                columns=['snopes', 'fake news'],
                index=['origin', 'retweet', 'reply'])))


def tsa_sma_hourly(ts, window=24, center=True, drop_na=True):
    sma = ts.rolling(window, center=center).mean()
    if drop_na is True:
        sma = sma[sma.notnull()]
    return sma


def tsa_ccf(ts1,
            ts2,
            rule='H',
            max_lag=15,
            do_sma=False,
            fill_zero_as_mean=False):
    # ts1 keep fixed, shift ts2
    # positive lag, shift ts2 to right,
    #   in time series analysis, ts2 went ahead of ts1
    # negative lag, shift ts2 to left,
    #   in time series analysis, ts2 left behind ts1
    ts1 = ts1.resample(rule).count()
    ts2 = ts2.resample(rule).count()
    if fill_zero_as_mean:
        ts1[ts1 == 0] = ts1.mean()
        ts2[ts1 == 0] = ts2.mean()
    # alignment, we want the overlapp part, using inner join
    ts1, ts2 = ts1.align(ts2, join='inner')
    if do_sma is True:
        ts1 = tsa_sma_hourly(ts1)
        ts2 = tsa_sma_hourly(ts2)
        y = ccf(ts1, ts2)
    else:
        y = ccf(ts1, ts2)
    y_half = len(y) // 2
    x = range(-y_half, y_half + 1)
    full = pd.Series(y, index=x)
    return full.loc[-max_lag:max_lag]


def tsa_ccf2(s1, s2, max_lag=15, do_sma=False):
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


def bot_or_not(top_users):
    """
    Bot_or_not check of top activate users
    --------------------------------------------------------------------
    Parameters:
        top_users: a list of top users account (could be a list of
            screen_name, or user_id, or user_id_str)
    --------------------------------------------------------------------
    Return:
        List, BotOrNot Checking result.
    """
    twitter_app_auth = {
        'access_token': '1434877380-SHayIupA2JDUiHhdF4H5O43GJ7HrU7RH0Vp4akW',
        'access_token_secret': 'w3SP8E5Q0zVH4OFhPiF6rqb1yS5pykVHw124AvMnzkAuC',
        'consumer_key': 'kHR8FKcFgg0U4knJTa19vjvhX',
        'consumer_secret': 'niCpYQCGP5HJr4IqYvYvTD8EBpdhQk7jYoQSJr38aLewNHTBsA'
    }
    bon = botornot.BotOrNot(**twitter_app_auth)
    return list(bon.check_accounts_in(top_users))


if __name__ == '__main__':
    f1 = '/home/shao/t_ftw_snopes-9-30.csv'
    f2 = '/home/shao/t_ftw_fake-9-30.csv'
    df1 = pd.read_csv(f1, parse_dates=['tweet_created_at'])
    df1 = twtl_drop_duplicates(df1)
    top_users = list(get_top_user(df1))
    top_bot_checks = top_user_bot_or_not(top_users)
