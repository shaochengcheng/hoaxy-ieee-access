import simplejson as json
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


logger = logging.getLogger(__name__)


def case_study1(fn='case_studies_I_article_id_108396.json',
                ofn='case-studies-I.pdf'
                ):
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
            & (df.quoted_status.isnull())
            ]
    rs['hub_user_origins_n'] = len(df2)
    rs['hub_user_origins_r'] = float(rs['hub_user_origins_n']) / rs['hub_user_tweets_n']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.15))
    patches, texts, autotexts = ax1.pie(
            [ rs['hub_user_tweets_n'], rs['ntotal']-rs['hub_user_tweets_n']],
            labels=['@'+rs['hub_user_screen_name'], 'Others'],
            explode=(0, 0.1),
            colors=('lightcoral', 'lightskyblue'),
            autopct='%1.1f%%',
            shadow=True,
            radius=1,
            startangle=180-7.2,
            labeldistance=1.15,
            textprops=dict( ha='center', family='monospace', fontsize=9),
            )
    df2.loc[:, 'created_at'] = pd.to_datetime(df2.created_at)
    df2.set_index('created_at', inplace=True)
    df2.resample('1D').size().plot(ax=ax2, logy=True, legend=False)
    ax1.text(x=0, y=-1.85, s='(a) Shares of tweets', ha='center', fontsize=9.5)
    ax2.set_xlabel('(b) Timeline of @{}'.format(
        rs['hub_user_screen_name']), fontsize=9.5)
    ax2.set_ylabel('Volume')
    plt.tight_layout(pad=0.5)
    plt.savefig(ofn)




