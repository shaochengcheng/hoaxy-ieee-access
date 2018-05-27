import tweepy


def to_bulk(a, size=100):
    """Transform a list into list of list. Each element of the new list is a
    list with size=100 (except the last one).
    """
    r = []
    qt, rm = divmod(len(a), size)
    i = -1
    for i in range(qt):
        r.append(a[i * size:(i + 1) * size])
    if rm != 0:
        r.append(a[(i + 1) * size:])
    return r


def fetch_status(api, ids):
    """Fetch status.

    Parameters
    ----------
    api : object
        TweepAPI instance.
    uids : list
        A list of tweet ids.

    Returns
    -------
    A list of tweets (Tweepy tweet model.)
    """
    rs = []
    for id_block in to_bulk(ids):
        try:
            rs += api.statuses_lookup(id=ids)
        except tweepy.TweepError as e:
            logger.warning(e)
    return rs


def do_auth()
    ak = dict(
            consumer_key='UxlMUvN1HR9zrFxn2wugD0uKn',
            consumer_secret='NSvKMwDFgB4uGgNJ7L0ET0qGPI5FAgTY9G0DclKrz7if3j8Gax',
            access_token='849085485859438592-xP6r8emYEBvbGajLNLauCFfNCLdAtGC',
            access_token_secret='RyiT4RHswMsu6fwPX9hLgQjh7E6PyvQXt21myBdD96RlL'
            )
    auth = tweepy.OAuthHandler(ak['consumer_key'], ak['consumer_secret'])
    auth.set_access_token(ak['access_token'], ak['access_token_secret'])
    return tweepy.API(auth, wait_on_rate_limit=True))


def fetch_replied_tweets(in_fn, out_fn):
    df = pd.read_csv(in_fn)
    api = do_auth()
    rs = fetch_status(api, ids=df.)
