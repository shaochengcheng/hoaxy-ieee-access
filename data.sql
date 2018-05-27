--- CASE STUDIES I, posting origin tweets
SELECT
    tw.raw_id AS tweet_raw_id,
    tw.created_at AS tweet_created_at,
    atw.retweeted_status_id,
    atw.quoted_status_id,
    atw.in_reply_to_status_id,
    tw.json_data#>>'{user, id}' AS user_raw_id,
    tw.json_data#>>'{user, screen_name}' AS user_screen_name
FROM tweet AS tw
    JOIN ass_tweet AS atw ON atw.id=tw.id
    JOIN ass_tweet_url AS atu ON atu.tweet_id=tw.id
    JOIN url AS u ON u.id=atu.url_id
    JOIN article AS a ON a.id=u.article_id
WHERE tw.created_at < '20170401'
    AND a.id=108396
;

\COPY ( SELECT tw.raw_id AS tweet_raw_id, tw.created_at AS tweet_created_at, atw.retweeted_status_id, atw.quoted_status_id, atw.in_reply_to_status_id, tw.json_data#>>'{user, id}' AS user_raw_id, tw.json_data#>>'{user, screen_name}' AS user_screen_name FROM tweet AS tw JOIN ass_tweet AS atw ON atw.id=tw.id JOIN ass_tweet_url AS atu ON atu.tweet_id=tw.id JOIN url AS u ON u.id=atu.url_id JOIN article AS a ON a.id=u.article_id WHERE tw.created_at < '20170401' AND a.id=108396) TO '/u/shaoc/case_studies_I_article_id_108396.csv' WITH HEADER CSV

--- CASE STUDIES I, JSON OUTPUT
SELECT JSON_AGG(tw.json_data)::text
FROM tweet AS tw
    JOIN ass_tweet AS atw ON atw.id=tw.id
    JOIN ass_tweet_url AS atu ON atu.tweet_id=tw.id
    JOIN url AS u ON u.id=atu.url_id
    JOIN article AS a ON a.id=u.article_id
WHERE tw.created_at < '20170401'
    AND a.id=108396
;

psql -U shaoc -d hoaxy -h recall.ils.indiana.edu -p 5433 -c "COPY ( SELECT JSON_AGG(tw.json_data)::text FROM tweet AS tw JOIN ass_tweet AS atw ON atw.id=tw.id JOIN ass_tweet_url AS atu ON atu.tweet_id=tw.id JOIN url AS u ON u.id=atu.url_id JOIN article AS a ON a.id=u.article_id WHERE tw.created_at < '20170401' AND a.id=108396) TO STDOUT" > /u/shaoc/case_studies_I_article_id_108396.json



--- CASE STUDIES III, replying
SELECT DISTINCT
    a.id AS article_id,
    tw.raw_id AS tweet_raw_id,
    tw.json_data->>'in_reply_to_screen_name' AS in_reply_to_screen_name,
    tw.json_data#>>'{user, screen_name}' AS from_screen_name,
    tw.json_data#>>'{user, id}' AS from_user_raw_id
FROM tweet AS tw
JOIN ass_tweet AS atw ON atw.id=tw.id
JOIN ass_tweet_url AS atu ON atu.tweet_id=tw.id
JOIN url AS u ON u.id=atu.url_id
JOIN article AS a ON u.article_id=a.id
JOIN site AS s ON s.id=u.site_id
WHERE s.site_type LIKE 'claim' AND s.is_enabled IS TRUE
    AND a.canonical_url SIMILAR TO 'https?://[^/]+/_%'
    AND a.date_captured<'2017-04-01'
    AND tw.created_at<'2017-04-01'
    AND atw.in_reply_to_status_id IS NOT NULL AND atw.quoted_status_id IS NULL
;


\COPY ( SELECT DISTINCT a.id AS article_id, tw.raw_id AS tweet_raw_id, tw.json_data->>'in_reply_to_screen_name' AS in_reply_to_screen_name, tw.json_data#>>'{user, screen_name}' AS from_screen_name, tw.json_data#>>'{user, id}' AS from_user_raw_id FROM tweet AS tw JOIN ass_tweet AS atw ON atw.id=tw.id JOIN ass_tweet_url AS atu ON atu.tweet_id=tw.id JOIN url AS u ON u.id=atu.url_id JOIN article AS a ON u.article_id=a.id JOIN site AS s ON s.id=u.site_id WHERE s.site_type LIKE 'claim' AND s.is_enabled IS TRUE AND a.canonical_url SIMILAR TO 'https?://[^/]+/_%' AND a.date_captured<'2017-04-01' AND tw.created_at<'2017-04-01' AND atw.in_reply_to_status_id IS NOT NULL AND atw.quoted_status_id IS NULL) TO '/u/shaoc/claim_by_replying.csv' WITH HEADER CSV





--- CASE STUDIES III, JSON OUTPUT
SELECT JSON_AGG(tw.json_data)::text
FROM tweet AS tw
    JOIN ass_tweet AS atw ON atw.id=tw.id
    JOIN ass_tweet_url AS atu ON atu.tweet_id=tw.id
    JOIN url AS u ON u.id=atu.url_id
    JOIN article AS a ON a.id=u.article_id
WHERE tw.created_at < '20170401'
    AND a.id=849178
;

\COPY ( SELECT JSON_AGG(tw.json_data)::text FROM tweet AS tw JOIN ass_tweet AS atw ON atw.id=tw.id JOIN ass_tweet_url AS atu ON atu.tweet_id=tw.id JOIN url AS u ON u.id=atu.url_id JOIN article AS a ON a.id=u.article_id WHERE tw.created_at < '20170401' AND a.id=849178 ) TO '/u/shaoc/case-studies-III.json'


