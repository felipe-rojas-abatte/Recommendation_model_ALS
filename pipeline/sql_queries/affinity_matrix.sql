WITH matrix AS (
  SELECT userid, 
         item_id AS itemid,
         case when COUNT(*) > 0 then 1 else 0 end AS interaccion,
         COUNT(*) AS n_items,
         sum(price) AS total_price,
         PARSE_DATE("%Y%m%d", event_dt) AS event_day,
         FROM `wmt-1257d458107910dad54c01f5c8.search.search_a2c`
         WHERE event_dt BETWEEN ':init_date' AND ':end_date'
         AND userid IS NOT NULL
         AND userid NOT IN ('null','')
         AND item_id IS NOT NULL
         AND item_id NOT IN ('(not set)','null','') 
         GROUP BY userid, item_id, event_day
),
total_n_items_price_per_user_per_day AS (  --tippud
  SELECT userid,
         event_day,
         sum(n_items) AS n_total_items_per_day,
         sum(total_price) AS n_total_price_per_day,
         FROM matrix
         GROUP BY userid, event_day
),
weighted_matrix AS (
  SELECT m.userid,
         m.itemid,
         m.interaccion,
         m.n_items,
         m.total_price,
         tippud.n_total_items_per_day,
         tippud.n_total_price_per_day,
         m.event_day,
         (m.n_items/tippud.n_total_items_per_day) AS weight_items,
         (m.total_price/tippud.n_total_price_per_day) AS weight_price 
         FROM matrix m
         LEFT JOIN total_n_items_price_per_user_per_day tippud
         ON m.userid = tippud.userid AND m.event_day = tippud.event_day
)

SELECT userid,
       itemid,
       1.0 + SUM(weight_items) + SUM(weight_price) as affinity_score,
       FROM weighted_matrix
       WHERE interaccion = 1
       GROUP BY userid, itemid
       ORDER BY userid ASC, itemid ASC