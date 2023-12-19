SELECT PARSE_DATE("%Y%m%d", event_dt) AS fecha,
       userid,
       correlation_id,
       LOWER(TRIM(search_query)) AS search_query,
       item_id,
       item_list_index,
       price,
       item_name
       FROM `wmt-1257d458107910dad54c01f5c8.search.search_a2c`
       WHERE event_dt = ':init_date'
       AND userid NOT IN ('null','')
       AND item_id NOT IN ('null','')
       AND correlation_id NOT IN ('null','')
       AND search_query NOT IN ('null','')
       AND LENGTH(TRIM(search_query)) >= 3 
       AND item_list_index >= 0