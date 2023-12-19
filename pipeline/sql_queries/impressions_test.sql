SELECT SCL_CREATED_DATE,
       customer_id, 
       correlation_id,
       LOWER(TRIM(request_term)) AS search_query,
       response_item_number,
       response_sku,
       response_position,
       response_name
       FROM `wmt-1257d458107910dad54c01f5c8.dw_customer_search_recom_se.vds_customer_search_result` 
       WHERE SCL_CREATED_DATE = PARSE_DATE('%Y%m%d', ':init_date')
       AND customer_id NOT IN ('null','')
       AND correlation_id NOT IN ('null','')
       AND LENGTH(TRIM(request_term)) >= 3 