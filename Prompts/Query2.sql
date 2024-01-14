-- m_search.guests?
-- booking status?


SELECT
    f_profile.name as f_name,
    f_profile.key as f_key,
    g.id as g_id,
    g.name as g_name,
    g.booking_status as g_booking_status,
    g.config as g_config,
    m_search.event_start as s_event_start,
    m_search.event_end as s_event_end,
    FROM_UNIXTIME(m_search.event_start) as s_event_start_date,
    FROM_UNIXTIME(m_search.event_end) as s_event_end_date,
    eventtypes_data.name as g_eventtypes,
    m_search.name as s_name,
    m_search.address__locality as s_adr_locality,
    m_search.address__country_code as s_adr_country_code,
    m_search.address__organization as s_adr_organization,
    m_search.geodata__lat as f_locality_lat,
    m_search.geodata__lon as f_locality_lon,
    m_search.description as s_description,
    m_search.budget as s_budget,
    genres_data.name as s_genres,
    requirements_data.name as s_requirements
FROM `formation_profile` f_profile
	RIGHT JOIN `gig` AS g ON f_profile.id = g.formation
    LEFT JOIN 0_connactz_musician_search AS m_search ON m_search.id = g.search
    LEFT JOIN taxonomy_term_field_data eventtypes_data ON eventtypes_data.tid = m_search.eventtype
    -- genres
    LEFT JOIN search_profile__genres s_genres ON s_genres.entity_id = m_search.id
    LEFT JOIN taxonomy_term_field_data genres_data ON genres_data.tid = s_genres.genres_target_id
    -- requirements (probably wrong id)
    LEFT JOIN search_profile__requirements s_requirements ON s_requirements.entity_id = m_search.id
    LEFT JOIN taxonomy_term_field_data requirements_data ON requirements_data.tid = s_requirements.requirements_target_id
WHERE g.booking_status NOT IN (0,2,7,11,12);