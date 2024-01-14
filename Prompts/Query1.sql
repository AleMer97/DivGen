SELECT
    f_profile.name as f_name,
    f_profile.key as f_key,
    f_profile.description as f_description,
    f_profile.distance as f_distance,
    f_profile.address__locality as f_locality,
    f_profile.geodata__lat as f_locality_lat,
    f_profile.geodata__lon as f_locality_lon,
    GROUP_CONCAT(DISTINCT casts_data.name) as f_casts,
    GROUP_CONCAT(DISTINCT genres_data.name) as f_genres,
    GROUP_CONCAT(DISTINCT eventtypes_data.name) as f_eventtypes,
    GROUP_CONCAT(DISTINCT m_profile.name) as f_musicians
FROM `formation_profile` f_profile
         LEFT JOIN formation_profile__casts f_casts ON f_casts.entity_id = f_profile.id
         LEFT JOIN taxonomy_term_field_data casts_data ON casts_data.tid = f_casts.casts_target_id
         LEFT JOIN formation_profile__genres f_genres ON f_genres.entity_id = f_profile.id
         LEFT JOIN taxonomy_term_field_data genres_data ON genres_data.tid = f_genres.genres_target_id
         LEFT JOIN formation_profile__eventtypes f_eventtypes ON f_eventtypes.entity_id = f_profile.id
         LEFT JOIN taxonomy_term_field_data eventtypes_data ON eventtypes_data.tid = f_eventtypes.eventtypes_target_id
         LEFT JOIN `membership` membership ON membership.formation = f_profile.id
         LEFT JOIN `musician_profile` m_profile ON m_profile.id = membership.musician
GROUP BY f_profile.id