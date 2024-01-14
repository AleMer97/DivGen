SELECT 
m_profile.*,
m_profile.user_id,
m_profile.name,
m_profile.vita,
GROUP_CONCAT(DISTINCT skills_data.name) as m_skills
FROM musician_profile m_profile
LEFT JOIN `musician_profile__skills` as m_skills ON m_profile.id = m_skills.entity_id
LEFT JOIN taxonomy_term_field_data skills_data ON skills_data.tid = m_skills.skills_target_id
GROUP BY m_profile.user_id
