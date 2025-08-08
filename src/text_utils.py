import pandas as pd
import ast

def parse_string_list(s):
    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
    return []

def crear_input_oferta(row):
    exp_level = 'No especificado' if pd.isna(row.get('formatted_experience_level')) else row.get('formatted_experience_level')
    description = str(row.get('description', ''))
    if 'Job Type:' in description:
        description = description.split('Job Type:')[0].strip()
    parts = [
        f"[TITLE] {row.get('title', 'N/A')}",
        f"[EXPERIENCE] {exp_level}",
        f"[SKILLS] {', '.join([str(s) for s in row.get('skills_list', [])])}" if row.get('skills_list') else "",
        f"[INDUSTRIES] {', '.join([str(i) for i in row.get('industries_list', [])])}" if row.get('industries_list') else "",
        f"[DESCRIPTION] {description}"
    ]
    return " ".join(part for part in parts if part)

def crear_input_cv(row):
    tagged_lists = [
        ("POSITION", parse_string_list(row.get('positions'))),
        ("SKILLS", parse_string_list(row.get('skills'))),
        ("EDUCATION", parse_string_list(row.get('degree_names'))),
        ("UNIVERSITY", parse_string_list(row.get('educational_institution_name'))),
        ("PREVIOUS_COMPANIES", parse_string_list(row.get('professional_company_names'))),
        ("CERTIFICATIONS", parse_string_list(row.get('certification_skills'))),
        ("LANGUAGES", parse_string_list(row.get('languages'))),
        ("EXTRACURRICULAR", parse_string_list(row.get('extra_curricular_activity_types')))
    ]
    parts = [f"[{tag}] {', '.join([str(item) for item in items])}" for tag, items in tagged_lists if items]
    if row.get('responsibilities'):
        parts.append(f"[EXPERIENCE_SUMMARY] {row.get('responsibilities')}")
    return " ".join(parts)