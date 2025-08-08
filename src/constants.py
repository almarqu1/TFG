# --- CONSTANTES DE ANOTACIÓN Y MODELO ---

# Mapeo de categorías a scores numéricos.
CATEGORY_TO_SCORE = {
    '🟢 STRONG YES': 92.5,
    '🟡 YES': 77.0,
    '🟠 MAYBE': 59.5,
    '🔴 WEAK NO': 39.5,
    '⚫ STRONG NO': 14.5
}

# Define el orden explícito de las categorías para la UI y el cálculo de Kappa.
ORDERED_CATEGORIES = [
    '⚫ STRONG NO',
    '🔴 WEAK NO',
    '🟠 MAYBE',
    '🟡 YES',
    '🟢 STRONG YES'
]

# --- CONSTANTES PARA EL MUESTREO DIRIGIDO ---

# Mapeo de categorías a palabras clave para el muestreo dirigido.
INDUSTRY_KEYWORDS_MAP = {
    # Creative & Communication
    'Art/Creative': ['art', 'creative', 'artist', 'illustrator', 'graphic', 'photographer', 'designer'],
    'Design': ['design', 'designer', 'ui/ux', 'ux/ui', 'user experience', 'user interface', 'visual', 'product design'],
    'Advertising': ['advertising', 'ad', 'campaign', 'media buyer', 'ppc', 'sem', 'paid search'],
    'Marketing': ['marketing', 'seo', 'content', 'social media', 'digital marketing', 'brand', 'growth'],
    'Public Relations': ['public relations', 'pr', 'communications', 'press release', 'media relations'],
    'Writing/Editing': ['writer', 'editor', 'copywriter', 'content creator', 'technical writer', 'author'],

    # Business & Management
    'Product Management': ['product manager', 'product owner', 'product strategy', 'roadmap'],
    'Project Management': ['project manager', 'pmo', 'agile', 'scrum', 'prince2', 'program manager'],
    'General Business': ['business', 'operations', 'ops', 'coordinator', 'associate', 'analyst'],
    'Business Development': ['business development', 'bizdev', 'partnerships', 'strategic alliances'],
    'Management': ['manager', 'management', 'lead', 'director', 'supervisor', 'vp', 'executive'],
    'Strategy/Planning': ['strategy', 'strategic', 'planner', 'corporate development', 'consultant'],
    'Consulting': ['consultant', 'consulting', 'advisory', 'mckinsey', 'bcg', 'bain', 'deloitte', 'pwc', 'ey', 'kpmg'],

    # STEM & Technical
    'Information Technology': ['it', 'tech', 'software', 'developer', 'engineer', 'programmer', 'code', 'database', 'dba', 'network', 'sysadmin', 'cloud', 'aws', 'azure', 'gcp', 'cybersecurity', 'devops'],
    'Engineering': ['engineer', 'engineering', 'mechanical', 'electrical', 'civil', 'chemical', 'hardware'],
    'Analyst': ['analyst', 'data analyst', 'business analyst', 'intelligence', 'reporting', 'analytics'],
    'Science': ['science', 'scientist', 'researcher', 'lab', 'laboratory', 'phd', 'postdoc'],
    'Research': ['research', 'r&d', 'researcher', 'study', 'clinical'],
    'Quality Assurance': ['qa', 'quality assurance', 'tester', 'sdet', 'automation testing', 'manual testing'],

    # Corporate & Operations
    'Finance': ['finance', 'financial', 'cfa', 'accountant', 'investment', 'banking', 'fintech', 'portfolio'],
    'Accounting/Auditing': ['accounting', 'accountant', 'cpa', 'audit', 'auditor', 'bookkeeping', 'tax'],
    'Human Resources': ['human resources', 'hr', 'recruiter', 'talent acquisition', 'hrbp', 'payroll'],
    'Administrative': ['administrative', 'admin', 'assistant', 'executive assistant', 'receptionist', 'office manager'],
    'Legal': ['legal', 'lawyer', 'paralegal', 'counsel', 'attorney', 'compliance'],
    'Customer Service': ['customer service', 'support', 'customer success', 'help desk', 'client services'],
    'Sales': ['sales', 'account executive', 'sdr', 'bdr', 'sales development', 'business development'],

    # Supply Chain & Manufacturing
    'Supply Chain': ['supply chain', 'logistics', 'procurement', 'sourcing', 'warehouse'],
    'Purchasing': ['purchasing', 'buyer', 'procurement', 'sourcing', 'vendor management'],
    'Distribution': ['distribution', 'logistics', 'warehouse', 'fulfillment', 'shipping'],
    'Manufacturing': ['manufacturing', 'plant', 'production', 'factory', 'operations', 'six sigma', 'lean'],
    'Production': ['production', 'manufacturing', 'plant', 'assembly', 'operator'],
    
    # Health & Education
    'Health Care Provider': ['health', 'healthcare', 'medical', 'doctor', 'nurse', 'rn', 'physician', 'hospital', 'clinic', 'pharma'],
    'Education': ['education', 'teacher', 'professor', 'academic', 'school', 'university', 'e-learning'],
    'Training': ['training', 'trainer', 'learning and development', 'l&d', 'corporate trainer', 'instructional design'],

    # Other (Catch-all)
    'Other': ['general', 'associate', 'intern', 'entry level', 'various'] 
}