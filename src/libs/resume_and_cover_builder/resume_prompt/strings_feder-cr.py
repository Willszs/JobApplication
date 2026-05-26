from src.libs.resume_and_cover_builder.template_base import prompt_header_template, prompt_education_template, prompt_working_experience_template, prompt_projects_template, prompt_achievements_template, prompt_certifications_template, prompt_additional_skills_template

prompt_header = """
Act as an HR expert and resume writer specializing in ATS-friendly resumes. Your task is to create a professional and polished header for the resume. The header should:

1. **Contact Information**: Include your full name, city and country, phone number, email address, LinkedIn profile, and GitHub profile. Exclude any information that is not provided.
2. **Formatting**: Ensure the contact details are presented clearly and are easy to read.

- **My information:**  
  {personal_information}
""" + prompt_header_template


prompt_education = """
Act as an HR expert and resume writer with a specialization in creating ATS-friendly resumes. Your task is to articulate the educational background for a resume. For each educational entry, ensure you include:

1. **Institution Name and Location**: Specify the university or educational institution’s name and location.
2. **Degree and Field of Study**: Clearly indicate the degree earned and the field of study.
3. **Grade**: Include your Grade if it is strong and relevant.
4. **Relevant Coursework**: List key courses with their grades to showcase your academic strengths.

- **My information:**  
  {education_details}
"""+ prompt_education_template


prompt_working_experience = """
Act as an HR expert and resume writer with a specialization in creating ATS-friendly resumes. Your task is to detail the work experience for a resume. For each job entry, ensure you include:

1. **Company Name and Location**: Provide the name of the company and its location.
2. **Job Title**: Clearly state your job title.
3. **Dates of Employment**: Include the start and end dates of your employment.
4. **Responsibilities and Achievements**: Describe your key responsibilities and notable achievements, emphasizing measurable results and specific contributions.

- **My information:**  
  {experience_details}
"""+ prompt_working_experience_template


prompt_projects = """
Act as an HR expert and resume writer with a specialization in creating ATS-friendly resumes. Your task is to highlight notable side projects. For each project, ensure you include:

1. **Project Name and Link**: Provide the name of the project and include a link to the GitHub repository or project page.
2. **Project Details**: Describe any notable recognition or achievements related to the project, such as GitHub stars or community feedback.
3. **Technical Contributions**: Highlight your specific contributions and the technologies used in the project. 

- **My information:**  
  {projects}
"""+ prompt_projects_template


prompt_achievements = """
Act as an HR expert and resume writer with a specialization in creating ATS-friendly resumes. Your task is to list significant achievements. For each achievement, ensure you include:

1. **Award or Recognition**: Clearly state the name of the award, recognition, scholarship, or honor.
2. **Description**: Provide a brief description of the achievement and its relevance to your career or academic journey.

- **My information:**  
  {achievements}
"""+ prompt_achievements_template


prompt_additional_skills = """
Act as an HR expert and resume writer with a specialization in ATS-friendly resumes. Your task is to create a concise **Skills** section based primarily on the generated Work Experience.

Rules:
- Use Generated Work Experience as the primary factual source.
- Use Known Skills and Languages only as supporting context.
- Group related skills into concise categories.
- Do not invent tools, brands, certifications, or domains.
- If Languages are provided, include them as the final inline item in the Skills section.
- Render each skill category as an inline item, one after another, not as a vertical bullet list.
- Return HTML only.

Generated Work Experience:
{work_experience}

Known Skills:
{skills}

Languages:
{languages}
"""+ prompt_additional_skills_template
