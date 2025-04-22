import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.job_parser import JobParser
from src.data_processing.resume_parser import ResumeParser
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any

def visualize_parsed_data(parsed_data: Dict[str, Any], title: str):
    """Create visualizations for parsed data."""
    plt.figure(figsize=(15, 10))
    
    if 'education' in parsed_data:
        # Education visualization
        plt.subplot(2, 2, 1)
        education_df = pd.DataFrame(parsed_data['education'])
        if not education_df.empty:
            education_df['degree'].value_counts().plot(kind='bar')
            plt.title('Education Distribution')
            plt.xticks(rotation=45)
    
    if 'skills' in parsed_data:
        # Skills visualization
        plt.subplot(2, 2, 2)
        skills_df = pd.DataFrame({'skills': parsed_data['skills']})
        if not skills_df.empty:
            skills_df['skills'].value_counts().head(10).plot(kind='barh')
            plt.title('Top 10 Skills')
    
    if 'experience' in parsed_data:
        # Experience visualization
        plt.subplot(2, 2, 3)
        experience_df = pd.DataFrame(parsed_data['experience'])
        if not experience_df.empty and 'institution' in experience_df.columns:
            experience_df['institution'].value_counts().head(5).plot(kind='bar')
            plt.title('Top 5 Institutions')
            plt.xticks(rotation=45)
    
    if 'publications' in parsed_data:
        # Publications visualization
        plt.subplot(2, 2, 4)
        pubs_df = pd.DataFrame({'publications': parsed_data['publications']})
        if not pubs_df.empty:
            pubs_df['publications'].value_counts().head(5).plot(kind='barh')
            plt.title('Top 5 Publication Venues')
    
    plt.tight_layout()
    plt.savefig(f'tests/results/{title.lower().replace(" ", "_")}_visualization.png')
    plt.close()

def test_job_parser():
    print("\nTesting JobParser...")
    job_parser = JobParser()
    
    # Test with sample job description
    with open('tests/data/raw/jobs/sample_job.txt', 'r', encoding='utf-8') as f:
        job_text = f.read()
    
    parsed_job = job_parser.parse_job(job_text)
    
    print("\nParsed Job Information:")
    print(f"Title: {parsed_job['title']}")
    print(f"Department: {parsed_job['department']}")
    print(f"Institution: {parsed_job['institution']}")
    print("\nQualifications:")
    for qual in parsed_job['qualifications']:
        print(f"- {qual}")
    print("\nResponsibilities:")
    for resp in parsed_job['responsibilities']:
        print(f"- {resp}")
    print("\nRequirements:")
    for req in parsed_job['requirements']:
        print(f"- {req}")
    
    # Create visualization
    os.makedirs('tests/results', exist_ok=True)
    visualize_parsed_data(parsed_job, "Job Parser Results")

def test_resume_parser():
    print("\nTesting ResumeParser...")
    resume_parser = ResumeParser()
    
    # Test with sample resume
    with open('tests/data/raw/resumes/sample_resume.txt', 'r', encoding='utf-8') as f:
        resume_text = f.read()
    
    parsed_resume = resume_parser.parse_resume(resume_text)
    
    print("\nParsed Resume Information:")
    print(f"Name: {parsed_resume['name']}")
    print(f"Email: {parsed_resume['email']}")
    print("\nEducation:")
    for edu in parsed_resume['education']:
        print(f"- {edu['degree']} from {edu['institution']} ({edu['year']})")
    print("\nSkills:")
    for skill in parsed_resume['skills']:
        print(f"- {skill}")
    print("\nExperience:")
    for exp in parsed_resume['experience']:
        print(f"- {exp['description']} at {exp['institution']} ({exp['year']})")
    print("\nPublications:")
    for pub in parsed_resume['publications']:
        print(f"- {pub}")
    
    # Create visualization
    visualize_parsed_data(parsed_resume, "Resume Parser Results")

def main():
    print("Starting parser tests...")
    test_job_parser()
    test_resume_parser()
    print("\nTests completed! Visualizations have been saved to tests/results/")

if __name__ == "__main__":
    main() 