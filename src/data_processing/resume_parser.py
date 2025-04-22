import pandas as pd
import re
import os
from tqdm import tqdm
import json
from typing import Dict, List, Optional, Any

class ResumeParser:
    def __init__(self):
        """Initialize the ResumeParser with regex patterns and keywords."""
        # Regex patterns
        self.email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        self.name_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})'
        self.degree_pattern = r'(?:Ph\.?D\.?|M\.?Sc\.?|B\.?Sc\.?|Master|Bachelor|Doctor)'
        self.year_pattern = r'(?:19|20)\d{2}'
        self.institution_pattern = r'(?:University|College|Institute|School)\s+(?:of\s+)?[A-Za-z\s,]+'
        
        # Keywords for sections
        self.section_headers = {
            'education': ['education', 'academic background', 'qualifications'],
            'experience': ['experience', 'work history', 'employment'],
            'skills': ['skills', 'technical skills', 'competencies'],
            'publications': ['publications', 'research papers', 'articles']
        }
        
        # Common academic fields
        self.fields = [
            "Computer Science",
            "Engineering",
            "Mathematics",
            "Physics",
            "Chemistry",
            "Biology",
            "Education",
            "Business",
            "Economics",
            "Law"
        ]

    def extract_name(self, text: str) -> str:
        """Extract name from the resume text."""
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            match = re.match(self.name_pattern, line.strip())
            if match:
                return match.group(1)
        return ""

    def extract_email(self, text: str) -> str:
        """Extract email from the resume text."""
        match = re.search(self.email_pattern, text)
        return match.group() if match else ""

    def extract_section(self, text: str, section_keywords: List[str]) -> str:
        """Extract a specific section from the text based on keywords."""
        lines = text.split('\n')
        section_text = []
        in_section = False
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if this line starts a section we're looking for
            if any(keyword in line_lower for keyword in section_keywords):
                in_section = True
                continue
            
            # Check if we've reached the next section
            if in_section and line.strip() and any(any(keyword in line_lower for keyword in keywords) 
                                                 for keywords in self.section_headers.values()):
                break
            
            if in_section and line.strip():
                section_text.append(line.strip())
        
        return '\n'.join(section_text)

    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information from the resume text."""
        education = []
        edu_text = self.extract_section(text, self.section_headers['education'])
        
        if not edu_text:
            return education
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in edu_text.split('\n\n') if p.strip()]
        
        for para in paragraphs:
            degree_match = re.search(self.degree_pattern, para)
            year_match = re.search(self.year_pattern, para)
            institution_match = re.search(self.institution_pattern, para)
            
            if degree_match:
                edu_entry = {
                    'degree': degree_match.group(),
                    'year': year_match.group() if year_match else "",
                    'institution': institution_match.group() if institution_match else "",
                }
                
                # Try to find the field of study
                for field in self.fields:
                    if field.lower() in para.lower():
                        edu_entry['field'] = field
                        break
                
                education.append(edu_entry)
        
        return education

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from the resume text."""
        skills_text = self.extract_section(text, self.section_headers['skills'])
        if not skills_text:
            return []
        
        # Split on common delimiters
        skills = re.split(r'[,;•\n]', skills_text)
        return [skill.strip() for skill in skills if skill.strip()]

    def extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience from the resume text."""
        experience = []
        exp_text = self.extract_section(text, self.section_headers['experience'])
        
        if not exp_text:
            return experience
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in exp_text.split('\n\n') if p.strip()]
        
        for para in paragraphs:
            # Try to extract year
            year_match = re.search(self.year_pattern, para)
            
            # Try to extract institution/organization
            institution_match = re.search(self.institution_pattern, para)
            
            exp_entry = {
                'description': para,
                'year': year_match.group() if year_match else "",
                'institution': institution_match.group() if institution_match else ""
            }
            
            experience.append(exp_entry)
        
        return experience

    def extract_publications(self, text: str) -> List[str]:
        """Extract publications from the resume text."""
        pub_text = self.extract_section(text, self.section_headers['publications'])
        if not pub_text:
            return []
        
        # Split into individual publications
        publications = [p.strip() for p in pub_text.split('\n') if p.strip()]
        return publications

    def parse_resume(self, text: str) -> Dict[str, Any]:
        """Parse a resume and extract structured information."""
        return {
            'name': self.extract_name(text),
            'email': self.extract_email(text),
            'education': self.extract_education(text),
            'skills': self.extract_skills(text),
            'experience': self.extract_experience(text),
            'publications': self.extract_publications(text)
        }

    def process_resume_dataset(self, input_path: str, output_path: str):
        """Process a dataset of resumes and save structured data."""
        all_resumes = []
        
        if os.path.isdir(input_path):
            resume_files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
            for file in tqdm(resume_files, desc="Parsing resumes"):
                try:
                    with open(os.path.join(input_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    parsed_data = self.parse_resume(text)
                    parsed_data['filename'] = file
                    all_resumes.append(parsed_data)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
        else:
            # Assume it's a CSV with resume texts
            df = pd.read_csv(input_path)
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing resumes"):
                try:
                    text = row['resume_text']  # Adjust column name as needed
                    parsed_data = self.parse_resume(text)
                    parsed_data['id'] = idx
                    all_resumes.append(parsed_data)
                except Exception as e:
                    print(f"Error processing resume {idx}: {str(e)}")
        
        # Convert to DataFrame and save
        df = pd.json_normalize(all_resumes)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"Parsed {len(all_resumes)} resumes and saved to {output_path}")
        return df

if __name__ == "__main__":
    parser = ResumeParser()
    parser.process_resume_dataset(
        'data/raw/resumes/nigerian_academic_resumes.csv',
        'data/processed/parsed_resumes.csv'
    ) 