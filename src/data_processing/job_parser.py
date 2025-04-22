import pandas as pd
import re
import os
from tqdm import tqdm
from typing import Dict, List, Optional, Any

class JobParser:
    def __init__(self):
        """Initialize the JobParser."""
        # Regex patterns
        self.title_pattern = r'(?:Professor|Lecturer|Researcher|Assistant|Associate)(?:[^,\n]*)'
        self.department_pattern = r'(?:Department\s+of\s+)?([A-Z][a-zA-Z\s]+(?:Science|Engineering|Studies|Technology|Arts|Education|Medicine|Law))'
        self.institution_pattern = r'(?:at|in)\s+([^,\n]+(?:University|College|Institute|School))'
        
        # Section identifiers
        self.section_headers = {
            "qualifications": ["qualifications", "requirements", "required qualifications"],
            "responsibilities": ["responsibilities", "duties", "key responsibilities"],
            "requirements": ["requirements", "essential requirements", "criteria"]
        }
        
        # Common fields/subjects
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

    def extract_title(self, text: str) -> str:
        """Extract job title from text."""
        match = re.search(self.title_pattern, text)
        return match.group().strip() if match else ""

    def extract_department(self, text: str) -> str:
        """Extract department from text."""
        match = re.search(self.department_pattern, text)
        return match.group(1).strip() if match else ""

    def extract_institution(self, text: str) -> str:
        """Extract institution from text."""
        match = re.search(self.institution_pattern, text)
        return match.group(1).strip() if match else ""

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

    def split_bullet_points(self, text: str) -> List[str]:
        """Split text into bullet points."""
        if not text:
            return []
        
        # Split on bullet points, dashes, or numbered lists
        points = re.split(r'\s*[\u2022\-\*]\s*|\d+\.\s+', text)
        return [point.strip() for point in points if point.strip()]

    def extract_qualifications(self, text: str) -> List[str]:
        """Extract qualifications from text."""
        qual_text = self.extract_section(text, self.section_headers["qualifications"])
        return self.split_bullet_points(qual_text)

    def extract_responsibilities(self, text: str) -> List[str]:
        """Extract responsibilities from text."""
        resp_text = self.extract_section(text, self.section_headers["responsibilities"])
        return self.split_bullet_points(resp_text)

    def extract_requirements(self, text: str) -> List[str]:
        """Extract requirements from text."""
        req_text = self.extract_section(text, self.section_headers["requirements"])
        return self.split_bullet_points(req_text)

    def parse_job(self, text: str) -> Dict[str, Any]:
        """Parse a job description into structured data."""
        return {
            'title': self.extract_title(text),
            'department': self.extract_department(text),
            'institution': self.extract_institution(text),
            'qualifications': self.extract_qualifications(text),
            'responsibilities': self.extract_responsibilities(text),
            'requirements': self.extract_requirements(text)
        }

    def process_job_dataset(self, input_path: str, output_path: str):
        """Process a dataset of job descriptions and save structured data."""
        all_jobs = []
        
        if os.path.isdir(input_path):
            job_files = [f for f in os.listdir(input_path) if f.endswith('.txt')]
            for file in tqdm(job_files, desc="Parsing jobs"):
                try:
                    with open(os.path.join(input_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    parsed_data = self.parse_job(text)
                    parsed_data['filename'] = file
                    all_jobs.append(parsed_data)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
        else:
            # Assume it's a CSV with job descriptions
            df = pd.read_csv(input_path)
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing jobs"):
                try:
                    text = row['description']  # Adjust column name as needed
                    parsed_data = self.parse_job(text)
                    parsed_data['id'] = idx
                    all_jobs.append(parsed_data)
                except Exception as e:
                    print(f"Error processing job {idx}: {str(e)}")
        
        # Convert to DataFrame and save
        df = pd.json_normalize(all_jobs)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"Parsed {len(all_jobs)} jobs and saved to {output_path}")
        return df

if __name__ == "__main__":
    parser = JobParser()
    parser.process_job_dataset(
        'data/raw/job_listings/nigerian_academic_jobs.csv',
        'data/processed/parsed_job_listings.csv'
    ) 