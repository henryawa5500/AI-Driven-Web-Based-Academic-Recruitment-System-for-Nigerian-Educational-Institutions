import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import json
import os

class JobMatcher:
    def __init__(self, method='tfidf'):
        """
        Initialize the job matcher.
        
        Args:
            method (str): 'tfidf' or 'bert' for embedding method
        """
        self.method = method
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        elif method == 'bert':
            # Load pre-trained BERT model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased")
        else:
            raise ValueError("Method must be 'tfidf' or 'bert'")
    
    def prepare_resume_text(self, resume_row):
        """
        Prepare a consolidated text representation of a resume.
        
        Args:
            resume_row: Row from the parsed resumes DataFrame
        
        Returns:
            str: Consolidated text representation
        """
        # Convert education to text
        education_text = ""
        if isinstance(resume_row.get('education'), str):
            try:
                education = json.loads(resume_row['education'])
                for edu in education:
                    degree = edu.get('degree', '')
                    subject = edu.get('subject', '')
                    institution = edu.get('institution', '')
                    education_text += f"{degree} {subject} {institution} "
            except:
                education_text = str(resume_row.get('education', ''))
        
        # Convert skills to text
        skills_text = ""
        if isinstance(resume_row.get('skills'), str):
            try:
                skills = json.loads(resume_row['skills'])
                skills_text = " ".join(skills)
            except:
                skills_text = str(resume_row.get('skills', ''))
        
        # Convert experience to text
        experience_text = ""
        if isinstance(resume_row.get('experience'), str):
            try:
                experience = json.loads(resume_row['experience'])
                experience_text = " ".join(experience)
            except:
                experience_text = str(resume_row.get('experience', ''))
        
        # Combine everything
        full_text = f"""
        {resume_row.get('name', '')}
        {education_text}
        {skills_text}
        {experience_text}
        {resume_row.get('publications', '')}
        """
        
        return full_text.strip()
    
    def prepare_job_text(self, job_row):
        """
        Prepare a consolidated text representation of a job posting.
        
        Args:
            job_row: Row from the parsed jobs DataFrame
        
        Returns:
            str: Consolidated text representation
        """
        # Convert qualifications to text
        qualifications_text = ""
        if isinstance(job_row.get('qualifications'), str):
            try:
                qualifications = json.loads(job_row['qualifications'])
                qualifications_text = " ".join(qualifications)
            except:
                qualifications_text = str(job_row.get('qualifications', ''))
        
        # Convert responsibilities to text
        responsibilities_text = ""
        if isinstance(job_row.get('responsibilities'), str):
            try:
                responsibilities = json.loads(job_row['responsibilities'])
                responsibilities_text = " ".join(responsibilities)
            except:
                responsibilities_text = str(job_row.get('responsibilities', ''))
        
        # Combine everything
        full_text = f"""
        {job_row.get('title', '')}
        {job_row.get('institution', '')}
        {qualifications_text}
        {responsibilities_text}
        {job_row.get('experience', '')}
        """
        
        return full_text.strip()
    
    def get_bert_embedding(self, text):
        """Generate BERT embedding for a text."""
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", 
                               max_length=512, truncation=True, padding=True)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the [CLS] token embedding as the document representation
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embedding
    
    def compute_similarity(self, resume_text, job_text):
        """Compute similarity between resume and job description."""
        if self.method == 'tfidf':
            # Fit and transform the texts
            texts = [resume_text, job_text]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Compute cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
        elif self.method == 'bert':
            # Get BERT embeddings
            resume_embedding = self.get_bert_embedding(resume_text)
            job_embedding = self.get_bert_embedding(job_text)
            
            # Compute cosine similarity
            similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        
        return similarity
    
    def match_resume_to_job(self, resume_row, job_row):
        """
        Match a single resume to a single job posting.
        
        Args:
            resume_row: Row from the parsed resumes DataFrame
            job_row: Row from the parsed jobs DataFrame
        
        Returns:
            float: Similarity score between 0 and 1
        """
        resume_text = self.prepare_resume_text(resume_row)
        job_text = self.prepare_job_text(job_row)
        
        return self.compute_similarity(resume_text, job_text)
    
    def match_resumes_to_job(self, resumes_df, job_row, top_n=10):
        """
        Match multiple resumes to a single job posting.
        
        Args:
            resumes_df: DataFrame of parsed resumes
            job_row: Row from the parsed jobs DataFrame
            top_n: Number of top matches to return
        
        Returns:
            DataFrame: Top matching resumes with similarity scores
        """
        # Prepare job text once
        job_text = self.prepare_job_text(job_row)
        
        # Compute similarities for all resumes
        similarities = []
        for _, resume_row in resumes_df.iterrows():
            resume_text = self.prepare_resume_text(resume_row)
            similarity = self.compute_similarity(resume_text, job_text)
            similarities.append(similarity)
        
        # Add similarity scores to resumes DataFrame
        result_df = resumes_df.copy()
        result_df['similarity_score'] = similarities
        
        # Sort by similarity score and return top N
        return result_df.sort_values('similarity_score', ascending=False).head(top_n)
    
    def match_jobs_to_resume(self, jobs_df, resume_row, top_n=10):
        """
        Match multiple jobs to a single resume.
        
        Args:
            jobs_df: DataFrame of parsed jobs
            resume_row: Row from the parsed resumes DataFrame
            top_n: Number of top matches to return
        
        Returns:
            DataFrame: Top matching jobs with similarity scores
        """
        # Prepare resume text once
        resume_text = self.prepare_resume_text(resume_row)
        
        # Compute similarities for all jobs
        similarities = []
        for _, job_row in jobs_df.iterrows():
            job_text = self.prepare_job_text(job_row)
            similarity = self.compute_similarity(resume_text, job_text)
            similarities.append(similarity)
        
        # Add similarity scores to jobs DataFrame
        result_df = jobs_df.copy()
        result_df['similarity_score'] = similarities
        
        # Sort by similarity score and return top N
        return result_df.sort_values('similarity_score', ascending=False).head(top_n)

if __name__ == "__main__":
    # Example usage
    matcher = JobMatcher(method='tfidf')
    
    # Load parsed data
    resumes_df = pd.read_csv('data/processed/parsed_resumes.csv')
    jobs_df = pd.read_csv('data/processed/parsed_job_listings.csv')
    
    # Match resumes to a specific job
    job_row = jobs_df.iloc[0]
    top_matches = matcher.match_resumes_to_job(resumes_df, job_row, top_n=5)
    print("\nTop 5 matching resumes for job:", job_row['title'])
    print(top_matches[['name', 'similarity_score']])
    
    # Match jobs to a specific resume
    resume_row = resumes_df.iloc[0]
    top_jobs = matcher.match_jobs_to_resume(jobs_df, resume_row, top_n=5)
    print("\nTop 5 matching jobs for resume:", resume_row['name'])
    print(top_jobs[['title', 'institution', 'similarity_score']]) 