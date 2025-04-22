from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
from src.data_processing.resume_parser import ResumeParser
from src.data_processing.job_parser import JobParser
from src.models.matcher import JobMatcher
from src.models.bias_detector import BiasDetector
import pandas as pd
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recruitment.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    name = db.Column(db.String(100))
    role = db.Column(db.String(20))  # 'admin', 'recruiter', 'candidate'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    institution = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    requirements = db.Column(db.Text)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='open')  # 'open', 'closed'

class Application(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'))
    candidate_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    resume_path = db.Column(db.String(500))
    status = db.Column(db.String(20), default='pending')  # 'pending', 'reviewed', 'accepted', 'rejected'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    similarity_score = db.Column(db.Float)
    bias_analysis = db.Column(db.Text)  # JSON string of bias analysis results

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize parsers and models
resume_parser = ResumeParser()
job_parser = JobParser()
job_matcher = JobMatcher(method='tfidf')
bias_detector = BiasDetector()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        role = request.form.get('role')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        user = User(email=email, name=name, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'candidate':
        applications = Application.query.filter_by(candidate_id=current_user.id).all()
        return render_template('candidate_dashboard.html', applications=applications)
    elif current_user.role == 'recruiter':
        jobs = Job.query.filter_by(created_by=current_user.id).all()
        return render_template('recruiter_dashboard.html', jobs=jobs)
    else:  # admin
        return render_template('admin_dashboard.html')

@app.route('/jobs')
def jobs():
    jobs = Job.query.filter_by(status='open').all()
    return render_template('jobs.html', jobs=jobs)

@app.route('/job/<int:job_id>')
def job_detail(job_id):
    job = Job.query.get_or_404(job_id)
    return render_template('job_detail.html', job=job)

@app.route('/apply/<int:job_id>', methods=['GET', 'POST'])
@login_required
def apply(job_id):
    if current_user.role != 'candidate':
        flash('Only candidates can apply for jobs')
        return redirect(url_for('jobs'))
    
    job = Job.query.get_or_404(job_id)
    
    if request.method == 'POST':
        if 'resume' not in request.files:
            flash('No resume file uploaded')
            return redirect(request.url)
        
        resume_file = request.files['resume']
        if resume_file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # Save resume
        resume_path = os.path.join('uploads', f"{current_user.id}_{job_id}_{resume_file.filename}")
        os.makedirs('uploads', exist_ok=True)
        resume_file.save(resume_path)
        
        # Parse resume
        with open(resume_path, 'r', encoding='utf-8', errors='ignore') as f:
            resume_text = f.read()
        parsed_resume = resume_parser.parse_resume(resume_text)
        
        # Parse job
        job_text = f"{job.title} {job.description} {job.requirements}"
        parsed_job = job_parser.parse_job(job_text)
        
        # Match resume to job
        similarity_score = job_matcher.match_resume_to_job(parsed_resume, parsed_job)
        
        # Detect bias
        features_df = bias_detector.prepare_features(
            pd.DataFrame([parsed_resume]),
            pd.DataFrame([parsed_job]),
            pd.DataFrame([{'resume_id': 0, 'job_id': 0, 'similarity_score': similarity_score}])
        )
        bias_detector.train_bias_model(features_df)
        bias_analysis = bias_detector.detect_bias_patterns(features_df)
        
        # Create application
        application = Application(
            job_id=job_id,
            candidate_id=current_user.id,
            resume_path=resume_path,
            similarity_score=similarity_score,
            bias_analysis=json.dumps(bias_analysis)
        )
        db.session.add(application)
        db.session.commit()
        
        flash('Application submitted successfully')
        return redirect(url_for('dashboard'))
    
    return render_template('apply.html', job=job)

@app.route('/admin/bias-analysis')
@login_required
def bias_analysis():
    if current_user.role != 'admin':
        flash('Only admins can access bias analysis')
        return redirect(url_for('dashboard'))
    
    applications = Application.query.all()
    return render_template('bias_analysis.html', applications=applications)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 