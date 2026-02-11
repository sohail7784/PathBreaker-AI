# PathBreaker AI - Design Document

## 1. System Architecture Overview

PathBreaker AI follows a modern microservices architecture with clear separation of concerns across multiple layers:

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Layer                          │
│              React Dashboard + Tailwind CSS                 │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTPS
┌─────────────────────────────────────────────────────────────┐
│                   API Gateway Layer                         │
│              FastAPI + JWT Authentication                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   AI Engine Layer                           │
│   AWS Bedrock (Claude) + Custom ML Models + LangChain      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Database Layer                            │
│        MongoDB + PostgreSQL + Redis + AWS S3               │
└─────────────────────────────────────────────────────────────┘
```

### Architecture Principles

- **Separation of Concerns**: Each layer has distinct responsibilities
- **Scalability**: Horizontal scaling at each layer independently
- **Security**: Defense in depth with multiple security layers
- **Performance**: Caching and async processing for optimal response times
- **Maintainability**: Modular design for easy updates and debugging

## 2. Component Architecture

### Frontend Components


#### Core Dashboard Component
```
src/
├── components/
│   ├── Dashboard/
│   │   ├── DashboardLayout.jsx (Main layout with navigation)
│   │   ├── FeatureCard.jsx (4 feature module cards)
│   │   ├── ProgressTracker.jsx (Overall progress visualization)
│   │   └── QuickActions.jsx (Shortcuts to common tasks)
│   │
│   ├── PathFinder/
│   │   ├── CareerQuiz.jsx (Interactive assessment)
│   │   ├── QuizQuestion.jsx (Individual question component)
│   │   ├── RoadmapDisplay.jsx (Visual learning path)
│   │   ├── CourseCard.jsx (Course recommendations)
│   │   ├── SkillTracker.jsx (Progress monitoring)
│   │   └── CareerComparison.jsx (Side-by-side career analysis)
│   │
│   ├── EscapeMatrix/
│   │   ├── StrategySelector.jsx (Input form for capital/time/skills)
│   │   ├── EarningCalculator.jsx (ROI and timeline calculator)
│   │   ├── StrategyCard.jsx (Individual earning strategy)
│   │   ├── StepByStepGuide.jsx (Detailed implementation guide)
│   │   ├── SuccessStories.jsx (Case studies and testimonials)
│   │   └── ResourceLinks.jsx (External platform links)
│   │
│   ├── CareerAccelerator/
│   │   ├── ResumeBuilder.jsx (Drag-drop resume creator)
│   │   ├── ResumeUploader.jsx (File upload with preview)
│   │   ├── ATSScoreCard.jsx (Score display with improvements)
│   │   ├── JobSearch.jsx (Advanced search with filters)
│   │   ├── JobCard.jsx (Individual job listing)
│   │   ├── InterviewPrep.jsx (Mock interview interface)
│   │   ├── QuestionBank.jsx (Company-specific questions)
│   │   └── SalaryNegotiator.jsx (Data-driven insights)
│   │
│   ├── CodeMitra/
│   │   ├── CodeEditor.jsx (Monaco editor integration)
│   │   ├── ErrorExplainer.jsx (AI-powered error breakdown)
│   │   ├── DebugAssistant.jsx (Step-by-step debugging)
│   │   ├── CodeImprover.jsx (Optimization suggestions)
│   │   ├── DSASolver.jsx (Problem solving with approaches)
│   │   └── LearningResources.jsx (Related tutorials)
│   │
│   └── Shared/
│       ├── Header.jsx (Navigation bar)
│       ├── Sidebar.jsx (Feature navigation)
│       ├── ChatBot.jsx (AI assistant overlay)
│       ├── Notification.jsx (Toast notifications)
│       └── LoadingSpinner.jsx (Loading states)
```

### Backend Services

#### Authentication Service
```python
# auth_service.py
- User registration with email verification
- Login with JWT token generation
- Token refresh mechanism
- Password reset flow
- OAuth integration (Google, LinkedIn)
- Session management
```

#### PathFinder Service
```python
# pathfinder_service.py
- Career assessment quiz logic
- AI-powered career matching algorithm
- Learning roadmap generation
- Course curation from multiple platforms
- Skill gap analysis
- Progress tracking and analytics
```

#### Escape Matrix Service
```python
# escape_matrix_service.py
- User profile analysis (capital, time, skills)
- Earning strategy recommendation engine
- ROI calculation for different strategies
- Step-by-step guide generation
- Success tracking and milestones
- Resource aggregation
```

#### Career Service
```python
# career_service.py
- Resume parsing and text extraction
- ATS scoring algorithm (0-100 scale)
- Job matching based on skills and preferences
- Job board scraping and aggregation
- Interview question bank management
- Application tracking system
```

#### CodeMitra Service
```python
# codemitra_service.py
- Code syntax analysis
- Error detection and classification
- AI-powered explanation generation
- Debug suggestion algorithm
- Code optimization recommendations
- DSA problem solving with multiple approaches
```

### AI Services Architecture

#### Conversational AI (AWS Bedrock Claude)
```python
# bedrock_service.py
import boto3
from langchain.llms import Bedrock

class ConversationalAI:
    def __init__(self):
        self.client = boto3.client('bedrock-runtime')
        self.model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
    
    def career_counseling(self, user_context, query):
        # Context-aware career guidance
        pass
    
    def generate_roadmap(self, career_path, user_level):
        # Personalized learning path generation
        pass
    
    def explain_code_error(self, code, error):
        # Simple language error explanation
        pass
```

#### Resume Parser (AWS Rekognition + Textract)
```python
# resume_parser.py
import boto3

class ResumeParser:
    def __init__(self):
        self.textract = boto3.client('textract')
        self.comprehend = boto3.client('comprehend')
    
    def extract_text(self, resume_file):
        # Extract text from PDF/image
        pass
    
    def parse_sections(self, text):
        # Identify education, experience, skills
        pass
    
    def extract_skills(self, text):
        # NLP-based skill extraction
        pass
```

#### Job Matcher (Custom ML Model)
```python
# job_matcher.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class JobMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = self.load_model()
    
    def match_jobs(self, user_skills, job_listings):
        # Collaborative filtering + skill matching
        pass
    
    def calculate_fit_score(self, user_profile, job_requirements):
        # 0-100 compatibility score
        pass
```

#### Skill Assessor (Custom ML Model)
```python
# skill_assessor.py
from transformers import pipeline

class SkillAssessor:
    def __init__(self):
        self.nlp = pipeline('text-classification')
    
    def assess_quiz_response(self, question, answer):
        # Evaluate understanding level
        pass
    
    def generate_skill_report(self, user_responses):
        # Comprehensive skill analysis
        pass
```

## 3. Data Flow Diagrams

### User Journey Flow

```
┌──────────┐
│ Student  │
└────┬─────┘
     │
     ▼
┌─────────────────┐
│ Assessment Quiz │ (10-15 questions)
└────┬────────────┘
     │
     ▼
┌──────────────────┐
│  AI Analysis     │ (AWS Bedrock + ML Models)
└────┬─────────────┘
     │
     ▼
┌──────────────────────────────────────┐
│      Feature Selection               │
│  ┌──────────┬──────────┬──────────┐ │
│  │PathFinder│  Escape  │  Career  │ │
│  │   Pro    │  Matrix  │Accelerator│ │
│  └──────────┴──────────┴──────────┘ │
└────┬─────────────────────────────────┘
     │
     ▼
┌──────────────────┐
│ Personalized     │
│ Dashboard Output │
└──────────────────┘
```

### Feature-specific Flows

#### PathFinder Flow
```
User Input                AI Processing           Output
─────────────────────────────────────────────────────────
Education Level    →                         →  Career Options
Interests          →   Career Matching       →  Salary Insights
Skills             →   Algorithm             →  Growth Potential
Location           →                         →
                                                     ↓
                                              Learning Roadmap
                                              ─────────────────
                                              Basic Courses
                                              Intermediate
                                              Advanced
                                              Certifications
                                                     ↓
                                              Progress Tracking
                                              ─────────────────
                                              Completed: 30%
                                              Next Milestone
                                              Estimated Time
```

#### Escape Matrix Flow
```
User Input              AI Analysis            Strategy Output
──────────────────────────────────────────────────────────────
Capital: ₹5,000    →                      →  Freelancing
Time: 10 hrs/week  →  Strategy Matcher   →  Content Creation
Skills: Writing    →  Algorithm          →  E-commerce
                                                    ↓
                                          Step-by-step Guide
                                          ──────────────────
                                          Week 1: Setup
                                          Week 2: First Client
                                          Week 3: Scale
                                                    ↓
                                          Success Tracking
                                          ────────────────
                                          Earnings: ₹12,000
                                          Clients: 5
                                          Rating: 4.8/5
```

#### Career Accelerator Flow
```
Resume Upload       Parsing & Analysis      Job Matching
────────────────────────────────────────────────────────
PDF/DOCX File  →   Text Extraction    →   Skill Match
                   Section Detection       Location Filter
                   Skill Extraction        Salary Range
                          ↓                      ↓
                   ATS Scoring          Job Recommendations
                   ───────────          ───────────────────
                   Score: 78/100        50 Matching Jobs
                   Missing Keywords     Sorted by Fit
                   Improvement Tips     Apply Directly
                          ↓                      ↓
                   Interview Prep       Application Tracking
                   ──────────────       ────────────────────
                   Mock Interview       Applied: 15
                   Question Bank        Interviews: 3
                   Negotiation Tips     Offers: 1
```

#### CodeMitra Flow
```
Code Input          Error Detection       AI Explanation
──────────────────────────────────────────────────────────
Python/JS Code  →  Syntax Analysis   →  Simple Language
Error Message   →  Error Type        →  Why It Happened
                   Classification         How to Fix
                          ↓                     ↓
                   Solution Suggestions   Learning Resources
                   ────────────────────   ──────────────────
                   Approach 1 (Simple)    Related Tutorial
                   Approach 2 (Optimal)   Documentation
                   Best Practices         Similar Problems
```

## 4. Database Schema

### User Collection (MongoDB)
```javascript
{
  _id: ObjectId,
  user_id: "UUID",
  email: "user@example.com",
  password_hash: "bcrypt_hash",
  created_at: ISODate,
  last_login: ISODate,
  
  profile: {
    full_name: "John Doe",
    age: 21,
    education: "B.Tech CSE",
    location: "Mumbai, India",
    phone: "+91-XXXXXXXXXX",
    languages: ["English", "Hindi"]
  },
  
  career_path: {
    selected_path: "Full Stack Developer",
    current_level: "Intermediate",
    target_role: "Senior Developer",
    estimated_completion: "6 months"
  },
  
  learning_progress: {
    completed_courses: [
      {
        course_id: "course_123",
        title: "React Fundamentals",
        completed_date: ISODate,
        certificate_url: "s3://..."
      }
    ],
    current_courses: ["course_456"],
    skills_acquired: ["React", "Node.js", "MongoDB"],
    quiz_scores: {
      "quiz_1": 85,
      "quiz_2": 92
    },
    total_learning_hours: 120
  },
  
  earnings_tracker: {
    strategies_tried: ["Freelancing", "Content Creation"],
    current_earnings: 15000,
    earnings_history: [
      {
        month: "2026-01",
        amount: 8000,
        source: "Freelancing"
      }
    ],
    milestones_achieved: ["First Client", "₹10K Earned"]
  },
  
  job_applications: {
    total_applied: 25,
    interviews_scheduled: 5,
    offers_received: 2,
    applications: [
      {
        job_id: "job_789",
        applied_date: ISODate,
        status: "Interview Scheduled",
        notes: "Technical round on 15th Feb"
      }
    ]
  },
  
  preferences: {
    notification_enabled: true,
    language: "en",
    theme: "light"
  }
}
```

### Jobs Collection (PostgreSQL)
```sql
CREATE TABLE jobs (
    job_id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    company VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    job_type VARCHAR(50), -- Full-time, Part-time, Internship
    experience_required VARCHAR(50), -- 0-2 years, 2-5 years
    
    description TEXT,
    required_skills TEXT[], -- Array of skills
    preferred_skills TEXT[],
    
    salary_min INTEGER,
    salary_max INTEGER,
    currency VARCHAR(10) DEFAULT 'INR',
    
    platform VARCHAR(100), -- AngelList, Wellfound, Naukri
    external_url TEXT,
    company_logo_url TEXT,
    
    posted_date TIMESTAMP,
    expires_date TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_skills (required_skills),
    INDEX idx_location (location),
    INDEX idx_posted_date (posted_date)
);
```

### Courses Collection (PostgreSQL)
```sql
CREATE TABLE courses (
    course_id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    platform VARCHAR(100), -- Udemy, Coursera, YouTube
    external_url TEXT NOT NULL,
    thumbnail_url TEXT,
    
    instructor VARCHAR(255),
    duration_hours DECIMAL(5,2),
    rating DECIMAL(3,2),
    num_reviews INTEGER,
    
    skill_category VARCHAR(100), -- Web Dev, AI/ML, Data Science
    difficulty_level VARCHAR(50), -- Beginner, Intermediate, Advanced
    
    is_free BOOLEAN DEFAULT true,
    price DECIMAL(10,2),
    currency VARCHAR(10) DEFAULT 'INR',
    
    certification_available BOOLEAN DEFAULT false,
    language VARCHAR(50) DEFAULT 'English',
    
    description TEXT,
    learning_outcomes TEXT[],
    prerequisites TEXT[],
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_category (skill_category),
    INDEX idx_difficulty (difficulty_level),
    INDEX idx_free (is_free)
);
```

### Career Paths Collection (PostgreSQL)
```sql
CREATE TABLE career_paths (
    path_id SERIAL PRIMARY KEY,
    path_name VARCHAR(255) NOT NULL,
    category VARCHAR(100), -- Tech, Business, Creative
    
    description TEXT,
    average_salary_entry INTEGER,
    average_salary_mid INTEGER,
    average_salary_senior INTEGER,
    
    growth_potential VARCHAR(50), -- High, Medium, Low
    job_demand VARCHAR(50), -- High, Medium, Low
    
    required_skills TEXT[],
    recommended_courses INTEGER[], -- Foreign key to courses
    
    learning_roadmap JSONB, -- Structured roadmap data
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Files Storage (AWS S3)
```
s3://pathbreaker-ai/
├── resumes/
│   └── {user_id}/
│       ├── original_resume.pdf
│       ├── optimized_resume.pdf
│       └── resume_versions/
├── certificates/
│   └── {user_id}/
│       └── {course_id}_certificate.pdf
├── portfolio/
│   └── {user_id}/
│       ├── project_screenshots/
│       └── project_files/
└── user_uploads/
    └── {user_id}/
        └── misc_documents/
```


## 5. AI Model Integration

### AWS Bedrock Integration

#### Configuration
```python
# config/bedrock_config.py
import boto3
from langchain.llms import Bedrock
from langchain.prompts import PromptTemplate

class BedrockConfig:
    def __init__(self):
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        self.model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        self.max_tokens = 2048
        self.temperature = 0.7
    
    def get_llm(self):
        return Bedrock(
            client=self.client,
            model_id=self.model_id,
            model_kwargs={
                'max_tokens': self.max_tokens,
                'temperature': self.temperature
            }
        )
```

#### Use Cases

1. **Career Counseling Chatbot**
```python
# services/career_counselor.py
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

class CareerCounselor:
    def __init__(self, bedrock_llm):
        self.llm = bedrock_llm
        self.memory = ConversationBufferMemory()
        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory
        )
    
    def get_career_advice(self, user_query, user_context):
        prompt = f"""
        User Profile: {user_context}
        Question: {user_query}
        
        Provide personalized career advice considering:
        - User's education and skills
        - Current job market trends in India
        - Realistic career paths
        - Actionable next steps
        """
        return self.chain.predict(input=prompt)
```

2. **Learning Roadmap Generation**
```python
def generate_roadmap(self, career_path, current_level):
    prompt = f"""
    Generate a detailed learning roadmap for {career_path}.
    Current Level: {current_level}
    
    Include:
    - 3-month milestones
    - Specific skills to learn
    - Free course recommendations
    - Project ideas for practice
    - Estimated time commitment
    
    Format as JSON with structured data.
    """
    return self.llm.predict(prompt)
```

3. **Code Error Explanation**
```python
def explain_error(self, code, error_message, language):
    prompt = f"""
    Explain this {language} error in simple language for beginners:
    
    Code:
    {code}
    
    Error:
    {error_message}
    
    Provide:
    1. What went wrong (in simple terms)
    2. Why it happened
    3. How to fix it (step-by-step)
    4. Best practices to avoid this
    """
    return self.llm.predict(prompt)
```

#### API Rate Limiting and Cost Optimization
```python
# utils/rate_limiter.py
from functools import wraps
import time
from cachetools import TTLCache

class BedrockRateLimiter:
    def __init__(self):
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        self.request_count = 0
        self.max_requests_per_minute = 50
    
    def rate_limit(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check cache first
            cache_key = str(args) + str(kwargs)
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Rate limiting logic
            if self.request_count >= self.max_requests_per_minute:
                time.sleep(60)
                self.request_count = 0
            
            result = func(*args, **kwargs)
            self.cache[cache_key] = result
            self.request_count += 1
            return result
        return wrapper
```

### Custom ML Models

#### Job Matching Model
```python
# ml_models/job_matcher.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

class JobMatchingModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.load_model()
    
    def train(self, user_profiles, job_listings, match_scores):
        """
        Train on historical data:
        - User profiles (skills, experience)
        - Job listings (requirements)
        - Match scores (successful placements)
        """
        # Feature engineering
        user_features = self.extract_user_features(user_profiles)
        job_features = self.extract_job_features(job_listings)
        
        # Train collaborative filtering model
        # Save model
        joblib.dump(self.model, 'models/job_matcher.pkl')
    
    def predict_match_score(self, user_profile, job_listing):
        """
        Returns 0-100 compatibility score
        """
        user_vec = self.vectorizer.transform([user_profile])
        job_vec = self.vectorizer.transform([job_listing])
        
        similarity = cosine_similarity(user_vec, job_vec)[0][0]
        score = int(similarity * 100)
        
        return score
    
    def get_top_matches(self, user_profile, job_listings, top_k=10):
        """
        Returns top K matching jobs
        """
        scores = []
        for job in job_listings:
            score = self.predict_match_score(user_profile, job)
            scores.append((job, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

#### Career Path Recommendation Model
```python
# ml_models/career_recommender.py
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class CareerRecommender:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=10)
        self.career_paths = self.load_career_paths()
    
    def train(self, training_data):
        """
        Training data includes:
        - User attributes (education, interests, skills)
        - Chosen career path
        - Success metrics (satisfaction, salary growth)
        """
        X = training_data[['education_level', 'math_score', 
                          'coding_score', 'creativity_score']]
        y = training_data['career_path']
        
        self.model.fit(X, y)
    
    def recommend_careers(self, user_assessment):
        """
        Returns top 5 career recommendations with confidence scores
        """
        features = self.extract_features(user_assessment)
        probabilities = self.model.predict_proba([features])[0]
        
        recommendations = []
        for idx, prob in enumerate(probabilities):
            career = self.career_paths[idx]
            recommendations.append({
                'career': career,
                'confidence': prob,
                'salary_range': self.get_salary_range(career),
                'growth_potential': self.get_growth_potential(career)
            })
        
        return sorted(recommendations, 
                     key=lambda x: x['confidence'], 
                     reverse=True)[:5]
```

#### Skill Assessment Model
```python
# ml_models/skill_assessor.py
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

class SkillAssessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.classifier = pipeline('text-classification')
    
    def assess_answer(self, question, user_answer, correct_answer):
        """
        Evaluates user's answer quality
        Returns: score (0-100), feedback
        """
        # Semantic similarity between user and correct answer
        user_embedding = self.get_embedding(user_answer)
        correct_embedding = self.get_embedding(correct_answer)
        
        similarity = torch.cosine_similarity(
            user_embedding, 
            correct_embedding, 
            dim=0
        ).item()
        
        score = int(similarity * 100)
        feedback = self.generate_feedback(score, user_answer)
        
        return {
            'score': score,
            'feedback': feedback,
            'areas_to_improve': self.identify_gaps(user_answer, correct_answer)
        }
    
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', 
                               padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
```

### RAG Implementation

```python
# services/rag_service.py
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGService:
    def __init__(self, bedrock_llm):
        self.llm = bedrock_llm
        self.embeddings = BedrockEmbeddings(
            client=boto3.client('bedrock-runtime'),
            model_id='amazon.titan-embed-text-v1'
        )
        self.vector_store = None
        self.setup_vector_store()
    
    def setup_vector_store(self):
        """
        Load and index course descriptions, job postings
        """
        # Load documents
        documents = self.load_documents()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(
            chunks, 
            self.embeddings
        )
    
    def semantic_search_courses(self, query, top_k=5):
        """
        Find most relevant courses based on user query
        """
        retriever = self.vector_store.as_retriever(
            search_kwargs={'k': top_k}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain({'query': query})
        return result
    
    def personalized_recommendations(self, user_profile):
        """
        Generate personalized course/job recommendations
        """
        query = f"""
        Find relevant opportunities for:
        Skills: {user_profile['skills']}
        Experience: {user_profile['experience']}
        Goals: {user_profile['career_goals']}
        """
        
        return self.semantic_search_courses(query)
```

## 6. Security Architecture

### Authentication & Authorization

```python
# security/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

# Configuration
SECRET_KEY = "your-secret-key-here"  # Store in environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class AuthService:
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def create_access_token(data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        try:
            token = credentials.credentials
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
            return user_id
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
```

### Data Encryption

```python
# security/encryption.py
from cryptography.fernet import Fernet
import os

class EncryptionService:
    def __init__(self):
        # Load encryption key from environment
        self.key = os.getenv('ENCRYPTION_KEY').encode()
        self.cipher = Fernet(self.key)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data before storing"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data when retrieving"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
```

### File Upload Security

```python
# security/file_validator.py
import magic
from fastapi import UploadFile, HTTPException

class FileValidator:
    ALLOWED_EXTENSIONS = {'.pdf', '.doc', '.docx'}
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    
    @staticmethod
    async def validate_resume(file: UploadFile):
        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in FileValidator.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed: {FileValidator.ALLOWED_EXTENSIONS}"
            )
        
        # Check file size
        content = await file.read()
        if len(content) > FileValidator.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 5MB limit"
            )
        
        # Verify actual file type (not just extension)
        mime = magic.from_buffer(content, mime=True)
        allowed_mimes = ['application/pdf', 'application/msword', 
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
        if mime not in allowed_mimes:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type"
            )
        
        # Reset file pointer
        await file.seek(0)
        return True
```

### API Rate Limiting

```python
# security/rate_limiter.py
from fastapi import Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Apply to routes
@app.post("/api/auth/login")
@limiter.limit("5/minute")  # 5 requests per minute
async def login(request: Request):
    pass

@app.post("/api/pathfinder/analyze")
@limiter.limit("10/hour")  # 10 requests per hour
async def analyze_career(request: Request):
    pass
```

### CORS Configuration

```python
# main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pathbreaker-ai.com",
        "https://www.pathbreaker-ai.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600
)
```

## 7. Deployment Architecture

### Development Environment

```yaml
# docker-compose.yml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
    environment:
      - REACT_APP_API_URL=http://localhost:8000
  
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
    environment:
      - DATABASE_URL=mongodb://mongo:27017/pathbreaker
      - POSTGRES_URL=postgresql://postgres:password@postgres:5432/pathbreaker
      - REDIS_URL=redis://redis:6379
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    depends_on:
      - mongo
      - postgres
      - redis
  
  mongo:
    image: mongo:6.0
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
  
  postgres:
    image: postgres:15
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=pathbreaker
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  mongo_data:
  postgres_data:
```

### Production Environment (AWS)

```
┌─────────────────────────────────────────────────────────┐
│                    CloudFront CDN                       │
│              (Global Content Delivery)                  │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌──────────────┐
│   S3 Bucket   │         │ Application  │
│  (Frontend)   │         │ Load Balancer│
└───────────────┘         └──────┬───────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌─────────┐  ┌─────────┐  ┌─────────┐
              │  EC2    │  │  EC2    │  │  EC2    │
              │Instance │  │Instance │  │Instance │
              │(Backend)│  │(Backend)│  │(Backend)│
              └────┬────┘  └────┬────┘  └────┬────┘
                   │            │            │
                   └────────────┼────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
            ┌──────────────┐        ┌─────────────┐
            │  DocumentDB  │        │  RDS        │
            │  (MongoDB)   │        │(PostgreSQL) │
            └──────────────┘        └─────────────┘
                    │
                    ▼
            ┌──────────────┐
            │ ElastiCache  │
            │   (Redis)    │
            └──────────────┘
```

#### Infrastructure as Code (Terraform)

```hcl
# terraform/main.tf
provider "aws" {
  region = "ap-south-1"  # Mumbai region
}

# VPC Configuration
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  
  tags = {
    Name = "pathbreaker-vpc"
  }
}

# EC2 Auto Scaling Group
resource "aws_autoscaling_group" "backend" {
  name                 = "pathbreaker-backend-asg"
  vpc_zone_identifier  = aws_subnet.private[*].id
  target_group_arns    = [aws_lb_target_group.backend.arn]
  health_check_type    = "ELB"
  
  min_size             = 2
  max_size             = 10
  desired_capacity     = 3
  
  launch_template {
    id      = aws_launch_template.backend.id
    version = "$Latest"
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "pathbreaker-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
}

# S3 for Frontend
resource "aws_s3_bucket" "frontend" {
  bucket = "pathbreaker-frontend"
  
  website {
    index_document = "index.html"
    error_document = "index.html"
  }
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  origin {
    domain_name = aws_s3_bucket.frontend.bucket_regional_domain_name
    origin_id   = "S3-pathbreaker-frontend"
  }
  
  enabled             = true
  default_root_object = "index.html"
  
  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-pathbreaker-frontend"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }
}
```


## 8. Scalability Design

### Horizontal Scaling Strategy

```python
# Load Balancing Configuration
"""
Traffic Distribution:
- Round-robin for general API requests
- Least connections for AI-heavy operations
- Session affinity for user-specific operations
"""

# Auto-scaling Rules
auto_scaling_rules = {
    "scale_up": {
        "metric": "CPUUtilization",
        "threshold": 70,  # percent
        "action": "add_2_instances",
        "cooldown": 300  # seconds
    },
    "scale_down": {
        "metric": "CPUUtilization",
        "threshold": 30,  # percent
        "action": "remove_1_instance",
        "cooldown": 600  # seconds
    }
}
```

### Database Optimization

#### MongoDB Sharding
```javascript
// Shard key: user_id for even distribution
sh.enableSharding("pathbreaker")
sh.shardCollection("pathbreaker.users", { "user_id": "hashed" })

// Indexes for performance
db.users.createIndex({ "email": 1 }, { unique: true })
db.users.createIndex({ "career_path.selected_path": 1 })
db.users.createIndex({ "created_at": -1 })
```

#### PostgreSQL Read Replicas
```sql
-- Master-Slave Replication
-- Master: Write operations
-- Slaves: Read operations (job search, course listings)

-- Connection pooling
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Optimize queries
CREATE INDEX idx_jobs_skills ON jobs USING GIN(required_skills);
CREATE INDEX idx_courses_category ON courses(skill_category, difficulty_level);
```

### Caching Strategy

```python
# cache/redis_cache.py
import redis
import json
from functools import wraps

class CacheService:
    def __init__(self):
        self.redis_client = redis.Redis(
            host='redis-cluster.pathbreaker.com',
            port=6379,
            db=0,
            decode_responses=True
        )
    
    def cache_result(self, ttl=3600):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                
                # Check cache
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Store in cache
                self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(result)
                )
                
                return result
            return wrapper
        return decorator

# Usage
cache = CacheService()

@cache.cache_result(ttl=7200)  # Cache for 2 hours
def get_popular_courses(category):
    # Expensive database query
    return db.courses.find({"category": category}).sort("rating", -1).limit(10)
```

### Async Job Processing

```python
# tasks/celery_tasks.py
from celery import Celery
import boto3

celery_app = Celery(
    'pathbreaker',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/0'
)

@celery_app.task(name='parse_resume')
def parse_resume_async(user_id, file_path):
    """
    Async resume parsing to avoid blocking API
    """
    textract = boto3.client('textract')
    
    # Extract text
    response = textract.detect_document_text(
        Document={'S3Object': {'Bucket': 'pathbreaker-resumes', 'Name': file_path}}
    )
    
    # Parse sections
    parsed_data = parse_resume_sections(response)
    
    # Update user profile
    update_user_profile(user_id, parsed_data)
    
    # Send notification
    send_notification(user_id, "Resume parsed successfully!")
    
    return parsed_data

@celery_app.task(name='scrape_jobs')
def scrape_jobs_async(platform):
    """
    Periodic job scraping from various platforms
    """
    jobs = scrape_platform(platform)
    bulk_insert_jobs(jobs)
    return len(jobs)

# Periodic tasks
celery_app.conf.beat_schedule = {
    'scrape-jobs-every-hour': {
        'task': 'scrape_jobs',
        'schedule': 3600.0,  # Every hour
        'args': ('angellist',)
    }
}
```

## 9. Monitoring & Analytics

### Application Monitoring

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

active_users = Gauge(
    'active_users',
    'Number of active users'
)

# Middleware for tracking
class MetricsMiddleware:
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response
```

### Error Logging

```python
# monitoring/logger.py
import logging
from logging.handlers import RotatingFileHandler
import sentry_sdk

# Sentry for error tracking
sentry_sdk.init(
    dsn="your-sentry-dsn",
    traces_sample_rate=1.0,
    environment="production"
)

# Custom logger
logger = logging.getLogger('pathbreaker')
logger.setLevel(logging.INFO)

# File handler
file_handler = RotatingFileHandler(
    'logs/pathbreaker.log',
    maxBytes=10485760,  # 10MB
    backupCount=10
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

logger.addHandler(file_handler)

# Usage
logger.info(f"User {user_id} completed career assessment")
logger.error(f"Failed to parse resume for user {user_id}", exc_info=True)
```

### User Analytics

```python
# analytics/user_tracking.py
from mixpanel import Mixpanel
import segment.analytics as analytics

class AnalyticsService:
    def __init__(self):
        self.mixpanel = Mixpanel('your-mixpanel-token')
        analytics.write_key = 'your-segment-key'
    
    def track_event(self, user_id, event_name, properties=None):
        """Track user events"""
        self.mixpanel.track(user_id, event_name, properties)
        analytics.track(user_id, event_name, properties)
    
    def track_page_view(self, user_id, page_name):
        """Track page views"""
        analytics.page(user_id, page_name)
    
    def identify_user(self, user_id, traits):
        """Update user profile"""
        self.mixpanel.people_set(user_id, traits)
        analytics.identify(user_id, traits)

# Usage examples
analytics = AnalyticsService()

# Track career path selection
analytics.track_event(
    user_id='user_123',
    event_name='Career Path Selected',
    properties={
        'path': 'Full Stack Developer',
        'confidence_score': 85
    }
)

# Track resume upload
analytics.track_event(
    user_id='user_123',
    event_name='Resume Uploaded',
    properties={
        'file_size': 245000,
        'ats_score': 78
    }
)
```

### A/B Testing Framework

```python
# analytics/ab_testing.py
import random

class ABTestingService:
    def __init__(self):
        self.experiments = {
            'resume_builder_v2': {
                'variants': ['control', 'variant_a', 'variant_b'],
                'weights': [0.33, 0.33, 0.34]
            },
            'career_quiz_length': {
                'variants': ['10_questions', '15_questions', '20_questions'],
                'weights': [0.33, 0.33, 0.34]
            }
        }
    
    def assign_variant(self, user_id, experiment_name):
        """Assign user to experiment variant"""
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return 'control'
        
        # Consistent assignment based on user_id
        random.seed(user_id)
        variant = random.choices(
            experiment['variants'],
            weights=experiment['weights']
        )[0]
        
        return variant
    
    def track_conversion(self, user_id, experiment_name, variant, converted):
        """Track conversion for A/B test"""
        analytics.track_event(
            user_id,
            'AB Test Conversion',
            {
                'experiment': experiment_name,
                'variant': variant,
                'converted': converted
            }
        )
```

## 10. API Endpoints

### Authentication Endpoints

```python
# routes/auth.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@router.post("/register")
async def register(request: RegisterRequest):
    """
    Register new user
    
    Returns:
        - user_id
        - access_token
        - message
    """
    # Check if user exists
    existing_user = db.users.find_one({"email": request.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    password_hash = auth_service.hash_password(request.password)
    
    # Create user
    user_id = str(uuid.uuid4())
    user = {
        "user_id": user_id,
        "email": request.email,
        "password_hash": password_hash,
        "profile": {
            "full_name": request.full_name,
            "phone": request.phone
        },
        "created_at": datetime.utcnow()
    }
    
    db.users.insert_one(user)
    
    # Generate token
    access_token = auth_service.create_access_token({"sub": user_id})
    
    return {
        "user_id": user_id,
        "access_token": access_token,
        "message": "Registration successful"
    }

@router.post("/login")
async def login(request: LoginRequest):
    """
    User login
    
    Returns:
        - access_token
        - user_profile
    """
    # Find user
    user = db.users.find_one({"email": request.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not auth_service.verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate token
    access_token = auth_service.create_access_token({"sub": user["user_id"]})
    
    # Update last login
    db.users.update_one(
        {"user_id": user["user_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    return {
        "access_token": access_token,
        "user_profile": {
            "user_id": user["user_id"],
            "email": user["email"],
            "full_name": user["profile"]["full_name"]
        }
    }

@router.post("/refresh")
async def refresh_token(user_id: str = Depends(auth_service.verify_token)):
    """
    Refresh access token
    """
    new_token = auth_service.create_access_token({"sub": user_id})
    return {"access_token": new_token}
```

### PathFinder Endpoints

```python
# routes/pathfinder.py
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/api/pathfinder", tags=["PathFinder"])

@router.get("/quiz")
async def get_career_quiz():
    """
    Get career assessment quiz questions
    
    Returns:
        - List of 15 questions with options
    """
    questions = [
        {
            "id": 1,
            "question": "What interests you the most?",
            "options": ["Technology", "Business", "Creative Arts", "Science"],
            "category": "interests"
        },
        # ... more questions
    ]
    return {"questions": questions}

@router.post("/analyze")
async def analyze_career_path(
    quiz_responses: dict,
    user_id: str = Depends(auth_service.verify_token)
):
    """
    Analyze quiz responses and recommend career paths
    
    Request Body:
        - quiz_responses: {question_id: answer}
    
    Returns:
        - Top 5 career recommendations
        - Confidence scores
        - Salary insights
    """
    # AI analysis using Bedrock
    recommendations = career_recommender.recommend_careers(quiz_responses)
    
    # Save to user profile
    db.users.update_one(
        {"user_id": user_id},
        {"$set": {"career_path.recommendations": recommendations}}
    )
    
    return {"recommendations": recommendations}

@router.get("/roadmap/{path_id}")
async def get_learning_roadmap(
    path_id: str,
    user_id: str = Depends(auth_service.verify_token)
):
    """
    Get personalized learning roadmap for selected career
    
    Returns:
        - Structured roadmap (beginner to advanced)
        - Curated courses
        - Estimated timeline
        - Milestones
    """
    # Get career path details
    career_path = db.career_paths.find_one({"path_id": path_id})
    
    # Get user's current level
    user = db.users.find_one({"user_id": user_id})
    current_level = user.get("career_path", {}).get("current_level", "beginner")
    
    # Generate personalized roadmap using AI
    roadmap = bedrock_service.generate_roadmap(career_path, current_level)
    
    # Get curated courses
    courses = rag_service.semantic_search_courses(
        f"courses for {career_path['path_name']}"
    )
    
    return {
        "roadmap": roadmap,
        "courses": courses,
        "estimated_duration": "6 months"
    }

@router.post("/progress")
async def update_progress(
    course_id: str,
    completed: bool,
    user_id: str = Depends(auth_service.verify_token)
):
    """
    Update learning progress
    """
    db.users.update_one(
        {"user_id": user_id},
        {
            "$push": {
                "learning_progress.completed_courses": {
                    "course_id": course_id,
                    "completed_date": datetime.utcnow()
                }
            }
        }
    )
    
    return {"message": "Progress updated"}
```

### Escape Matrix Endpoints

```python
# routes/escape_matrix.py
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/api/escape", tags=["Escape Matrix"])

@router.post("/analyze")
async def analyze_earning_potential(
    capital: int,
    time_hours_per_week: int,
    skills: list[str],
    user_id: str = Depends(auth_service.verify_token)
):
    """
    Analyze user's earning potential and recommend strategies
    
    Request Body:
        - capital: Available investment (₹)
        - time_hours_per_week: Available time
        - skills: List of existing skills
    
    Returns:
        - Recommended earning strategies
        - Expected ROI
        - Timeline to first earning
    """
    user_profile = {
        "capital": capital,
        "time": time_hours_per_week,
        "skills": skills
    }
    
    # AI-powered strategy recommendation
    strategies = escape_matrix_service.recommend_strategies(user_profile)
    
    return {"strategies": strategies}

@router.get("/strategies")
async def get_all_strategies():
    """
    Get all available earning strategies
    
    Returns:
        - List of strategies with details
    """
    strategies = [
        {
            "id": "freelancing",
            "name": "Freelancing",
            "min_capital": 0,
            "min_time": 10,
            "difficulty": "Medium",
            "potential_earning": "₹10,000 - ₹50,000/month"
        },
        # ... more strategies
    ]
    return {"strategies": strategies}

@router.get("/guide/{strategy_id}")
async def get_strategy_guide(strategy_id: str):
    """
    Get detailed step-by-step guide for a strategy
    
    Returns:
        - Week-by-week action plan
        - Required resources
        - Success tips
        - Common pitfalls
    """
    guide = escape_matrix_service.get_detailed_guide(strategy_id)
    return {"guide": guide}

@router.post("/track-earnings")
async def track_earnings(
    amount: int,
    source: str,
    user_id: str = Depends(auth_service.verify_token)
):
    """
    Track user's earnings
    """
    db.users.update_one(
        {"user_id": user_id},
        {
            "$inc": {"earnings_tracker.current_earnings": amount},
            "$push": {
                "earnings_tracker.earnings_history": {
                    "month": datetime.utcnow().strftime("%Y-%m"),
                    "amount": amount,
                    "source": source
                }
            }
        }
    )
    
    return {"message": "Earnings tracked"}
```

### Career Accelerator Endpoints

```python
# routes/career.py
from fastapi import APIRouter, UploadFile, File, Depends

router = APIRouter(prefix="/api/career", tags=["Career Accelerator"])

@router.post("/resume/upload")
async def upload_resume(
    file: UploadFile = File(...),
    user_id: str = Depends(auth_service.verify_token)
):
    """
    Upload and parse resume
    
    Returns:
        - Parsed resume data
        - ATS score
        - Improvement suggestions
    """
    # Validate file
    await file_validator.validate_resume(file)
    
    # Upload to S3
    file_path = f"resumes/{user_id}/{file.filename}"
    s3_service.upload_file(file, file_path)
    
    # Async parsing
    task = parse_resume_async.delay(user_id, file_path)
    
    return {
        "message": "Resume uploaded successfully",
        "task_id": task.id,
        "status": "processing"
    }

@router.get("/resume/score")
async def get_resume_score(user_id: str = Depends(auth_service.verify_token)):
    """
    Get ATS score for uploaded resume
    
    Returns:
        - Score (0-100)
        - Missing keywords
        - Improvement suggestions
    """
    user = db.users.find_one({"user_id": user_id})
    resume_data = user.get("resume_data", {})
    
    # Calculate ATS score
    score = ats_scorer.calculate_score(resume_data)
    
    return {
        "score": score["total_score"],
        "breakdown": score["breakdown"],
        "suggestions": score["suggestions"]
    }

@router.post("/jobs/search")
async def search_jobs(
    keywords: str,
    location: str = None,
    experience: str = None,
    salary_min: int = None,
    user_id: str = Depends(auth_service.verify_token)
):
    """
    Search for jobs across multiple platforms
    
    Returns:
        - Matching jobs
        - Fit scores
    """
    # Get user profile for matching
    user = db.users.find_one({"user_id": user_id})
    user_skills = user.get("learning_progress", {}).get("skills_acquired", [])
    
    # Search jobs
    jobs = job_matcher.search_jobs(
        keywords=keywords,
        location=location,
        experience=experience,
        salary_min=salary_min
    )
    
    # Calculate fit scores
    for job in jobs:
        job["fit_score"] = job_matcher.calculate_fit_score(user_skills, job)
    
    # Sort by fit score
    jobs.sort(key=lambda x: x["fit_score"], reverse=True)
    
    return {"jobs": jobs}

@router.get("/interview/questions")
async def get_interview_questions(
    company: str = None,
    role: str = None
):
    """
    Get interview questions for specific company/role
    
    Returns:
        - Technical questions
        - Behavioral questions
        - Company-specific questions
    """
    questions = interview_service.get_questions(company, role)
    return {"questions": questions}

@router.post("/interview/mock")
async def start_mock_interview(
    role: str,
    user_id: str = Depends(auth_service.verify_token)
):
    """
    Start AI-powered mock interview
    
    Returns:
        - Interview session ID
        - First question
    """
    session_id = str(uuid.uuid4())
    questions = interview_service.generate_questions(role)
    
    return {
        "session_id": session_id,
        "first_question": questions[0]
    }
```

### CodeMitra Endpoints

```python
# routes/codemitra.py
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/api/code", tags=["CodeMitra"])

@router.post("/analyze")
async def analyze_code(
    code: str,
    language: str,
    user_id: str = Depends(auth_service.verify_token)
):
    """
    Analyze code for errors and improvements
    
    Request Body:
        - code: Source code
        - language: Programming language
    
    Returns:
        - Errors found
        - Warnings
        - Suggestions
    """
    analysis = codemitra_service.analyze_code(code, language)
    return {"analysis": analysis}

@router.post("/debug")
async def debug_code(
    code: str,
    error_message: str,
    language: str,
    user_id: str = Depends(auth_service.verify_token)
):
    """
    Get AI-powered debugging help
    
    Returns:
        - Error explanation (simple language)
        - Why it happened
        - How to fix (step-by-step)
        - Best practices
    """
    explanation = bedrock_service.explain_code_error(code, error_message, language)
    
    return {
        "explanation": explanation["simple_explanation"],
        "cause": explanation["cause"],
        "solution": explanation["solution"],
        "best_practices": explanation["best_practices"]
    }

@router.post("/improve")
async def improve_code(
    code: str,
    language: str,
    user_id: str = Depends(auth_service.verify_token)
):
    """
    Get code improvement suggestions
    
    Returns:
        - Optimized code
        - Performance improvements
        - Best practices applied
    """
    improvements = codemitra_service.suggest_improvements(code, language)
    
    return {
        "improved_code": improvements["code"],
        "changes": improvements["changes"],
        "performance_gain": improvements["performance_gain"]
    }

@router.post("/dsa/solve")
async def solve_dsa_problem(
    problem_description: str,
    constraints: str,
    user_id: str = Depends(auth_service.verify_token)
):
    """
    Get multiple approaches to solve DSA problem
    
    Returns:
        - Approach 1 (Brute force)
        - Approach 2 (Optimized)
        - Approach 3 (Most optimal)
        - Time/Space complexity analysis
    """
    solutions = codemitra_service.solve_dsa(problem_description, constraints)
    
    return {"solutions": solutions}
```

---

## Document Summary

This design document provides a comprehensive technical blueprint for PathBreaker AI, covering:

1. **System Architecture**: Multi-layered architecture with clear separation of concerns
2. **Component Design**: Detailed frontend and backend component structure
3. **Data Flow**: Visual representation of user journeys and feature workflows
4. **Database Schema**: MongoDB and PostgreSQL schemas with indexing strategies
5. **AI Integration**: AWS Bedrock, custom ML models, and RAG implementation
6. **Security**: Authentication, encryption, file validation, and rate limiting
7. **Deployment**: Docker for development, AWS infrastructure for production
8. **Scalability**: Horizontal scaling, caching, and async processing
9. **Monitoring**: Metrics, logging, analytics, and A/B testing
10. **API Endpoints**: Complete REST API specification for all features

**Document Version**: 1.0  
**Last Updated**: February 11, 2026  
**Status**: Ready for Implementation
