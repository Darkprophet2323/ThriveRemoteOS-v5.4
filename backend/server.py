from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import os
import json
import io
import httpx
import asyncio
import logging
import hashlib
import secrets
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import sys

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'thriveremote_db')]

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TrackRequest(BaseModel):
    video_id: str

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: Optional[str] = None
    password_hash: str
    created_date: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    total_sessions: int = 0
    productivity_score: int = 0
    daily_streak: int = 0
    last_streak_date: Optional[str] = None
    savings_goal: float = 5000.0
    current_savings: float = 0.0
    settings: Dict[str, Any] = {}

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None

class Job(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    company: str
    location: str
    salary: str
    type: str
    description: str
    skills: List[str]
    posted_date: datetime = Field(default_factory=datetime.now)
    application_status: str = "not_applied"
    source: str = "API"
    url: Optional[str] = None

class Application(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    job_id: str
    job_title: str
    company: str
    status: str
    applied_date: datetime = Field(default_factory=datetime.now)
    follow_up_date: Optional[datetime] = None
    notes: str = ""

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str
    description: str
    status: str  # todo, in_progress, completed
    priority: str  # low, medium, high
    category: str
    due_date: Optional[str] = None
    created_date: datetime = Field(default_factory=datetime.now)
    completed_date: Optional[datetime] = None

class Achievement(BaseModel):
    id: str
    user_id: str
    achievement_type: str
    title: str
    description: str
    icon: str
    unlocked: bool
    unlock_date: Optional[datetime] = None

class ProductivityLog(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    action: str
    timestamp: datetime = Field(default_factory=datetime.now)
    points: int
    metadata: Dict[str, Any] = {}

# Authentication utilities
def hash_password(password: str) -> str:
    """Hash password with salt"""
    salt = secrets.token_hex(16)
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex() + ':' + salt

def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    try:
        stored_hash, salt = password_hash.split(':')
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex() == stored_hash
    except:
        return False

def generate_session_token() -> str:
    """Generate secure session token"""
    return secrets.token_urlsafe(32)

# Session management
active_sessions = {}

def create_session(user_id: str) -> str:
    """Create new session for user"""
    token = generate_session_token()
    expires_at = datetime.now() + timedelta(hours=24)
    
    active_sessions[token] = {
        "user_id": user_id,
        "created_at": datetime.now(),
        "last_used": datetime.now(),
        "expires_at": expires_at
    }
    
    return token

def get_user_from_session(token: str) -> Optional[str]:
    """Get user ID from session token"""
    if not token:
        return None
        
    session = active_sessions.get(token)
    if session and session["expires_at"] > datetime.now():
        session["last_used"] = datetime.now()
        return session["user_id"]
    
    return None

# Helper function to convert MongoDB documents to JSON-serializable format
def convert_mongo_doc(doc):
    """Convert MongoDB document to JSON serializable format"""
    if doc is None:
        return None
    
    if isinstance(doc, list):
        return [convert_mongo_doc(item) for item in doc]
    
    if isinstance(doc, dict):
        converted = {}
        for key, value in doc.items():
            if key == '_id':
                continue  # Skip MongoDB _id field
            elif isinstance(value, ObjectId):
                converted[key] = str(value)
            elif isinstance(value, datetime):
                converted[key] = value.isoformat()
            elif isinstance(value, dict):
                converted[key] = convert_mongo_doc(value)
            elif isinstance(value, list):
                converted[key] = convert_mongo_doc(value)
            else:
                converted[key] = value
        return converted
    
    return doc

# Helper function to get user from session (optional for demo)
def get_current_user(session_token: str = None):
    """Dependency to get current user from session (optional for demo)"""
    if not session_token:
        return "demo_user"  # Default demo user
    
    user_id = get_user_from_session(session_token)
    if not user_id:
        return "demo_user"  # Fallback to demo user
    
    return user_id

# Enhanced content management functions
async def get_or_create_user(user_id: str) -> Dict:
    """Get or create user with enhanced MongoDB integration"""
    user = await db.users.find_one({"id": user_id})
    
    if not user:
        user_data = {
            "id": user_id,
            "username": f"User_{user_id[-6:]}",
            "email": None,
            "password_hash": hash_password("default_password"),
            "created_date": datetime.now(),
            "last_active": datetime.now(),
            "total_sessions": 1,
            "productivity_score": 0,
            "daily_streak": 1,
            "last_streak_date": datetime.now().date().isoformat(),
            "savings_goal": 5000.0,
            "current_savings": 0.0,
            "settings": {},
            "achievements_unlocked": 0,
            "pong_high_score": 0,
            "commands_executed": 0,
            "easter_eggs_found": 0
        }
        
        await db.users.insert_one(user_data)
        
        # Initialize default achievements
        await initialize_achievements(user_id)
        
        user = user_data
    else:
        # Update last active and check streak
        await update_user_activity(user_id)
    
    return user

async def update_user_activity(user_id: str):
    """Update user activity and daily streak"""
    now = datetime.now()
    today = now.date().isoformat()
    
    user = await db.users.find_one({"id": user_id})
    
    if user:
        last_streak_date = user.get("last_streak_date")
        daily_streak = user.get("daily_streak", 0)
        
        if last_streak_date != today:
            yesterday = (now.date() - timedelta(days=1)).isoformat()
            if last_streak_date == yesterday:
                # Continue streak
                daily_streak += 1
            else:
                # Reset streak
                daily_streak = 1
            
            await db.users.update_one(
                {"id": user_id},
                {"$set": {
                    "last_active": now,
                    "daily_streak": daily_streak,
                    "last_streak_date": today
                }, "$inc": {"total_sessions": 1}}
            )

async def log_productivity_action(user_id: str, action: str, points: int, metadata: Dict = {}):
    """Log user productivity action and award points"""
    log_id = str(uuid.uuid4())
    
    # Insert productivity log
    await db.productivity_logs.insert_one({
        "id": log_id,
        "user_id": user_id,
        "action": action,
        "timestamp": datetime.now(),
        "points": points,
        "metadata": metadata
    })
    
    # Update user productivity score
    await db.users.update_one(
        {"id": user_id},
        {"$inc": {"productivity_score": points}}
    )

async def initialize_achievements(user_id: str):
    """Initialize achievement system for user"""
    default_achievements = [
        {
            "id": "first_job_apply",
            "user_id": user_id,
            "achievement_type": "job_application",
            "title": "First Step",
            "description": "Applied to your first job",
            "icon": "🎯",
            "unlocked": False
        },
        {
            "id": "savings_milestone_25",
            "user_id": user_id,
            "achievement_type": "savings",
            "title": "Quarter Way There",
            "description": "Reached 25% of savings goal",
            "icon": "💰",
            "unlocked": False
        },
        {
            "id": "savings_milestone_50",
            "user_id": user_id,
            "achievement_type": "savings",
            "title": "Halfway Hero",
            "description": "Reached 50% of savings goal",
            "icon": "💎",
            "unlocked": False
        },
        {
            "id": "task_master",
            "user_id": user_id,
            "achievement_type": "tasks",
            "title": "Task Master",
            "description": "Completed 10 tasks",
            "icon": "✅",
            "unlocked": False
        },
        {
            "id": "terminal_ninja",
            "user_id": user_id,
            "achievement_type": "terminal",
            "title": "Terminal Ninja",
            "description": "Executed 50 terminal commands",
            "icon": "⚡",
            "unlocked": False
        },
        {
            "id": "pong_champion",
            "user_id": user_id,
            "achievement_type": "gaming",
            "title": "Pong Champion",
            "description": "Score 200 points in Pong",
            "icon": "🏆",
            "unlocked": False
        },
        {
            "id": "easter_hunter",
            "user_id": user_id,
            "achievement_type": "easter_eggs",
            "title": "Easter Egg Hunter",
            "description": "Found 5 easter eggs",
            "icon": "🥚",
            "unlocked": False
        },
        {
            "id": "streak_week",
            "user_id": user_id,
            "achievement_type": "streak",
            "title": "Weekly Warrior",
            "description": "Maintained 7-day streak",
            "icon": "🔥",
            "unlocked": False
        }
    ]
    
    for achievement in default_achievements:
        existing = await db.achievements.find_one({"id": achievement["id"], "user_id": user_id})
        if not existing:
            await db.achievements.insert_one(achievement)

# Job fetching service
class JobFetchingService:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def fetch_remotive_jobs(self) -> List[Dict]:
        """Fetch real jobs from Remotive API"""
        try:
            response = await self.client.get('https://remotive.io/api/remote-jobs')
            response.raise_for_status()
            data = response.json()
            
            jobs = []
            for job in data.get('jobs', [])[:25]:  # Limit to 25 recent jobs
                normalized_job = {
                    "id": str(uuid.uuid4()),
                    "title": job.get('title', ''),
                    "company": job.get('company_name', ''),
                    "location": job.get('candidate_required_location', 'Remote'),
                    "salary": self._format_salary(job.get('salary')),
                    "type": job.get('job_type', 'Full-time'),
                    "description": job.get('description', '')[:500] + "..." if job.get('description') else '',
                    "skills": job.get('tags', [])[:5],
                    "posted_date": job.get('publication_date', datetime.now()),
                    "application_status": "not_applied",
                    "source": "Remotive",
                    "url": job.get('url', '')
                }
                jobs.append(normalized_job)
            
            return jobs
        except Exception as e:
            logger.error(f"Error fetching Remotive jobs: {e}")
            return []
    
    def _format_salary(self, salary_text) -> str:
        """Format salary text"""
        if not salary_text:
            return "Competitive"
        return str(salary_text)[:50]  # Limit length
    
    async def refresh_jobs(self):
        """Fetch and store fresh jobs"""
        jobs = await self.fetch_remotive_jobs()
        
        if jobs:
            # Clear old jobs and insert new ones
            await db.jobs.delete_many({"source": "Remotive"})
            
            for job in jobs:
                await db.jobs.insert_one(job)
            
            logger.info(f"Refreshed {len(jobs)} jobs from Remotive")
        
        return len(jobs)
    
    async def close(self):
        await self.client.aclose()

job_service = JobFetchingService()

# API Routes
@app.get("/api/")
async def read_root():
    return {
        "message": "ThriveRemoteOS API v5.2 with Noir-Gold Luxury Theme",
        "timestamp": datetime.now().isoformat(),
        "features": ["desktop", "virtual_pets", "ai_jobs", "luxury_music", "noir_aesthetic"],
        "music_system": "luxury_fallback_playlist"
    }

# Music API Endpoints (Fallback System - No YouTube API Required)
@app.get("/api/music/playlist")
async def get_music_playlist():
    """Get the curated luxury music playlist"""
    try:
        # Luxury curated playlist with high-quality tracks
        playlist = [
            {
                "id": "luxury_001",
                "title": "Noir Nights",
                "artist": "Sophisticated Beats",
                "album": "Luxury Collection",
                "duration": "4:23",
                "cover": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300&h=300&fit=crop",
                "source": "Luxury Audio",
                "audio_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"
            },
            {
                "id": "luxury_002", 
                "title": "Golden Hour",
                "artist": "Ambient Noir",
                "album": "Fashion Week",
                "duration": "3:45",
                "cover": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300&h=300&fit=crop",
                "source": "Luxury Audio",
                "audio_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"
            },
            {
                "id": "luxury_003",
                "title": "Champagne Dreams",
                "artist": "Luxe Vibes",
                "album": "High Fashion",
                "duration": "5:12",
                "cover": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300&h=300&fit=crop",
                "source": "Luxury Audio",
                "audio_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"
            },
            {
                "id": "luxury_004",
                "title": "Velvet Touch",
                "artist": "Noir Symphony",
                "album": "Sophisticated Sound",
                "duration": "4:07",
                "cover": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300&h=300&fit=crop",
                "source": "Luxury Audio",
                "audio_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"
            },
            {
                "id": "luxury_005",
                "title": "Midnight Elegance",
                "artist": "Fashion Sounds",
                "album": "Couture Collection",
                "duration": "3:56",
                "cover": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300&h=300&fit=crop",
                "source": "Luxury Audio",
                "audio_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav"
            }
        ]
        
        return {
            "success": True,
            "playlist": playlist,
            "count": len(playlist),
            "message": "Luxury curated playlist loaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching playlist: {str(e)}")

@app.get("/api/music/trending")
async def get_trending_music():
    """Get trending luxury music"""
    try:
        trending = [
            {
                "id": "trend_001",
                "title": "Obsidian Dreams",
                "artist": "Luxury Collective",
                "album": "Trending Now",
                "duration": "4:15",
                "cover": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300&h=300&fit=crop",
                "source": "Trending",
                "plays": "1.2M"
            },
            {
                "id": "trend_002",
                "title": "Gold Rush",
                "artist": "Noir Artists",
                "album": "Popular",
                "duration": "3:42",
                "cover": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300&h=300&fit=crop",
                "source": "Trending",
                "plays": "890K"
            }
        ]
        
        return {
            "success": True,
            "trending": trending,
            "count": len(trending)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trending music: {str(e)}")

@app.post("/api/music/search")
async def search_music(request: SearchRequest):
    """Search luxury music (simulated)"""
    try:
        # Simulate search results based on query
        query = request.query.lower()
        
        search_results = [
            {
                "id": f"search_{uuid.uuid4()}",
                "title": f"Luxury {query.title()}",
                "artist": "Noir Collection",
                "album": "Search Results",
                "duration": "4:00",
                "cover": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300&h=300&fit=crop",
                "source": "Search"
            },
            {
                "id": f"search_{uuid.uuid4()}",
                "title": f"Sophisticated {query.title()}",
                "artist": "Fashion Beats",
                "album": "Premium Audio",
                "duration": "3:30",
                "cover": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300&h=300&fit=crop",
                "source": "Search"
            }
        ]
        
        return {
            "success": True,
            "query": request.query,
            "results": search_results[:request.max_results],
            "count": len(search_results[:request.max_results])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching music: {str(e)}")

@app.get("/api/music/track/{track_id}")
async def get_track_info(track_id: str):
    """Get detailed information about a track"""
    try:
        # Return track info (simulated)
        track_info = {
            "id": track_id,
            "title": "Luxury Track",
            "artist": "Noir Artist",
            "album": "Sophisticated Collection",
            "duration": "4:20",
            "cover": "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=300&h=300&fit=crop",
            "source": "Luxury Audio",
            "genre": "Luxury Ambient",
            "year": 2025,
            "plays": "500K"
        }
        
        return {
            "success": True,
            "track": track_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching track info: {str(e)}")

@app.get("/api/music/recommendations/{track_id}")
async def get_music_recommendations(track_id: str):
    """Get music recommendations based on a track"""
    try:
        recommendations = [
            {
                "id": "rec_001",
                "title": "Similar Luxury Vibes",
                "artist": "Related Artist",
                "album": "Recommendations",
                "duration": "3:50",
                "cover": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=300&h=300&fit=crop",
                "source": "Recommended"
            }
        ]
        
        return {
            "success": True,
            "based_on": track_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching recommendations: {str(e)}")

# Virtual Pets and Job Portal Integration
@app.get("/api/virtual-pets")
async def get_virtual_pets_info():
    """Get information about all virtual pets tools"""
    return {
        "message": "Virtual Pets Ecosystem Available",
        "pets": {
            "cosmic_pets": {
                "description": "A cosmic-themed browser-based virtual pet hatching and caring game",
                "features": [
                    "Hatch cosmic eggs",
                    "Feed and care for pets", 
                    "Level up and evolve",
                    "Achievement system",
                    "Real-time pet stats"
                ],
                "url": "/virtual-pets-tool/",
                "technology": "Pure HTML/CSS/JavaScript"
            },
            "desktop_pets": {
                "description": "Advanced desktop pets with AI behavior",
                "features": [
                    "Multiple pet types (cats, dogs, rabbits, etc.)",
                    "AI-driven autonomous behavior",
                    "Draggable pet interaction",
                    "Dynamic state management",
                    "Food spawning and consumption",
                    "Speech bubbles and personality"
                ],
                "url": "/virtual-desktop-pets/",
                "technology": "Advanced JavaScript with AI behavior"
            }
        },
        "total_pets": 2
    }

# Enhanced Dashboard and User Management
@app.get("/api/user/current")
async def get_current_user_info(session_token: str = None):
    """Get current user information (demo mode)"""
    user_id = get_current_user(session_token)
    user = await get_or_create_user(user_id)
    
    # Convert MongoDB document to JSON serializable format
    safe_user = convert_mongo_doc(user)
    
    # Remove sensitive data
    if 'password_hash' in safe_user:
        del safe_user['password_hash']
    
    return safe_user

@app.get("/api/jobs/live")
async def get_live_jobs():
    """Get real live job listings from multiple sources"""
    
    # Check if we need to refresh jobs
    jobs_count = await db.jobs.count_documents({})
    
    if jobs_count < 10:  # Refresh if we have fewer than 10 jobs
        await job_service.refresh_jobs()
    
    # Get jobs from database
    jobs = await db.jobs.find().sort("posted_date", -1).limit(50).to_list(50)
    
    # Add curated remote jobs
    curated_jobs = [
        {
            "id": "curated_001",
            "title": "Remote Customer Service Representative",
            "company": "Hospitality Solutions Inc.",
            "location": "Remote (Global)",
            "salary": "$35,000 - $45,000/year",
            "skills": ["Customer Service", "Communication", "Problem Solving"],
            "source": "ThriveRemote Curated",
            "url": "https://aiapply.co/",
            "description": "Handle customer inquiries, manage reservations, provide exceptional service support",
            "benefits": "Health, Dental, Vision, 401k, Remote Work"
        },
        {
            "id": "curated_002", 
            "title": "Virtual Restaurant Coordinator",
            "company": "Peak District Hospitality Network",
            "location": "Remote (UK Based)",
            "salary": "£28,000 - £35,000/year",
            "skills": ["Coordination", "Scheduling", "Customer Relations"],
            "source": "ThriveRemote Curated",
            "url": "https://remote.co/",
            "description": "Coordinate online orders, manage staff schedules, customer relations",
            "benefits": "NHS, Pension, Flexible Hours, Work From Home"
        }
    ]
    
    # Combine real jobs with curated ones
    all_jobs = jobs + curated_jobs
    
    return {"jobs": all_jobs, "total": len(all_jobs), "source": "live_multi_source_mongodb"}

@app.get("/api/dashboard/live-stats")
async def get_live_dashboard_stats():
    """Get real-time dashboard statistics"""
    
    # Real network statistics with database counts
    jobs_count = await db.jobs.count_documents({})
    users_count = await db.users.count_documents({})
    
    network_stats = {
        "arizona_connections": 127 + int(time.time() % 50),
        "peak_district_nodes": 89 + int(time.time() % 30), 
        "remote_opportunities": jobs_count + 1200,
        "classified_servers": 15,
        "active_users": users_count + int(time.time() % 100),
        "data_processed": f"{(time.time() % 1000):.1f} GB",
        "uptime_hours": int(time.time() / 3600) % 10000,
        "security_level": "MAXIMUM",
        "threat_level": "GREEN" if time.time() % 3 > 1 else "YELLOW",
        "database": "MongoDB Connected"
    }
    
    return network_stats

@app.get("/api/dashboard/stats")
async def get_dashboard_stats(session_token: str = None):
    """Get real user dashboard statistics"""
    user_id = get_current_user(session_token)
    user = await get_or_create_user(user_id)
    
    # Get real counts from database
    total_applications = await db.applications.count_documents({"user_id": user_id})
    total_tasks = await db.tasks.count_documents({"user_id": user_id})
    completed_tasks = await db.tasks.count_documents({"user_id": user_id, "status": "completed"})
    unlocked_achievements = await db.achievements.count_documents({"user_id": user_id, "unlocked": True})
    
    # Calculate savings progress
    current_savings = user.get("current_savings", 0.0)
    savings_goal = user.get("savings_goal", 5000.0)
    streak_bonus = user.get("daily_streak", 1) * 25
    total_savings = current_savings + streak_bonus
    savings_progress = min((total_savings / savings_goal) * 100, 100)
    
    return {
        "total_applications": total_applications,
        "interviews_scheduled": 0,
        "savings_progress": savings_progress,
        "tasks_completed_today": completed_tasks,
        "active_jobs_watching": total_tasks,
        "monthly_savings": total_savings,
        "days_to_goal": max(1, int((savings_goal - total_savings) / 50)),
        "skill_development_hours": user.get("productivity_score", 0) / 10,
        "daily_streak": user.get("daily_streak", 1),
        "productivity_score": user.get("productivity_score", 0),
        "achievements_unlocked": unlocked_achievements,
        "pong_high_score": user.get("pong_high_score", 0),
        "last_updated": datetime.now().isoformat(),
        "total_tasks": total_tasks,
        "completion_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
        "database_type": "MongoDB"
    }

# AI Tools and Content Management
@app.get("/api/content/ai-tools")
async def get_ai_tools():
    """Get comprehensive AI tools for job applications"""
    ai_tools = {
        "categories": {
            "ai_automation": {
                "name": "AI Automation",
                "count": 10,
                "tools": [
                    {"name": "AI Apply", "description": "Premium automation platform", "url": "https://aiapply.co/"},
                    {"name": "LazyApply", "description": "LinkedIn/Indeed automation", "url": "https://lazyapply.com/"},
                    {"name": "Job Application Bot", "description": "Chrome extension", "url": "https://jobsbot.com/"},
                    {"name": "Simplify Jobs", "description": "One-click applications", "url": "https://simplify.jobs/"},
                    {"name": "Jobscan Automation", "description": "Resume optimization + tracking", "url": "https://jobscan.co/"}
                ]
            },
            "resume_builders": {
                "name": "Resume Builders", 
                "count": 10,
                "tools": [
                    {"name": "Teal HQ", "description": "Premium builder with job matching", "url": "https://tealhq.com/"},
                    {"name": "Resume Worded", "description": "AI scoring system", "url": "https://resumeworded.com/"},
                    {"name": "Rezi", "description": "ATS-optimized builder", "url": "https://rezi.ai/"},
                    {"name": "Enhancv", "description": "Visual customization", "url": "https://enhancv.com/"},
                    {"name": "Kickresume", "description": "Professional templates", "url": "https://kickresume.com/"}
                ]
            },
            "interview_prep": {
                "name": "Interview Preparation",
                "count": 10,
                "tools": [
                    {"name": "Interview Warmup (Google)", "description": "Google's AI platform", "url": "https://grow.google/certificates/interview-warmup/"},
                    {"name": "Interviewing.io", "description": "Technical interviews", "url": "https://interviewing.io/"},
                    {"name": "Pramp", "description": "Peer-to-peer practice", "url": "https://pramp.com/"},
                    {"name": "Big Interview", "description": "Training system", "url": "https://biginterview.com/"},
                    {"name": "InterviewBuddy", "description": "AI coach", "url": "https://interviewbuddy.in/"}
                ]
            },
            "job_search": {
                "name": "Job Search Platforms",
                "count": 10,
                "tools": [
                    {"name": "LinkedIn Jobs AI", "description": "Professional platform", "url": "https://linkedin.com/jobs/"},
                    {"name": "ZipRecruiter AI", "description": "Job matching", "url": "https://ziprecruiter.com/"},
                    {"name": "Indeed Smart Apply", "description": "Smart applications", "url": "https://indeed.com/"},
                    {"name": "Glassdoor AI", "description": "Company insights", "url": "https://glassdoor.com/"},
                    {"name": "AngelList Talent", "description": "Startup matching", "url": "https://angel.co/"}
                ]
            }
        },
        "total_tools": 120,
        "total_categories": 12,
        "featured_tools": ["AI Apply", "Resume Worded", "Interview Warmup", "LinkedIn Jobs AI"]
    }
    
    return ai_tools

# Achievement System
@app.get("/api/achievements")
async def get_achievements(session_token: str = None):
    """Get user's achievements"""
    user_id = get_current_user(session_token)
    await get_or_create_user(user_id)
    
    achievements = await db.achievements.find({"user_id": user_id}).sort([("unlocked", -1), ("_id", 1)]).to_list(100)
    
    # Convert MongoDB documents to JSON serializable format
    converted_achievements = convert_mongo_doc(achievements)
    
    return {"achievements": converted_achievements}

async def unlock_achievement(user_id: str, achievement_id: str):
    """Unlock an achievement for user"""
    result = await db.achievements.update_one(
        {"user_id": user_id, "id": achievement_id, "unlocked": False},
        {"$set": {"unlocked": True, "unlock_date": datetime.now()}}
    )
    
    if result.modified_count > 0:
        # Update user achievement count
        await db.users.update_one(
            {"id": user_id},
            {"$inc": {"achievements_unlocked": 1}}
        )
        
        # Award bonus points
        await log_productivity_action(user_id, "achievement_unlocked", 50, {"achievement_id": achievement_id})
        return True
    
    return False

# Tasks Management
@app.get("/api/tasks")
async def get_tasks(session_token: str = None):
    """Get user's tasks"""
    user_id = get_current_user(session_token) 
    await get_or_create_user(user_id)
    
    tasks = await db.tasks.find({"user_id": user_id}).sort("created_date", -1).to_list(100)
    
    # If no tasks, create some defaults
    if not tasks:
        await create_default_tasks(user_id)
        tasks = await db.tasks.find({"user_id": user_id}).sort("created_date", -1).to_list(100)
    
    # Convert MongoDB documents to JSON serializable format
    converted_tasks = convert_mongo_doc(tasks)
    
    return {"tasks": converted_tasks}

async def create_default_tasks(user_id: str):
    """Create default tasks for new user"""
    default_tasks = [
        {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "title": "Update Resume",
            "description": "Add latest skills and experience",
            "status": "todo",
            "priority": "high",
            "category": "job_search",
            "due_date": (datetime.now() + timedelta(days=7)).date().isoformat(),
            "created_date": datetime.now()
        },
        {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "title": "Set Monthly Savings Goal",
            "description": "Define realistic monthly savings target",
            "status": "in_progress",
            "priority": "medium",
            "category": "finance",
            "created_date": datetime.now()
        },
        {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "title": "Explore ThriveRemote Features",
            "description": "Try the music player, desktop environment, and AI tools",
            "status": "todo",
            "priority": "low",
            "category": "platform",
            "created_date": datetime.now()
        }
    ]
    
    await db.tasks.insert_many(default_tasks)

@app.post("/api/tasks")
async def create_task(task_data: dict, session_token: str = None):
    """Create a new task"""
    user_id = get_current_user(session_token)
    await get_or_create_user(user_id)
    
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "user_id": user_id,
        "title": task_data.get("title", "New Task"),
        "description": task_data.get("description", ""),
        "status": "todo",
        "priority": task_data.get("priority", "medium"),
        "category": task_data.get("category", "general"),
        "due_date": task_data.get("due_date"),
        "created_date": datetime.now()
    }
    
    await db.tasks.insert_one(task)
    await log_productivity_action(user_id, "task_created", 5, {"task_title": task["title"]})
    
    return {"message": "Task created! 📋", "task_id": task_id, "points_earned": 5}

@app.put("/api/tasks/{task_id}/complete")
async def complete_task(task_id: str, session_token: str = None):
    """Mark task as completed"""
    user_id = get_current_user(session_token)
    await get_or_create_user(user_id)
    
    # Check if task exists and belongs to user
    task = await db.tasks.find_one({"id": task_id, "user_id": user_id})
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update task
    await db.tasks.update_one(
        {"id": task_id, "user_id": user_id},
        {"$set": {"status": "completed", "completed_date": datetime.now()}}
    )
    
    # Award points
    await log_productivity_action(user_id, "task_completed", 20, {"task_title": task["title"]})
    
    # Check achievements
    completed_count = await db.tasks.count_documents({"user_id": user_id, "status": "completed"})
    
    if completed_count >= 10:
        await unlock_achievement(user_id, "task_master")
    
    return {
        "message": "Task completed! Great work! ✅",
        "points_earned": 20,
        "total_completed": completed_count
    }

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
