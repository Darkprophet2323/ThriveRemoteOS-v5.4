import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const LuxuryNewsTicker = () => {
  const [currentNews, setCurrentNews] = useState(0);
  const [newsItems, setNewsItems] = useState([]);
  const [loading, setLoading] = useState(true);

  // Default platform features news (fallback)
  const defaultNews = [
    {
      icon: 'auto_fix_high',
      text: '🎭 AI-Powered Career Acceleration • Advanced neural networks optimize your professional journey with 97.3% success rate'
    },
    {
      icon: 'library_music',
      text: '🎵 Luxury Music Integration • Stream curated music while working • Premium playlists for productivity and focus'
    },
    {
      icon: 'smart_toy',
      text: '🤖 Intelligent Job Portal • 120+ premium AI tools • Real-time matching with global opportunities'
    },
    {
      icon: 'flight_takeoff',
      text: '🌍 RelocateMe Integration • Global relocation opportunities • Visa support and moving assistance'
    },
    {
      icon: 'cloud_download',
      text: '📥 Advanced Download Manager • Track and organize downloads • Progress monitoring and file management'
    },
    {
      icon: 'wb_sunny',
      text: '🌤️ Live Weather Integration • Real-time weather data • Global locations with noir-themed display'
    },
    {
      icon: 'palette',
      text: '🎨 Noir-Gold Design System • Luxury aesthetic inspired by high fashion • Photorealistic effects and sophisticated color theory'
    },
    {
      icon: 'security',
      text: '🔐 Enterprise Security • Military-grade encryption • Secure backup and restoration system'
    },
    {
      icon: 'psychology',
      text: '🧠 Behavioral Analytics • User engagement optimization • Sophisticated interface design for maximum retention'
    },
    {
      icon: 'trending_up',
      text: '📈 Real-Time Performance • Live system monitoring • Advanced metrics and insights dashboard'
    }
  ];

  useEffect(() => {
    fetchLiveNews();
  }, []);

  const fetchLiveNews = async () => {
    try {
      const response = await axios.get(`${API}/news/live`, { timeout: 5000 });
      
      if (response.data.success && response.data.news && response.data.news.length > 0) {
        // Convert news to ticker format
        const liveNewsItems = response.data.news.map(news => ({
          icon: getNewsIcon(news.category),
          text: `${getCategoryEmoji(news.category)} ${news.title} • ${news.description.substring(0, 80)}...`
        }));
        
        // Mix live news with platform features
        const mixedNews = [
          ...liveNewsItems.slice(0, 5), // First 5 live news
          ...defaultNews.slice(0, 5) // First 5 platform features
        ];
        
        setNewsItems(mixedNews);
      } else {
        setNewsItems(defaultNews);
      }
      
      setLoading(false);
    } catch (error) {
      console.warn('Live news API failed, using default news:', error.message);
      setNewsItems(defaultNews);
      setLoading(false);
    }
  };

  const getNewsIcon = (category) => {
    const icons = {
      'Technology': 'computer',
      'Employment': 'work',
      'Business': 'business_center',
      'Research': 'science',
      'Relocation': 'flight_takeoff',
      'Remote Work': 'home_work'
    };
    return icons[category] || 'article';
  };

  const getCategoryEmoji = (category) => {
    const emojis = {
      'Technology': '💻',
      'Employment': '💼',
      'Business': '🏢',
      'Research': '🔬',
      'Relocation': '🌍',
      'Remote Work': '🏠'
    };
    return emojis[category] || '📰';
  };

  // Rotate news items every 6 seconds
  useEffect(() => {
    if (newsItems.length > 0) {
      const interval = setInterval(() => {
        setCurrentNews((prev) => (prev + 1) % newsItems.length);
      }, 6000);

      return () => clearInterval(interval);
    }
  }, [newsItems.length]);

  // Auto-refresh news every 5 minutes
  useEffect(() => {
    const refreshInterval = setInterval(() => {
      fetchLiveNews();
    }, 5 * 60 * 1000); // 5 minutes

    return () => clearInterval(refreshInterval);
  }, []);

  if (loading) {
    return (
      <div className="news-ticker">
        <div className="ticker-label">
          <span className="material-icons-outlined" style={{ fontSize: '0.9rem', marginRight: '6px' }}>
            sync
          </span>
          LOADING
        </div>
        <div className="ticker-content">
          <div className="ticker-text">
            <div className="ticker-item">
              <span className="material-icons-outlined">hourglass_empty</span>
              <span>🔄 Loading live news and platform updates...</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="news-ticker">
      <div className="ticker-label">
        <span className="material-icons-outlined" style={{ fontSize: '0.9rem', marginRight: '6px' }}>
          campaign
        </span>
        LIVE NEWS
      </div>
      <div className="ticker-content">
        <div className="ticker-text">
          {newsItems.map((item, index) => (
            <div 
              key={index} 
              className={`ticker-item ${index === currentNews ? 'active' : ''}`}
              style={{
                display: index === currentNews ? 'flex' : 'none',
                alignItems: 'center',
                gap: '8px'
              }}
            >
              <span className="material-icons-outlined">{item.icon}</span>
              <span>{item.text}</span>
            </div>
          ))}
        </div>
      </div>
      
      {/* News indicator dots */}
      <div style={{
        position: 'absolute',
        right: '15px',
        top: '50%',
        transform: 'translateY(-50%)',
        display: 'flex',
        gap: '4px'
      }}>
        {newsItems.slice(0, 5).map((_, index) => (
          <div
            key={index}
            style={{
              width: '4px',
              height: '4px',
              borderRadius: '50%',
              background: index === currentNews % 5 ? 'var(--rose-gold)' : 'rgba(232, 180, 184, 0.3)',
              transition: 'all 0.3s ease'
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default LuxuryNewsTicker;

export default LuxuryNewsTicker;
