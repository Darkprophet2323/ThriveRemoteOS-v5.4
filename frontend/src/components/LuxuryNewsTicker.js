import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ProfessionalNewsTicker = () => {
  const [currentNews, setCurrentNews] = useState(0);
  const [newsItems, setNewsItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [systemMetrics, setSystemMetrics] = useState({
    cpu: 0,
    memory: 0,
    uptime: '0h'
  });

  // Professional default news (fallback)
  const defaultNews = [
    {
      icon: 'trending_up',
      text: 'Professional Remote Work Platform • Real-time job matching with global opportunities'
    },
    {
      icon: 'business_center',
      text: 'Enterprise-Grade Performance • 120+ AI tools integrated for maximum productivity'
    },
    {
      icon: 'flight_takeoff',
      text: 'Global Relocation Services • RelocateMe integration with visa support and moving assistance'
    },
    {
      icon: 'cloud_download',
      text: 'Advanced Download Management • Professional file organization and progress tracking'
    },
    {
      icon: 'wb_sunny',
      text: 'Live Weather Intelligence • Real-time meteorological data with air quality monitoring'
    },
    {
      icon: 'security',
      text: 'Enterprise Security • Military-grade encryption with comprehensive data protection'
    }
  ];

  useEffect(() => {
    fetchLiveNews();
    fetchSystemMetrics();
    
    // Refresh data periodically
    const newsInterval = setInterval(fetchLiveNews, 5 * 60 * 1000); // 5 minutes
    const metricsInterval = setInterval(fetchSystemMetrics, 10000); // 10 seconds
    
    return () => {
      clearInterval(newsInterval);
      clearInterval(metricsInterval);
    };
  }, []);

  const fetchSystemMetrics = async () => {
    try {
      const response = await axios.get(`${API}/system/performance`, { timeout: 3000 });
      if (response.data.success) {
        setSystemMetrics({
          cpu: Math.round(response.data.performance.cpu_usage),
          memory: Math.round(response.data.performance.memory_usage),
          uptime: `${Math.round(response.data.performance.system_uptime_hours)}h`
        });
      }
    } catch (error) {
      // Silent fail for metrics
    }
  };

  const fetchLiveNews = async () => {
    try {
      const response = await axios.get(`${API}/news/live`, { timeout: 5000 });
      
      if (response.data.success && response.data.news && response.data.news.length > 0) {
        const liveNewsItems = response.data.news.slice(0, 4).map(news => ({
          icon: getNewsIcon(news.category),
          text: `${getCategoryEmoji(news.category)} ${news.title} • ${news.source} • ${new Date(news.published_at).toLocaleDateString()}`
        }));
        
        // Mix live news with platform features
        const mixedNews = [
          ...liveNewsItems,
          ...defaultNews.slice(0, 3)
        ];
        
        setNewsItems(mixedNews);
      } else {
        setNewsItems(defaultNews);
      }
      
      setLoading(false);
    } catch (error) {
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

  // Rotate news items
  useEffect(() => {
    if (newsItems.length > 0) {
      const interval = setInterval(() => {
        setCurrentNews((prev) => (prev + 1) % newsItems.length);
      }, 8000); // 8 seconds per item

      return () => clearInterval(interval);
    }
  }, [newsItems.length]);

  return (
    <div className="news-ticker">
      <div className="ticker-label">
        <span className="material-icons-outlined" style={{ fontSize: '0.875rem', marginRight: '8px' }}>
          campaign
        </span>
        {loading ? 'LOADING' : 'LIVE NEWS'}
      </div>
      
      <div className="ticker-content">
        <div className="ticker-text">
          {newsItems.length > 0 && (
            <div 
              className="ticker-item active"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
            >
              <span className="material-icons-outlined">
                {newsItems[currentNews]?.icon}
              </span>
              <span>{newsItems[currentNews]?.text}</span>
            </div>
          )}
        </div>
      </div>

      {/* System Metrics */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '16px',
        marginLeft: 'auto',
        paddingRight: '16px',
        fontSize: '0.75rem',
        color: '#718096',
        fontFamily: 'SF Mono, Monaco, Consolas, monospace'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span>CPU</span>
          <span style={{ fontWeight: '600', color: '#3182ce' }}>{systemMetrics.cpu}%</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span>MEM</span>
          <span style={{ fontWeight: '600', color: '#38a169' }}>{systemMetrics.memory}%</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span>UP</span>
          <span style={{ fontWeight: '600', color: '#d69e2e' }}>{systemMetrics.uptime}</span>
        </div>
      </div>

      {/* News indicator dots */}
      <div style={{
        position: 'absolute',
        right: '120px',
        top: '50%',
        transform: 'translateY(-50%)',
        display: 'flex',
        gap: '3px'
      }}>
        {newsItems.slice(0, 5).map((_, index) => (
          <div
            key={index}
            style={{
              width: '3px',
              height: '3px',
              borderRadius: '50%',
              background: index === currentNews % 5 ? '#3182ce' : 'rgba(113, 128, 150, 0.3)',
              transition: 'all 0.3s ease'
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default ProfessionalNewsTicker;
