import React, { useState, useEffect } from 'react';

const LuxuryNewsTicker = () => {
  const [currentNews, setCurrentNews] = useState(0);

  // Sophisticated platform features news
  const newsItems = [
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
    },
    {
      icon: 'workspace_premium',
      text: '💎 Premium Experience • Sophisticated minimal design • Elegant noir aesthetics for discerning users'
    },
    {
      icon: 'language',
      text: '🌐 Global Platform • Multi-language support • International job opportunities and networking'
    },
    {
      icon: 'auto_awesome',
      text: '✨ Personalization Engine • AI learns your preferences • Customized experience tailored to your style'
    }
  ];

  // Rotate news items every 6 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentNews((prev) => (prev + 1) % newsItems.length);
    }, 6000);

    return () => clearInterval(interval);
  }, [newsItems.length]);

  return (
    <div className="news-ticker">
      <div className="ticker-label">
        <span className="material-icons-outlined" style={{ fontSize: '0.9rem', marginRight: '6px' }}>
          campaign
        </span>
        FEATURES
      </div>
      <div className="ticker-content">
        <div className="ticker-text">
          {newsItems.map((item, index) => (
            <div key={index} className="ticker-item">
              <span className="material-icons-outlined">{item.icon}</span>
              <span>{item.text}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default LuxuryNewsTicker;
