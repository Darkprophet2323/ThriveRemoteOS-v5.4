import React, { useState, useEffect } from 'react';

const WeatherWidget = () => {
  const [weather, setWeather] = useState(null);
  const [location, setLocation] = useState('London');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Demo weather data for luxury experience (fallback when API not available)
  const demoWeatherData = {
    location: 'London, UK',
    temperature: 18,
    condition: 'Partly Cloudy',
    humidity: 65,
    windSpeed: 12,
    icon: '⛅',
    forecast: [
      { day: 'Today', temp: '18°', icon: '⛅' },
      { day: 'Tomorrow', temp: '20°', icon: '☀️' },
      { day: 'Wed', temp: '16°', icon: '🌧️' },
      { day: 'Thu', temp: '19°', icon: '⛅' },
      { day: 'Fri', temp: '22°', icon: '☀️' }
    ]
  };

  useEffect(() => {
    fetchWeather();
    
    // Update weather every 10 minutes
    const interval = setInterval(fetchWeather, 600000);
    return () => clearInterval(interval);
  }, [location]);

  const fetchWeather = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // For demo purposes, we'll use the demo data
      // In production, you'd integrate with a weather API like OpenWeatherMap
      setTimeout(() => {
        setWeather(demoWeatherData);
        setLoading(false);
      }, 1000);
      
    } catch (err) {
      setError('Failed to fetch weather data');
      setWeather(demoWeatherData); // Fallback to demo data
      setLoading(false);
    }
  };

  const getWeatherIcon = (condition) => {
    if (condition.includes('sunny') || condition.includes('clear')) return '☀️';
    if (condition.includes('cloud')) return '⛅';
    if (condition.includes('rain')) return '🌧️';
    if (condition.includes('snow')) return '❄️';
    if (condition.includes('storm')) return '⛈️';
    return '🌤️';
  };

  const getCurrentTime = () => {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getCurrentDate = () => {
    return new Date().toLocaleDateString('en-GB', { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  if (loading) {
    return (
      <div style={{
        background: 'linear-gradient(135deg, rgba(248, 246, 244, 0.95), rgba(253, 245, 230, 0.9))',
        padding: '20px',
        borderRadius: '12px',
        border: '1px solid rgba(212, 175, 55, 0.3)',
        textAlign: 'center',
        minHeight: '200px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <div>
          <div style={{ fontSize: '2rem', marginBottom: '10px' }}>🌤️</div>
          <div style={{ color: '#D4AF37', fontWeight: '500' }}>Loading weather...</div>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(248, 246, 244, 0.95), rgba(253, 245, 230, 0.9))',
      padding: '20px',
      borderRadius: '12px',
      border: '1px solid rgba(212, 175, 55, 0.3)',
      fontFamily: 'Inter, sans-serif',
      backdropFilter: 'blur(20px)',
      boxShadow: '0 10px 40px rgba(0, 0, 0, 0.1)'
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '15px',
        paddingBottom: '10px',
        borderBottom: '1px solid rgba(212, 175, 55, 0.2)'
      }}>
        <h3 style={{ 
          margin: 0, 
          color: '#D4AF37', 
          fontFamily: 'Playfair Display, serif',
          fontSize: '1.1rem'
        }}>
          🌤️ Weather Station
        </h3>
        <button
          onClick={fetchWeather}
          style={{
            background: 'rgba(212, 175, 55, 0.2)',
            border: '1px solid rgba(212, 175, 55, 0.4)',
            borderRadius: '6px',
            padding: '4px 8px',
            cursor: 'pointer',
            fontSize: '0.75rem',
            color: '#0D0D0D'
          }}
        >
          🔄 Refresh
        </button>
      </div>

      {/* Current Time & Date */}
      <div style={{
        textAlign: 'center',
        marginBottom: '15px',
        padding: '10px',
        background: 'rgba(212, 175, 55, 0.1)',
        borderRadius: '8px'
      }}>
        <div style={{ 
          fontSize: '1.2rem', 
          fontWeight: '600', 
          color: '#0D0D0D',
          fontFamily: 'Playfair Display, serif'
        }}>
          {getCurrentTime()}
        </div>
        <div style={{ 
          fontSize: '0.8rem', 
          color: '#666',
          marginTop: '2px'
        }}>
          {getCurrentDate()}
        </div>
      </div>

      {/* Current Weather */}
      <div style={{
        textAlign: 'center',
        marginBottom: '20px'
      }}>
        <div style={{ fontSize: '3rem', marginBottom: '5px' }}>
          {weather?.icon || getWeatherIcon(weather?.condition)}
        </div>
        <div style={{ 
          fontSize: '2rem', 
          fontWeight: '600', 
          color: '#0D0D0D',
          marginBottom: '5px'
        }}>
          {weather?.temperature}°C
        </div>
        <div style={{ 
          fontSize: '0.9rem', 
          color: '#666',
          marginBottom: '3px'
        }}>
          {weather?.condition}
        </div>
        <div style={{ 
          fontSize: '0.8rem', 
          color: '#D4AF37',
          fontWeight: '500'
        }}>
          📍 {weather?.location}
        </div>
      </div>

      {/* Weather Details */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '10px',
        marginBottom: '15px'
      }}>
        <div style={{
          background: 'rgba(255, 255, 255, 0.6)',
          padding: '8px',
          borderRadius: '6px',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '0.7rem', color: '#666' }}>Humidity</div>
          <div style={{ fontSize: '0.9rem', fontWeight: '500', color: '#0D0D0D' }}>
            💧 {weather?.humidity}%
          </div>
        </div>
        <div style={{
          background: 'rgba(255, 255, 255, 0.6)',
          padding: '8px',
          borderRadius: '6px',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '0.7rem', color: '#666' }}>Wind</div>
          <div style={{ fontSize: '0.9rem', fontWeight: '500', color: '#0D0D0D' }}>
            💨 {weather?.windSpeed} km/h
          </div>
        </div>
      </div>

      {/* 5-Day Forecast */}
      <div>
        <h4 style={{ 
          margin: '0 0 10px 0', 
          fontSize: '0.9rem', 
          color: '#D4AF37',
          textAlign: 'center'
        }}>
          5-Day Forecast
        </h4>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          gap: '5px'
        }}>
          {weather?.forecast?.map((day, index) => (
            <div key={index} style={{
              background: 'rgba(255, 255, 255, 0.5)',
              padding: '8px 4px',
              borderRadius: '6px',
              textAlign: 'center',
              flex: 1,
              border: index === 0 ? '1px solid rgba(212, 175, 55, 0.4)' : '1px solid transparent'
            }}>
              <div style={{ 
                fontSize: '0.7rem', 
                color: '#666',
                marginBottom: '3px'
              }}>
                {day.day}
              </div>
              <div style={{ 
                fontSize: '1.2rem',
                marginBottom: '3px'
              }}>
                {day.icon}
              </div>
              <div style={{ 
                fontSize: '0.8rem', 
                fontWeight: '500',
                color: '#0D0D0D'
              }}>
                {day.temp}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Location Selector */}
      <div style={{
        marginTop: '15px',
        paddingTop: '10px',
        borderTop: '1px solid rgba(212, 175, 55, 0.2)',
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>
        <span style={{ fontSize: '0.8rem', color: '#666' }}>📍 Location:</span>
        <select
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          style={{
            flex: 1,
            padding: '4px 8px',
            border: '1px solid rgba(212, 175, 55, 0.3)',
            borderRadius: '4px',
            fontSize: '0.8rem',
            background: 'rgba(255, 255, 255, 0.8)'
          }}
        >
          <option value="London">London, UK</option>
          <option value="New York">New York, USA</option>
          <option value="Tokyo">Tokyo, Japan</option>
          <option value="Paris">Paris, France</option>
          <option value="Sydney">Sydney, Australia</option>
          <option value="Dubai">Dubai, UAE</option>
        </select>
      </div>

      {error && (
        <div style={{
          marginTop: '10px',
          padding: '8px',
          background: 'rgba(220, 20, 60, 0.1)',
          border: '1px solid rgba(220, 20, 60, 0.3)',
          borderRadius: '6px',
          color: '#DC143C',
          fontSize: '0.8rem',
          textAlign: 'center'
        }}>
          ⚠️ {error}
        </div>
      )}
    </div>
  );
};

export default WeatherWidget;
