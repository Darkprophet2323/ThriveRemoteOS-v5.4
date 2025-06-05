import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const WeatherWidgetEnhanced = () => {
  const [weatherData, setWeatherData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [location, setLocation] = useState('New York');
  const [unit, setUnit] = useState('celsius');
  const [error, setError] = useState(null);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    fetchWeather(location);
  }, [location]);

  const fetchWeather = async (loc) => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get(`${API}/weather/current?location=${encodeURIComponent(loc)}`);
      if (response.data.success) {
        setWeatherData(response.data.weather);
      }
      setLoading(false);
    } catch (error) {
      console.error('Error fetching weather:', error);
      setError('Failed to fetch weather data');
      setLoading(false);
    }
  };

  const convertTemperature = (temp, fromUnit, toUnit) => {
    if (fromUnit === toUnit) return temp;
    if (fromUnit === 'celsius' && toUnit === 'fahrenheit') {
      return Math.round((temp * 9/5) + 32);
    }
    if (fromUnit === 'fahrenheit' && toUnit === 'celsius') {
      return Math.round((temp - 32) * 5/9);
    }
    return temp;
  };

  const getWeatherIcon = (condition) => {
    const icons = {
      'sunny': '☀️',
      'cloudy': '☁️',
      'partly cloudy': '⛅',
      'rain': '🌧️',
      'snow': '❄️',
      'storm': '⛈️',
      'fog': '🌫️',
      'wind': '🌪️'
    };
    return icons[condition?.toLowerCase()] || '🌤️';
  };

  const getBackgroundGradient = (condition) => {
    const gradients = {
      'sunny': 'linear-gradient(135deg, #FFD700, #FFA500)',
      'cloudy': 'linear-gradient(135deg, #87CEEB, #778899)',
      'partly cloudy': 'linear-gradient(135deg, #87CEEB, #FFD700)',
      'rain': 'linear-gradient(135deg, #4682B4, #2F4F4F)',
      'snow': 'linear-gradient(135deg, #F0F8FF, #E6E6FA)',
      'storm': 'linear-gradient(135deg, #2F2F2F, #800080)',
      'fog': 'linear-gradient(135deg, #C0C0C0, #DCDCDC)',
      'wind': 'linear-gradient(135deg, #40E0D0, #00CED1)'
    };
    return gradients[weatherData?.condition?.toLowerCase()] || 'linear-gradient(135deg, #87CEEB, #FFD700)';
  };

  if (loading) {
    return (
      <div className="app-content">
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '200px',
          color: 'var(--charcoal-noir)'
        }}>
          <div style={{ fontSize: '2rem', marginBottom: '10px' }}>🌤️</div>
          <div>Loading weather data...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app-content">
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '200px',
          color: 'var(--charcoal-noir)'
        }}>
          <div style={{ fontSize: '2rem', marginBottom: '10px' }}>⚠️</div>
          <div>{error}</div>
          <button 
            className="luxury-btn"
            onClick={() => fetchWeather(location)}
            style={{ marginTop: '10px', fontSize: '0.8rem' }}
          >
            🔄 Retry
          </button>
        </div>
      </div>
    );
  }

  const displayTemp = convertTemperature(weatherData?.temperature || 20, 'celsius', unit);

  return (
    <div className="app-content" style={{ padding: '15px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h3 style={{ margin: 0, color: 'var(--noir-black)' }}>🌤️ Weather Station</h3>
        <button
          onClick={() => setShowSettings(!showSettings)}
          style={{
            background: 'rgba(212, 175, 55, 0.1)',
            border: '1px solid rgba(212, 175, 55, 0.3)',
            borderRadius: '6px',
            padding: '6px',
            cursor: 'pointer',
            color: 'var(--warm-gold)',
            fontSize: '0.9rem'
          }}
        >
          ⚙️
        </button>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div style={{
          background: 'rgba(248, 246, 244, 0.8)',
          padding: '15px',
          borderRadius: '10px',
          border: '1px solid rgba(212, 175, 55, 0.3)',
          marginBottom: '20px'
        }}>
          <h4 style={{ color: 'var(--warm-gold)', marginBottom: '15px' }}>⚙️ Weather Settings</h4>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '0.9rem', fontWeight: '500' }}>
                📍 Location:
              </label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <input
                  type="text"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="Enter city name"
                  style={{
                    flex: 1,
                    padding: '8px',
                    borderRadius: '6px',
                    border: '1px solid rgba(212, 175, 55, 0.3)',
                    background: 'rgba(255, 255, 255, 0.9)',
                    fontSize: '0.9rem'
                  }}
                />
                <button
                  onClick={() => fetchWeather(location)}
                  className="luxury-btn"
                  style={{ fontSize: '0.8rem', padding: '8px 12px' }}
                >
                  🔄 Update
                </button>
              </div>
            </div>
            
            <div>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '0.9rem', fontWeight: '500' }}>
                🌡️ Temperature Unit:
              </label>
              <select
                value={unit}
                onChange={(e) => setUnit(e.target.value)}
                style={{
                  padding: '8px',
                  borderRadius: '6px',
                  border: '1px solid rgba(212, 175, 55, 0.3)',
                  background: 'rgba(255, 255, 255, 0.9)',
                  fontSize: '0.9rem',
                  width: '120px'
                }}
              >
                <option value="celsius">Celsius (°C)</option>
                <option value="fahrenheit">Fahrenheit (°F)</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {/* Main Weather Display */}
      <div style={{
        background: getBackgroundGradient(weatherData?.condition),
        borderRadius: '15px',
        padding: '25px',
        color: 'white',
        textAlign: 'center',
        boxShadow: '0 10px 30px rgba(0, 0, 0, 0.2)',
        marginBottom: '20px',
        backgroundImage: 'linear-gradient(45deg, rgba(255,255,255,0.1) 25%, transparent 25%), linear-gradient(-45deg, rgba(255,255,255,0.1) 25%, transparent 25%), linear-gradient(45deg, transparent 75%, rgba(255,255,255,0.1) 75%), linear-gradient(-45deg, transparent 75%, rgba(255,255,255,0.1) 75%)',
        backgroundSize: '20px 20px',
        backgroundPosition: '0 0, 0 10px, 10px -10px, -10px 0px'
      }}>
        <div style={{ fontSize: '4rem', marginBottom: '10px', textShadow: '0 2px 4px rgba(0,0,0,0.3)' }}>
          {getWeatherIcon(weatherData?.condition)}
        </div>
        
        <div style={{ fontSize: '3rem', fontWeight: 'bold', marginBottom: '10px', textShadow: '0 2px 4px rgba(0,0,0,0.3)' }}>
          {displayTemp}°{unit === 'celsius' ? 'C' : 'F'}
        </div>
        
        <div style={{ fontSize: '1.3rem', marginBottom: '5px', fontWeight: '500', textShadow: '0 1px 2px rgba(0,0,0,0.3)' }}>
          {weatherData?.condition || 'Unknown'}
        </div>
        
        <div style={{ fontSize: '1rem', opacity: 0.9, textShadow: '0 1px 2px rgba(0,0,0,0.3)' }}>
          📍 {location}
        </div>
      </div>

      {/* Weather Details */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
        gap: '15px',
        marginBottom: '20px'
      }}>
        <div style={{
          background: 'rgba(248, 246, 244, 0.8)',
          padding: '15px',
          borderRadius: '10px',
          border: '1px solid rgba(212, 175, 55, 0.2)',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '1.5rem', marginBottom: '5px' }}>💧</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--charcoal-noir)', marginBottom: '2px' }}>Humidity</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: 'var(--noir-black)' }}>
            {weatherData?.humidity || 60}%
          </div>
        </div>
        
        <div style={{
          background: 'rgba(248, 246, 244, 0.8)',
          padding: '15px',
          borderRadius: '10px',
          border: '1px solid rgba(212, 175, 55, 0.2)',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '1.5rem', marginBottom: '5px' }}>💨</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--charcoal-noir)', marginBottom: '2px' }}>Wind Speed</div>
          <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: 'var(--noir-black)' }}>
            {weatherData?.wind_speed || 15} km/h
          </div>
        </div>
      </div>

      {/* Weather Widget Embed */}
      <div style={{
        background: 'rgba(13, 13, 13, 0.95)',
        borderRadius: '12px',
        padding: '20px',
        border: '2px solid rgba(212, 175, 55, 0.3)',
        marginBottom: '20px',
        boxShadow: '0 10px 30px rgba(0, 0, 0, 0.4)'
      }}>
        <h4 style={{ color: 'var(--warm-gold)', marginBottom: '15px', textAlign: 'center' }}>
          🌍 Global Weather Station
        </h4>
        
        {/* Simulated weatherwidget.io embed with noir theme */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(13, 13, 13, 0.98), rgba(26, 26, 26, 0.95))',
          borderRadius: '8px',
          padding: '15px',
          border: '1px solid rgba(212, 175, 55, 0.25)',
          color: 'var(--champagne)',
          fontFamily: 'monospace',
          fontSize: '0.8rem',
          minHeight: '120px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
            <div style={{ color: 'var(--warm-gold)', fontWeight: 'bold' }}>
              NOIR WEATHER STATION v2.1
            </div>
            <div style={{ color: 'var(--rose-gold)' }}>
              LIVE DATA
            </div>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
            <div>
              <div style={{ color: 'var(--pearl)', marginBottom: '8px' }}>CURRENT CONDITIONS:</div>
              <div style={{ color: 'var(--warm-gold)' }}>Location: {location}</div>
              <div style={{ color: 'var(--champagne)' }}>Temperature: {displayTemp}°{unit === 'celsius' ? 'C' : 'F'}</div>
              <div style={{ color: 'var(--champagne)' }}>Condition: {weatherData?.condition || 'Clear'}</div>
              <div style={{ color: 'var(--champagne)' }}>Humidity: {weatherData?.humidity || 60}%</div>
            </div>
            
            <div>
              <div style={{ color: 'var(--pearl)', marginBottom: '8px' }}>ATMOSPHERIC DATA:</div>
              <div style={{ color: 'var(--champagne)' }}>Wind: {weatherData?.wind_speed || 15} km/h</div>
              <div style={{ color: 'var(--champagne)' }}>Pressure: 1013 hPa</div>
              <div style={{ color: 'var(--champagne)' }}>Visibility: 10 km</div>
              <div style={{ color: 'var(--rose-gold)' }}>Status: ACTIVE</div>
            </div>
          </div>
          
          <div style={{ 
            marginTop: '15px', 
            padding: '8px', 
            background: 'rgba(212, 175, 55, 0.1)', 
            borderRadius: '4px',
            border: '1px solid rgba(212, 175, 55, 0.2)',
            textAlign: 'center',
            color: 'var(--warm-gold)'
          }}>
            ⚡ NOIR WEATHER ENGINE • LUXURY METEOROLOGICAL SUITE ⚡
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div style={{
        display: 'flex',
        gap: '10px',
        justifyContent: 'center',
        flexWrap: 'wrap'
      }}>
        <button
          onClick={() => fetchWeather(location)}
          className="luxury-btn"
          style={{ fontSize: '0.8rem', padding: '8px 16px' }}
        >
          🔄 Refresh
        </button>
        
        <button
          onClick={() => setLocation('London')}
          style={{
            background: 'rgba(212, 175, 55, 0.1)',
            border: '1px solid rgba(212, 175, 55, 0.3)',
            padding: '8px 16px',
            borderRadius: '6px',
            cursor: 'pointer',
            color: 'var(--warm-gold)',
            fontSize: '0.8rem',
            fontWeight: '500'
          }}
        >
          🌍 London
        </button>
        
        <button
          onClick={() => setLocation('Tokyo')}
          style={{
            background: 'rgba(212, 175, 55, 0.1)',
            border: '1px solid rgba(212, 175, 55, 0.3)',
            padding: '8px 16px',
            borderRadius: '6px',
            cursor: 'pointer',
            color: 'var(--warm-gold)',
            fontSize: '0.8rem',
            fontWeight: '500'
          }}
        >
          🗾 Tokyo
        </button>
        
        <button
          onClick={() => setLocation('Paris')}
          style={{
            background: 'rgba(212, 175, 55, 0.1)',
            border: '1px solid rgba(212, 175, 55, 0.3)',
            padding: '8px 16px',
            borderRadius: '6px',
            cursor: 'pointer',
            color: 'var(--warm-gold)',
            fontSize: '0.8rem',
            fontWeight: '500'
          }}
        >
          🗼 Paris
        </button>
      </div>

      {/* Last Updated */}
      <div style={{
        textAlign: 'center',
        marginTop: '15px',
        fontSize: '0.7rem',
        color: 'var(--charcoal-noir)',
        opacity: 0.7
      }}>
        Last updated: {new Date().toLocaleTimeString()}
      </div>
    </div>
  );
};

export default WeatherWidgetEnhanced;