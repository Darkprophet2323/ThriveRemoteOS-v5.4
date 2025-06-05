import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ProfessionalWeatherWidget = () => {
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
      const response = await axios.get(`${API}/weather/enhanced?location=${encodeURIComponent(loc)}`);
      if (response.data.success) {
        setWeatherData(response.data.weather);
      }
      setLoading(false);
    } catch (error) {
      console.error('Error fetching weather:', error);
      setError('Unable to fetch weather data');
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
      'clear': 'wb_sunny',
      'clouds': 'cloud',
      'rain': 'water_drop',
      'snow': 'ac_unit',
      'thunderstorm': 'thunderstorm',
      'drizzle': 'grain',
      'mist': 'blur_on',
      'fog': 'blur_on'
    };
    return icons[condition?.toLowerCase()] || 'wb_cloudy';
  };

  const getAirQualityColor = (aqi) => {
    if (aqi <= 50) return '#38a169';
    if (aqi <= 100) return '#d69e2e';
    if (aqi <= 150) return '#ed8936';
    return '#e53e3e';
  };

  if (loading) {
    return (
      <div className="app-content">
        <div className="flex items-center justify-center" style={{ minHeight: '200px' }}>
          <div className="loading">
            <div className="text-lg font-semibold text-secondary mb-2">Loading Weather</div>
            <div className="text-sm text-muted">Fetching meteorological data...</div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app-content">
        <div className="card">
          <div className="card-content text-center py-12">
            <div className="text-4xl mb-4">⚠️</div>
            <h3 className="text-lg font-semibold text-secondary mb-2">Weather Unavailable</h3>
            <p className="text-sm text-muted mb-4">{error}</p>
            <button 
              className="btn-primary"
              onClick={() => fetchWeather(location)}
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  const displayTemp = convertTemperature(weatherData?.temperature || 20, 'celsius', unit);
  const feelsLike = convertTemperature(weatherData?.feels_like || 20, 'celsius', unit);

  return (
    <div className="app-content">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-primary mb-1">Weather Station</h1>
          <p className="text-sm text-muted">Real-time meteorological data</p>
        </div>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="btn-subtle"
        >
          Settings
        </button>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="card mb-6">
          <div className="card-header">
            <h3 className="text-lg font-semibold">Weather Settings</h3>
          </div>
          <div className="card-content">
            <div className="grid grid-cols-2 gap-4">
              <div className="form-group">
                <label className="form-label">Location</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    className="form-input"
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                    placeholder="Enter city name"
                  />
                  <button
                    onClick={() => fetchWeather(location)}
                    className="btn-primary"
                  >
                    Update
                  </button>
                </div>
              </div>
              
              <div className="form-group">
                <label className="form-label">Temperature Unit</label>
                <select
                  className="form-select"
                  value={unit}
                  onChange={(e) => setUnit(e.target.value)}
                >
                  <option value="celsius">Celsius (°C)</option>
                  <option value="fahrenheit">Fahrenheit (°F)</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Weather Display */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        
        {/* Current Weather */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="card-content p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-xl font-semibold text-primary">{location}</h2>
                  <p className="text-sm text-muted">Current Conditions</p>
                </div>
                <span className="material-icons-outlined text-4xl text-primary">
                  {getWeatherIcon(weatherData?.condition)}
                </span>
              </div>
              
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <div className="text-4xl font-bold text-primary mb-2">
                    {displayTemp}°{unit === 'celsius' ? 'C' : 'F'}
                  </div>
                  <div className="text-lg text-secondary mb-1">
                    {weatherData?.condition || 'Unknown'}
                  </div>
                  <div className="text-sm text-muted">
                    Feels like {feelsLike}°{unit === 'celsius' ? 'C' : 'F'}
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="text-muted">Humidity</div>
                    <div className="font-semibold">{weatherData?.humidity || 0}%</div>
                  </div>
                  <div>
                    <div className="text-muted">Wind</div>
                    <div className="font-semibold">{weatherData?.wind_speed || 0} km/h</div>
                  </div>
                  <div>
                    <div className="text-muted">Pressure</div>
                    <div className="font-semibold">{weatherData?.pressure || 0} hPa</div>
                  </div>
                  <div>
                    <div className="text-muted">Visibility</div>
                    <div className="font-semibold">{weatherData?.visibility || 0} km</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Additional Info */}
        <div className="space-y-4">
          
          {/* Sun Times */}
          <div className="card">
            <div className="card-content p-4">
              <h3 className="font-semibold text-primary mb-3">Sun & Moon</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted">Sunrise</span>
                  <span className="font-medium">{weatherData?.sunrise || '6:30 AM'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">Sunset</span>
                  <span className="font-medium">{weatherData?.sunset || '7:30 PM'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">UV Index</span>
                  <span className="font-medium">{weatherData?.uv_index || 3}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Air Quality */}
          {weatherData?.air_quality && (
            <div className="card">
              <div className="card-content p-4">
                <h3 className="font-semibold text-primary mb-3">Air Quality</h3>
                <div className="mb-3">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-muted">AQI</span>
                    <span 
                      className="px-2 py-1 rounded text-xs font-medium text-white"
                      style={{ backgroundColor: getAirQualityColor(weatherData.air_quality.aqi) }}
                    >
                      {weatherData.air_quality.aqi}
                    </span>
                  </div>
                  <div className="text-sm font-medium">{weatherData.air_quality.category}</div>
                </div>
                
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-muted">PM2.5</span>
                    <span>{weatherData.air_quality.pollutants.pm25} μg/m³</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">PM10</span>
                    <span>{weatherData.air_quality.pollutants.pm10} μg/m³</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted">Ozone</span>
                    <span>{weatherData.air_quality.pollutants.ozone} μg/m³</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Weather Forecast */}
      {weatherData?.forecast && (
        <div className="card mb-6">
          <div className="card-header">
            <h3 className="text-lg font-semibold">3-Day Forecast</h3>
          </div>
          <div className="card-content p-4">
            <div className="grid grid-cols-3 gap-4">
              {weatherData.forecast.map((day, index) => (
                <div key={index} className="text-center p-4 rounded-lg bg-secondary">
                  <div className="font-semibold text-primary mb-2">{day.day}</div>
                  <span className="material-icons-outlined text-2xl text-primary mb-2">
                    {getWeatherIcon(day.condition)}
                  </span>
                  <div className="text-sm mb-1">
                    <span className="font-semibold">{convertTemperature(day.high, 'celsius', unit)}°</span>
                    <span className="text-muted mx-1">/</span>
                    <span className="text-muted">{convertTemperature(day.low, 'celsius', unit)}°</span>
                  </div>
                  <div className="text-xs text-muted">
                    {day.precipitation} chance of rain
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Quick Location Access */}
      <div className="card">
        <div className="card-header">
          <h3 className="text-lg font-semibold">Quick Locations</h3>
        </div>
        <div className="card-content p-4">
          <div className="flex flex-wrap gap-2">
            {['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Toronto'].map(city => (
              <button
                key={city}
                onClick={() => setLocation(city)}
                className={`btn-subtle text-sm ${location === city ? 'bg-primary text-white' : ''}`}
              >
                {city}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Data Source */}
      <div className="text-center mt-6">
        <p className="text-xs text-muted">
          Last updated: {new Date().toLocaleTimeString()} • 
          Data source: Enhanced Weather API • 
          Location: {location}
        </p>
      </div>
    </div>
  );
};

export default ProfessionalWeatherWidget;