import React, { useState, useEffect } from 'react';

const EnhancedSettings = () => {
  const [settings, setSettings] = useState({
    theme: 'noir-gold-luxury',
    soundEffects: true,
    musicPlayer: true,
    autoRefreshJobs: true,
    notifications: true,
    animationSpeed: 'normal',
    windowOpacity: 0.98,
    backgroundParticles: true,
    newsTickerSpeed: 'normal',
    autoSaveDocuments: true,
    downloadLocation: 'Downloads',
    weatherLocation: 'New York',
    weatherUnit: 'celsius',
    languagePreference: 'en',
    timezone: 'UTC',
    privacyMode: false,
    darkMode: false,
    highContrast: false,
    fontSize: 'medium',
    showSystemInfo: true
  });

  const [activeTab, setActiveTab] = useState('appearance');

  useEffect(() => {
    // Load settings from localStorage
    const savedSettings = localStorage.getItem('thriveRemoteSettings');
    if (savedSettings) {
      setSettings({ ...settings, ...JSON.parse(savedSettings) });
    }
  }, []);

  const updateSetting = (key, value) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    localStorage.setItem('thriveRemoteSettings', JSON.stringify(newSettings));
    
    // Apply theme changes immediately
    if (key === 'theme') {
      applyTheme(value);
    }
    
    // Apply other visual changes
    if (key === 'darkMode') {
      document.body.classList.toggle('dark-mode', value);
    }
    
    if (key === 'highContrast') {
      document.body.classList.toggle('high-contrast', value);
    }
    
    if (key === 'fontSize') {
      document.body.classList.remove('font-small', 'font-medium', 'font-large');
      document.body.classList.add(`font-${value}`);
    }
  };

  const applyTheme = (theme) => {
    const root = document.documentElement;
    document.body.className = ''; // Clear existing classes
    
    switch (theme) {
      case 'noir-gold-luxury':
        root.style.setProperty('--primary-color', '#D4AF37');
        root.style.setProperty('--secondary-color', '#E8B4B8');
        root.style.setProperty('--background-color', '#F8F6F4');
        root.style.setProperty('--text-color', '#0D0D0D');
        root.style.setProperty('--warm-gold', '#D4AF37');
        root.style.setProperty('--rose-gold', '#E8B4B8');
        root.style.setProperty('--champagne', '#F7E7CE');
        root.style.setProperty('--noir-black', '#0D0D0D');
        document.body.classList.add('theme-noir-gold');
        break;
      case 'dark-elegance':
        root.style.setProperty('--primary-color', '#8B4513');
        root.style.setProperty('--secondary-color', '#A0522D');
        root.style.setProperty('--background-color', '#1A1A1A');
        root.style.setProperty('--text-color', '#F5F5DC');
        root.style.setProperty('--warm-gold', '#CD853F');
        root.style.setProperty('--rose-gold', '#BC8F8F');
        root.style.setProperty('--champagne', '#F5E6D3');
        root.style.setProperty('--noir-black', '#F5F5DC');
        document.body.classList.add('theme-dark-elegance');
        break;
      case 'champagne-dreams':
        root.style.setProperty('--primary-color', '#F7E7CE');
        root.style.setProperty('--secondary-color', '#E8B4B8');
        root.style.setProperty('--background-color', '#FDF5E6');
        root.style.setProperty('--text-color', '#2F2F2F');
        root.style.setProperty('--warm-gold', '#DAA520');
        root.style.setProperty('--rose-gold', '#E8B4B8');
        root.style.setProperty('--champagne', '#F7E7CE');
        root.style.setProperty('--noir-black', '#2F2F2F');
        document.body.classList.add('theme-champagne-dreams');
        break;
      case 'midnight-rose':
        root.style.setProperty('--primary-color', '#C41E3A');
        root.style.setProperty('--secondary-color', '#800020');
        root.style.setProperty('--background-color', '#0D0D0D');
        root.style.setProperty('--text-color', '#F8F6F4');
        root.style.setProperty('--warm-gold', '#FFD700');
        root.style.setProperty('--rose-gold', '#C41E3A');
        root.style.setProperty('--champagne', '#F8F6F4');
        root.style.setProperty('--noir-black', '#F8F6F4');
        document.body.classList.add('theme-midnight-rose');
        break;
      default:
        break;
    }
    
    // Apply additional visual effects based on settings
    if (settings.backgroundParticles) {
      document.body.classList.add('particles-enabled');
    } else {
      document.body.classList.remove('particles-enabled');
    }
    
    // Apply animation speed
    root.style.setProperty('--animation-speed', 
      settings.animationSpeed === 'slow' ? '1.5s' :
      settings.animationSpeed === 'fast' ? '0.3s' :
      settings.animationSpeed === 'instant' ? '0s' : '0.6s'
    );
    
    // Apply window opacity
    root.style.setProperty('--window-opacity', settings.windowOpacity);
    
    // Dispatch custom event for other components to react
    window.dispatchEvent(new CustomEvent('themeChanged', { 
      detail: { theme, settings } 
    }));
  };

  const resetSettings = () => {
    const defaultSettings = {
      theme: 'noir-gold-luxury',
      soundEffects: true,
      musicPlayer: true,
      autoRefreshJobs: true,
      notifications: true,
      animationSpeed: 'normal',
      windowOpacity: 0.98,
      backgroundParticles: true,
      newsTickerSpeed: 'normal',
      autoSaveDocuments: true,
      downloadLocation: 'Downloads',
      weatherLocation: 'New York',
      weatherUnit: 'celsius',
      languagePreference: 'en',
      timezone: 'UTC',
      privacyMode: false,
      darkMode: false,
      highContrast: false,
      fontSize: 'medium',
      showSystemInfo: true
    };
    setSettings(defaultSettings);
    localStorage.setItem('thriveRemoteSettings', JSON.stringify(defaultSettings));
    applyTheme('noir-gold-luxury');
  };

  const exportSettings = () => {
    const dataStr = JSON.stringify(settings, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = 'thrive-remote-settings.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const SettingRow = ({ label, description, children }) => (
    <div className="setting-item" style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '12px 0',
      borderBottom: '1px solid rgba(212, 175, 55, 0.1)'
    }}>
      <div style={{ flex: 1 }}>
        <label style={{ color: 'var(--noir-black)', fontWeight: '500', display: 'block' }}>
          {label}
        </label>
        {description && (
          <div style={{ fontSize: '0.8rem', color: 'var(--charcoal-noir)', opacity: 0.7, marginTop: '2px' }}>
            {description}
          </div>
        )}
      </div>
      <div style={{ marginLeft: '15px' }}>
        {children}
      </div>
    </div>
  );

  const Checkbox = ({ checked, onChange }) => (
    <label style={{ 
      position: 'relative', 
      display: 'inline-block', 
      width: '50px', 
      height: '24px',
      cursor: 'pointer'
    }}>
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        style={{ opacity: 0, width: 0, height: 0 }}
      />
      <span style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: checked ? 'var(--warm-gold)' : 'rgba(158, 158, 158, 0.3)',
        borderRadius: '24px',
        transition: 'all 0.3s ease',
        border: '1px solid ' + (checked ? 'var(--warm-gold)' : 'rgba(158, 158, 158, 0.5)')
      }}>
        <span style={{
          position: 'absolute',
          content: '',
          height: '18px',
          width: '18px',
          left: checked ? '28px' : '3px',
          bottom: '2px',
          backgroundColor: 'white',
          borderRadius: '50%',
          transition: 'all 0.3s ease',
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)'
        }}></span>
      </span>
    </label>
  );

  const Select = ({ value, onChange, options }) => (
    <select
      value={value}
      onChange={onChange}
      style={{
        padding: '6px 10px',
        borderRadius: '6px',
        border: '1px solid rgba(212, 175, 55, 0.3)',
        background: 'rgba(255, 255, 255, 0.8)',
        color: 'var(--noir-black)',
        fontSize: '0.9rem',
        minWidth: '120px'
      }}
    >
      {options.map(option => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );

  const Slider = ({ value, onChange, min, max, step }) => (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={onChange}
      style={{
        width: '100px',
        height: '4px',
        borderRadius: '2px',
        background: 'rgba(212, 175, 55, 0.3)',
        outline: 'none',
        opacity: 0.7,
        transition: 'opacity 0.2s'
      }}
    />
  );

  return (
    <div className="app-content">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h3>⚙️ System Settings</h3>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            className="luxury-btn"
            onClick={exportSettings}
            style={{ fontSize: '0.8rem', padding: '6px 12px' }}
          >
            📤 Export
          </button>
          <button 
            onClick={resetSettings}
            style={{
              background: 'rgba(244, 67, 54, 0.1)',
              color: '#F44336',
              border: '1px solid rgba(244, 67, 54, 0.3)',
              padding: '6px 12px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.8rem',
              fontWeight: '500'
            }}
          >
            🔄 Reset
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div style={{
        display: 'flex',
        gap: '2px',
        marginBottom: '20px',
        borderBottom: '2px solid rgba(212, 175, 55, 0.2)'
      }}>
        {[
          { id: 'appearance', label: '🎨 Appearance', icon: '🎨' },
          { id: 'system', label: '🔧 System', icon: '🔧' },
          { id: 'accessibility', label: '♿ Accessibility', icon: '♿' },
          { id: 'advanced', label: '⚡ Advanced', icon: '⚡' }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '10px 16px',
              border: 'none',
              background: activeTab === tab.id ? 'var(--gold-gradient)' : 'rgba(248, 246, 244, 0.5)',
              color: activeTab === tab.id ? 'var(--noir-black)' : 'var(--charcoal-noir)',
              borderRadius: '8px 8px 0 0',
              cursor: 'pointer',
              fontWeight: activeTab === tab.id ? '600' : '400',
              fontSize: '0.9rem',
              transition: 'all 0.3s ease'
            }}
          >
            {tab.icon} {tab.label.split(' ')[1]}
          </button>
        ))}
      </div>

      <div className="settings-sections">
        {/* Appearance Tab */}
        {activeTab === 'appearance' && (
          <div className="setting-group">
            <h4 style={{ color: 'var(--warm-gold)', marginBottom: '15px' }}>🎨 Appearance & Themes</h4>
            
            <SettingRow 
              label="Theme Selection" 
              description="Choose your preferred visual theme"
            >
              <Select
                value={settings.theme}
                onChange={(e) => updateSetting('theme', e.target.value)}
                options={[
                  { value: 'noir-gold-luxury', label: 'Noir-Gold Luxury' },
                  { value: 'dark-elegance', label: 'Dark Elegance' },
                  { value: 'champagne-dreams', label: 'Champagne Dreams' },
                  { value: 'midnight-rose', label: 'Midnight Rose' }
                ]}
              />
            </SettingRow>

            <SettingRow 
              label="Dark Mode" 
              description="Toggle dark mode overlay"
            >
              <Checkbox
                checked={settings.darkMode}
                onChange={(e) => updateSetting('darkMode', e.target.checked)}
              />
            </SettingRow>

            <SettingRow 
              label="Background Particles" 
              description="Animated background effects"
            >
              <Checkbox
                checked={settings.backgroundParticles}
                onChange={(e) => updateSetting('backgroundParticles', e.target.checked)}
              />
            </SettingRow>

            <SettingRow 
              label="Window Opacity" 
              description="Transparency level of windows"
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Slider
                  value={settings.windowOpacity}
                  onChange={(e) => updateSetting('windowOpacity', parseFloat(e.target.value))}
                  min="0.5"
                  max="1"
                  step="0.05"
                />
                <span style={{ fontSize: '0.8rem', minWidth: '40px' }}>
                  {Math.round(settings.windowOpacity * 100)}%
                </span>
              </div>
            </SettingRow>

            <SettingRow 
              label="Animation Speed" 
              description="Speed of UI animations"
            >
              <Select
                value={settings.animationSpeed}
                onChange={(e) => updateSetting('animationSpeed', e.target.value)}
                options={[
                  { value: 'slow', label: 'Slow' },
                  { value: 'normal', label: 'Normal' },
                  { value: 'fast', label: 'Fast' },
                  { value: 'instant', label: 'Instant' }
                ]}
              />
            </SettingRow>
          </div>
        )}

        {/* System Tab */}
        {activeTab === 'system' && (
          <div className="setting-group">
            <h4 style={{ color: 'var(--warm-gold)', marginBottom: '15px' }}>🔧 System & Performance</h4>
            
            <SettingRow 
              label="Sound Effects" 
              description="Enable click and hover sounds"
            >
              <Checkbox
                checked={settings.soundEffects}
                onChange={(e) => updateSetting('soundEffects', e.target.checked)}
              />
            </SettingRow>

            <SettingRow 
              label="Music Player" 
              description="Enable luxury music integration"
            >
              <Checkbox
                checked={settings.musicPlayer}
                onChange={(e) => updateSetting('musicPlayer', e.target.checked)}
              />
            </SettingRow>

            <SettingRow 
              label="Auto-refresh Jobs" 
              description="Automatically update job listings"
            >
              <Checkbox
                checked={settings.autoRefreshJobs}
                onChange={(e) => updateSetting('autoRefreshJobs', e.target.checked)}
              />
            </SettingRow>

            <SettingRow 
              label="System Notifications" 
              description="Show desktop notifications"
            >
              <Checkbox
                checked={settings.notifications}
                onChange={(e) => updateSetting('notifications', e.target.checked)}
              />
            </SettingRow>

            <SettingRow 
              label="Auto-save Documents" 
              description="Automatically save changes in Text Atelier"
            >
              <Checkbox
                checked={settings.autoSaveDocuments}
                onChange={(e) => updateSetting('autoSaveDocuments', e.target.checked)}
              />
            </SettingRow>

            <SettingRow 
              label="News Ticker Speed" 
              description="Speed of scrolling news ticker"
            >
              <Select
                value={settings.newsTickerSpeed}
                onChange={(e) => updateSetting('newsTickerSpeed', e.target.value)}
                options={[
                  { value: 'slow', label: 'Slow' },
                  { value: 'normal', label: 'Normal' },
                  { value: 'fast', label: 'Fast' }
                ]}
              />
            </SettingRow>

            <SettingRow 
              label="Download Location" 
              description="Default download folder"
            >
              <Select
                value={settings.downloadLocation}
                onChange={(e) => updateSetting('downloadLocation', e.target.value)}
                options={[
                  { value: 'Downloads', label: 'Downloads' },
                  { value: 'Desktop', label: 'Desktop' },
                  { value: 'Documents', label: 'Documents' },
                  { value: 'Custom', label: 'Custom Path' }
                ]}
              />
            </SettingRow>
          </div>
        )}

        {/* Accessibility Tab */}
        {activeTab === 'accessibility' && (
          <div className="setting-group">
            <h4 style={{ color: 'var(--warm-gold)', marginBottom: '15px' }}>♿ Accessibility Options</h4>
            
            <SettingRow 
              label="High Contrast Mode" 
              description="Enhanced contrast for better visibility"
            >
              <Checkbox
                checked={settings.highContrast}
                onChange={(e) => updateSetting('highContrast', e.target.checked)}
              />
            </SettingRow>

            <SettingRow 
              label="Font Size" 
              description="Text size throughout the interface"
            >
              <Select
                value={settings.fontSize}
                onChange={(e) => updateSetting('fontSize', e.target.value)}
                options={[
                  { value: 'small', label: 'Small' },
                  { value: 'medium', label: 'Medium' },
                  { value: 'large', label: 'Large' },
                  { value: 'xl', label: 'Extra Large' }
                ]}
              />
            </SettingRow>

            <SettingRow 
              label="Language Preference" 
              description="Interface language"
            >
              <Select
                value={settings.languagePreference}
                onChange={(e) => updateSetting('languagePreference', e.target.value)}
                options={[
                  { value: 'en', label: 'English' },
                  { value: 'es', label: 'Spanish' },
                  { value: 'fr', label: 'French' },
                  { value: 'de', label: 'German' },
                  { value: 'it', label: 'Italian' }
                ]}
              />
            </SettingRow>

            <SettingRow 
              label="Timezone" 
              description="Your local timezone"
            >
              <Select
                value={settings.timezone}
                onChange={(e) => updateSetting('timezone', e.target.value)}
                options={[
                  { value: 'UTC', label: 'UTC' },
                  { value: 'EST', label: 'Eastern' },
                  { value: 'PST', label: 'Pacific' },
                  { value: 'CST', label: 'Central' },
                  { value: 'MST', label: 'Mountain' }
                ]}
              />
            </SettingRow>
          </div>
        )}

        {/* Advanced Tab */}
        {activeTab === 'advanced' && (
          <div className="setting-group">
            <h4 style={{ color: 'var(--warm-gold)', marginBottom: '15px' }}>⚡ Advanced Settings</h4>
            
            <SettingRow 
              label="Privacy Mode" 
              description="Enhanced privacy and security features"
            >
              <Checkbox
                checked={settings.privacyMode}
                onChange={(e) => updateSetting('privacyMode', e.target.checked)}
              />
            </SettingRow>

            <SettingRow 
              label="Show System Info" 
              description="Display system information in taskbar"
            >
              <Checkbox
                checked={settings.showSystemInfo}
                onChange={(e) => updateSetting('showSystemInfo', e.target.checked)}
              />
            </SettingRow>

            <SettingRow 
              label="Weather Location" 
              description="Default location for weather widget"
            >
              <input
                type="text"
                value={settings.weatherLocation}
                onChange={(e) => updateSetting('weatherLocation', e.target.value)}
                style={{
                  padding: '6px 10px',
                  borderRadius: '6px',
                  border: '1px solid rgba(212, 175, 55, 0.3)',
                  background: 'rgba(255, 255, 255, 0.8)',
                  color: 'var(--noir-black)',
                  fontSize: '0.9rem',
                  width: '120px'
                }}
              />
            </SettingRow>

            <SettingRow 
              label="Temperature Unit" 
              description="Celsius or Fahrenheit"
            >
              <Select
                value={settings.weatherUnit}
                onChange={(e) => updateSetting('weatherUnit', e.target.value)}
                options={[
                  { value: 'celsius', label: '°C' },
                  { value: 'fahrenheit', label: '°F' }
                ]}
              />
            </SettingRow>
          </div>
        )}
      </div>

      {/* Settings Summary */}
      <div style={{
        marginTop: '30px',
        padding: '15px',
        background: 'rgba(212, 175, 55, 0.1)',
        borderRadius: '10px',
        border: '1px solid rgba(212, 175, 55, 0.3)'
      }}>
        <h4 style={{ color: 'var(--warm-gold)', marginBottom: '10px' }}>📊 Current Configuration</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px', fontSize: '0.8rem' }}>
          <div>Theme: <strong>{settings.theme.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</strong></div>
          <div>Sound Effects: <strong>{settings.soundEffects ? 'Enabled' : 'Disabled'}</strong></div>
          <div>Auto-refresh: <strong>{settings.autoRefreshJobs ? 'Enabled' : 'Disabled'}</strong></div>
          <div>Font Size: <strong>{settings.fontSize.charAt(0).toUpperCase() + settings.fontSize.slice(1)}</strong></div>
          <div>Weather: <strong>{settings.weatherLocation} ({settings.weatherUnit})</strong></div>
          <div>Language: <strong>{settings.languagePreference.toUpperCase()}</strong></div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedSettings;