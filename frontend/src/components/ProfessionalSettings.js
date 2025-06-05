import React, { useState, useEffect } from 'react';

const ProfessionalSettings = () => {
  const [settings, setSettings] = useState({
    theme: 'executive-professional',
    soundEffects: true,
    musicPlayer: true,
    autoRefreshJobs: true,
    notifications: true,
    animationSpeed: 'normal',
    windowOpacity: 0.98,
    backgroundParticles: false,
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
    showSystemInfo: true,
    autoUpdates: true,
    dataUsage: 'standard',
    performanceMode: 'balanced'
  });

  const [activeTab, setActiveTab] = useState('appearance');
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = () => {
    const savedSettings = localStorage.getItem('thriveRemoteSettings');
    if (savedSettings) {
      const parsed = JSON.parse(savedSettings);
      setSettings({ ...settings, ...parsed });
      applySettings(parsed);
    }
  };

  const updateSetting = (key, value) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    setHasChanges(true);
    
    // Apply setting immediately
    applySetting(key, value);
  };

  const saveSettings = () => {
    localStorage.setItem('thriveRemoteSettings', JSON.stringify(settings));
    applySettings(settings);
    setHasChanges(false);
    
    // Show confirmation
    showNotification('Settings saved successfully', 'success');
  };

  const applySetting = (key, value) => {
    const root = document.documentElement;
    
    switch (key) {
      case 'theme':
        applyTheme(value);
        break;
      case 'darkMode':
        document.body.classList.toggle('dark-mode', value);
        break;
      case 'highContrast':
        document.body.classList.toggle('high-contrast', value);
        break;
      case 'fontSize':
        document.body.classList.remove('font-small', 'font-medium', 'font-large', 'font-xl');
        document.body.classList.add(`font-${value}`);
        break;
      case 'windowOpacity':
        root.style.setProperty('--window-opacity', value);
        break;
      case 'animationSpeed':
        const speed = value === 'slow' ? '0.8s' : value === 'fast' ? '0.2s' : value === 'instant' ? '0s' : '0.4s';
        root.style.setProperty('--transition-normal', speed);
        break;
      default:
        break;
    }
  };

  const applySettings = (settingsObj) => {
    Object.entries(settingsObj).forEach(([key, value]) => {
      applySetting(key, value);
    });
  };

  const applyTheme = (theme) => {
    const root = document.documentElement;
    document.body.className = document.body.className.replace(/theme-\w+/g, '');
    
    switch (theme) {
      case 'executive-professional':
        root.style.setProperty('--primary-color', '#1a202c');
        root.style.setProperty('--secondary-color', '#2d3748');
        root.style.setProperty('--accent-color', '#3182ce');
        root.style.setProperty('--background-color', '#f7fafc');
        root.style.setProperty('--text-color', '#1a202c');
        document.body.classList.add('theme-executive');
        break;
      case 'dark-professional':
        root.style.setProperty('--primary-color', '#2d3748');
        root.style.setProperty('--secondary-color', '#4a5568');
        root.style.setProperty('--accent-color', '#63b3ed');
        root.style.setProperty('--background-color', '#1a202c');
        root.style.setProperty('--text-color', '#f7fafc');
        document.body.classList.add('theme-dark-professional');
        break;
      case 'corporate-blue':
        root.style.setProperty('--primary-color', '#2b6cb0');
        root.style.setProperty('--secondary-color', '#3182ce');
        root.style.setProperty('--accent-color', '#63b3ed');
        root.style.setProperty('--background-color', '#ebf8ff');
        root.style.setProperty('--text-color', '#2d3748');
        document.body.classList.add('theme-corporate');
        break;
      default:
        break;
    }
    
    // Dispatch theme change event
    window.dispatchEvent(new CustomEvent('themeChanged', { 
      detail: { theme, settings } 
    }));
  };

  const resetSettings = () => {
    if (window.confirm('Are you sure you want to reset all settings to defaults?')) {
      const defaultSettings = {
        theme: 'executive-professional',
        soundEffects: true,
        musicPlayer: true,
        autoRefreshJobs: true,
        notifications: true,
        animationSpeed: 'normal',
        windowOpacity: 0.98,
        backgroundParticles: false,
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
        showSystemInfo: true,
        autoUpdates: true,
        dataUsage: 'standard',
        performanceMode: 'balanced'
      };
      
      setSettings(defaultSettings);
      localStorage.setItem('thriveRemoteSettings', JSON.stringify(defaultSettings));
      applySettings(defaultSettings);
      setHasChanges(false);
      showNotification('Settings reset to defaults', 'info');
    }
  };

  const exportSettings = () => {
    const dataStr = JSON.stringify(settings, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    const exportFileDefaultName = 'thrive-remote-settings.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    showNotification('Settings exported successfully', 'success');
  };

  const showNotification = (message, type) => {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: ${type === 'success' ? '#38a169' : type === 'error' ? '#e53e3e' : '#3182ce'};
      color: white;
      padding: 12px 20px;
      border-radius: 8px;
      font-weight: 500;
      z-index: 10000;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.remove();
    }, 3000);
  };

  const SettingRow = ({ label, description, children }) => (
    <div className="flex justify-between items-center py-3 border-b border-gray-200">
      <div className="flex-1">
        <label className="text-sm font-semibold text-primary">{label}</label>
        {description && (
          <div className="text-xs text-muted mt-1">{description}</div>
        )}
      </div>
      <div className="ml-4">
        {children}
      </div>
    </div>
  );

  const Toggle = ({ checked, onChange }) => (
    <label className="relative inline-flex items-center cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="sr-only"
      />
      <div className={`w-11 h-6 rounded-full transition-colors ${
        checked ? 'bg-blue-600' : 'bg-gray-300'
      }`}>
        <div className={`w-5 h-5 bg-white rounded-full shadow-md transform transition-transform ${
          checked ? 'translate-x-5' : 'translate-x-0'
        } mt-0.5 ml-0.5`}></div>
      </div>
    </label>
  );

  const Select = ({ value, onChange, options }) => (
    <select
      value={value}
      onChange={onChange}
      className="form-select text-sm"
      style={{ minWidth: '120px' }}
    >
      {options.map(option => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );

  const Slider = ({ value, onChange, min, max, step }) => (
    <div className="flex items-center gap-2">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={onChange}
        className="w-20 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
      />
      <span className="text-xs font-medium text-secondary min-w-8">
        {typeof value === 'number' ? Math.round(value * 100) + '%' : value}
      </span>
    </div>
  );

  return (
    <div className="app-content">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-primary mb-1">System Settings</h1>
          <p className="text-sm text-muted">Configure your ThriveRemoteOS experience</p>
        </div>
        <div className="flex gap-2">
          {hasChanges && (
            <button 
              className="btn-primary"
              onClick={saveSettings}
            >
              Save Changes
            </button>
          )}
          <button 
            className="btn-secondary"
            onClick={exportSettings}
          >
            Export
          </button>
          <button 
            className="btn-subtle"
            onClick={resetSettings}
          >
            Reset
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-1 mb-6 border-b border-gray-200">
        {[
          { id: 'appearance', label: 'Appearance', icon: 'palette' },
          { id: 'system', label: 'System', icon: 'settings' },
          { id: 'accessibility', label: 'Accessibility', icon: 'accessibility' },
          { id: 'privacy', label: 'Privacy', icon: 'security' },
          { id: 'advanced', label: 'Advanced', icon: 'tune' }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'text-blue-600 border-blue-600 bg-blue-50'
                : 'text-gray-500 border-transparent hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <span className="material-icons-outlined text-sm mr-2">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Settings Content */}
      <div className="card">
        <div className="card-content p-6">
          
          {/* Appearance Tab */}
          {activeTab === 'appearance' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Appearance Settings</h3>
              
              <SettingRow 
                label="Theme" 
                description="Choose your preferred interface theme"
              >
                <Select
                  value={settings.theme}
                  onChange={(e) => updateSetting('theme', e.target.value)}
                  options={[
                    { value: 'executive-professional', label: 'Executive Professional' },
                    { value: 'dark-professional', label: 'Dark Professional' },
                    { value: 'corporate-blue', label: 'Corporate Blue' }
                  ]}
                />
              </SettingRow>

              <SettingRow 
                label="Dark Mode Overlay" 
                description="Apply dark mode styling"
              >
                <Toggle
                  checked={settings.darkMode}
                  onChange={(e) => updateSetting('darkMode', e.target.checked)}
                />
              </SettingRow>

              <SettingRow 
                label="Window Opacity" 
                description="Transparency level of application windows"
              >
                <Slider
                  value={settings.windowOpacity}
                  onChange={(e) => updateSetting('windowOpacity', parseFloat(e.target.value))}
                  min="0.7"
                  max="1"
                  step="0.05"
                />
              </SettingRow>

              <SettingRow 
                label="Animation Speed" 
                description="Speed of interface transitions"
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

              <SettingRow 
                label="Background Effects" 
                description="Enable subtle background animations"
              >
                <Toggle
                  checked={settings.backgroundParticles}
                  onChange={(e) => updateSetting('backgroundParticles', e.target.checked)}
                />
              </SettingRow>
            </div>
          )}

          {/* System Tab */}
          {activeTab === 'system' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">System Settings</h3>
              
              <SettingRow 
                label="Sound Effects" 
                description="Enable interface sound feedback"
              >
                <Toggle
                  checked={settings.soundEffects}
                  onChange={(e) => updateSetting('soundEffects', e.target.checked)}
                />
              </SettingRow>

              <SettingRow 
                label="Background Music" 
                description="Enable ambient productivity music"
              >
                <Toggle
                  checked={settings.musicPlayer}
                  onChange={(e) => updateSetting('musicPlayer', e.target.checked)}
                />
              </SettingRow>

              <SettingRow 
                label="Auto-refresh Jobs" 
                description="Automatically update job listings"
              >
                <Toggle
                  checked={settings.autoRefreshJobs}
                  onChange={(e) => updateSetting('autoRefreshJobs', e.target.checked)}
                />
              </SettingRow>

              <SettingRow 
                label="System Notifications" 
                description="Show desktop notifications"
              >
                <Toggle
                  checked={settings.notifications}
                  onChange={(e) => updateSetting('notifications', e.target.checked)}
                />
              </SettingRow>

              <SettingRow 
                label="Auto-save Documents" 
                description="Automatically save changes"
              >
                <Toggle
                  checked={settings.autoSaveDocuments}
                  onChange={(e) => updateSetting('autoSaveDocuments', e.target.checked)}
                />
              </SettingRow>

              <SettingRow 
                label="Performance Mode" 
                description="Optimize system performance"
              >
                <Select
                  value={settings.performanceMode}
                  onChange={(e) => updateSetting('performanceMode', e.target.value)}
                  options={[
                    { value: 'power-saver', label: 'Power Saver' },
                    { value: 'balanced', label: 'Balanced' },
                    { value: 'performance', label: 'High Performance' }
                  ]}
                />
              </SettingRow>
            </div>
          )}

          {/* Accessibility Tab */}
          {activeTab === 'accessibility' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Accessibility Options</h3>
              
              <SettingRow 
                label="High Contrast Mode" 
                description="Enhanced contrast for better visibility"
              >
                <Toggle
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
                label="Language" 
                description="Interface language preference"
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
                description="Your local timezone setting"
              >
                <Select
                  value={settings.timezone}
                  onChange={(e) => updateSetting('timezone', e.target.value)}
                  options={[
                    { value: 'UTC', label: 'UTC' },
                    { value: 'EST', label: 'Eastern' },
                    { value: 'PST', label: 'Pacific' },
                    { value: 'CST', label: 'Central' },
                    { value: 'MST', label: 'Mountain' },
                    { value: 'GMT', label: 'Greenwich' }
                  ]}
                />
              </SettingRow>
            </div>
          )}

          {/* Privacy Tab */}
          {activeTab === 'privacy' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Privacy & Security</h3>
              
              <SettingRow 
                label="Privacy Mode" 
                description="Enhanced privacy protection"
              >
                <Toggle
                  checked={settings.privacyMode}
                  onChange={(e) => updateSetting('privacyMode', e.target.checked)}
                />
              </SettingRow>

              <SettingRow 
                label="Data Usage" 
                description="Control data consumption"
              >
                <Select
                  value={settings.dataUsage}
                  onChange={(e) => updateSetting('dataUsage', e.target.value)}
                  options={[
                    { value: 'minimal', label: 'Minimal' },
                    { value: 'standard', label: 'Standard' },
                    { value: 'full', label: 'Full Features' }
                  ]}
                />
              </SettingRow>

              <SettingRow 
                label="Automatic Updates" 
                description="Install updates automatically"
              >
                <Toggle
                  checked={settings.autoUpdates}
                  onChange={(e) => updateSetting('autoUpdates', e.target.checked)}
                />
              </SettingRow>

              <SettingRow 
                label="Show System Info" 
                description="Display system information in taskbar"
              >
                <Toggle
                  checked={settings.showSystemInfo}
                  onChange={(e) => updateSetting('showSystemInfo', e.target.checked)}
                />
              </SettingRow>
            </div>
          )}

          {/* Advanced Tab */}
          {activeTab === 'advanced' && (
            <div>
              <h3 className="text-lg font-semibold mb-4">Advanced Settings</h3>
              
              <SettingRow 
                label="Weather Location" 
                description="Default location for weather widget"
              >
                <input
                  type="text"
                  className="form-input text-sm"
                  value={settings.weatherLocation}
                  onChange={(e) => updateSetting('weatherLocation', e.target.value)}
                  style={{ width: '120px' }}
                />
              </SettingRow>

              <SettingRow 
                label="Temperature Unit" 
                description="Weather temperature display"
              >
                <Select
                  value={settings.weatherUnit}
                  onChange={(e) => updateSetting('weatherUnit', e.target.value)}
                  options={[
                    { value: 'celsius', label: 'Celsius (°C)' },
                    { value: 'fahrenheit', label: 'Fahrenheit (°F)' }
                  ]}
                />
              </SettingRow>

              <SettingRow 
                label="Download Location" 
                description="Default download directory"
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
            </div>
          )}
        </div>
      </div>

      {/* Settings Summary */}
      {hasChanges && (
        <div className="card mt-6">
          <div className="card-content p-4 bg-yellow-50 border-l-4 border-yellow-400">
            <div className="flex items-center">
              <span className="material-icons-outlined text-yellow-600 mr-2">info</span>
              <div>
                <h4 className="font-semibold text-yellow-800">Unsaved Changes</h4>
                <p className="text-sm text-yellow-700">You have unsaved changes. Click "Save Changes" to apply them.</p>
              </div>
              <button 
                className="btn-primary ml-auto"
                onClick={saveSettings}
              >
                Save Now
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProfessionalSettings;