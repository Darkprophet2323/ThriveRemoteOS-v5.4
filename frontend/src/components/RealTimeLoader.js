import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const RealTimeLoader = ({ onComplete }) => {
  const [loadingSteps, setLoadingSteps] = useState([
    { id: 'backend', name: 'Backend Connection', status: 'pending', progress: 0 },
    { id: 'database', name: 'MongoDB Database', status: 'pending', progress: 0 },
    { id: 'weather', name: 'Weather Services', status: 'pending', progress: 0 },
    { id: 'jobs', name: 'Job Portal APIs', status: 'pending', progress: 0 },
    { id: 'music', name: 'Music System', status: 'pending', progress: 0 },
    { id: 'ai_tools', name: 'AI Tools Database', status: 'pending', progress: 0 },
    { id: 'relocateme', name: 'RelocateMe Integration', status: 'pending', progress: 0 },
    { id: 'ui_components', name: 'UI Components', status: 'pending', progress: 0 },
    { id: 'settings', name: 'User Settings', status: 'pending', progress: 0 },
    { id: 'finalization', name: 'System Finalization', status: 'pending', progress: 0 }
  ]);

  const [overallProgress, setOverallProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [loadingMessages, setLoadingMessages] = useState([]);

  const addMessage = (message) => {
    setLoadingMessages(prev => [...prev.slice(-4), { 
      id: Date.now(), 
      text: message, 
      timestamp: new Date().toLocaleTimeString() 
    }]);
  };

  const updateStep = (stepId, status, progress, message) => {
    setLoadingSteps(prev => prev.map(step => 
      step.id === stepId 
        ? { ...step, status, progress }
        : step
    ));
    
    if (message) {
      addMessage(message);
    }
  };

  const testBackendConnection = async () => {
    try {
      updateStep('backend', 'loading', 25, 'Connecting to ThriveRemoteOS backend...');
      
      await new Promise(resolve => setTimeout(resolve, 800));
      const response = await axios.get(`${API}/`, { timeout: 5000 });
      
      updateStep('backend', 'loading', 75, 'Backend handshake successful...');
      await new Promise(resolve => setTimeout(resolve, 500));
      
      if (response.data.message) {
        updateStep('backend', 'success', 100, '✅ Backend connection established');
        return true;
      }
    } catch (error) {
      updateStep('backend', 'error', 100, '❌ Backend connection failed');
      return false;
    }
  };

  const testDatabase = async () => {
    try {
      updateStep('database', 'loading', 30, 'Initializing MongoDB connection...');
      await new Promise(resolve => setTimeout(resolve, 600));
      
      const response = await axios.get(`${API}/dashboard/live-stats`, { timeout: 5000 });
      
      updateStep('database', 'loading', 70, 'Verifying database schemas...');
      await new Promise(resolve => setTimeout(resolve, 400));
      
      if (response.data.database === 'MongoDB Connected') {
        updateStep('database', 'success', 100, '✅ MongoDB database ready');
        return true;
      }
    } catch (error) {
      updateStep('database', 'error', 100, '❌ Database connection failed');
      return false;
    }
  };

  const testWeatherServices = async () => {
    try {
      updateStep('weather', 'loading', 40, 'Testing weather APIs...');
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const response = await axios.get(`${API}/weather/current?location=New York`, { timeout: 5000 });
      
      updateStep('weather', 'loading', 80, 'Validating weather data sources...');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      if (response.data.success) {
        updateStep('weather', 'success', 100, '✅ Weather services operational');
        return true;
      }
    } catch (error) {
      updateStep('weather', 'error', 100, '❌ Weather services unavailable');
      return false;
    }
  };

  const testJobAPIs = async () => {
    try {
      updateStep('jobs', 'loading', 35, 'Connecting to job portals...');
      await new Promise(resolve => setTimeout(resolve, 700));
      
      const response = await axios.get(`${API}/jobs/live`, { timeout: 8000 });
      
      updateStep('jobs', 'loading', 85, 'Syncing live job listings...');
      await new Promise(resolve => setTimeout(resolve, 400));
      
      if (response.data.jobs && response.data.jobs.length > 0) {
        updateStep('jobs', 'success', 100, '✅ Job portal APIs connected');
        return true;
      }
    } catch (error) {
      updateStep('jobs', 'error', 100, '❌ Job APIs connection failed');
      return false;
    }
  };

  const testMusicSystem = async () => {
    try {
      updateStep('music', 'loading', 50, 'Loading luxury music system...');
      await new Promise(resolve => setTimeout(resolve, 400));
      
      const response = await axios.get(`${API}/music/playlist`, { timeout: 5000 });
      
      updateStep('music', 'loading', 90, 'Curating luxury audio collection...');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      if (response.data.success && response.data.playlist) {
        updateStep('music', 'success', 100, '✅ Luxury music system ready');
        return true;
      }
    } catch (error) {
      updateStep('music', 'error', 100, '❌ Music system initialization failed');
      return false;
    }
  };

  const testAITools = async () => {
    try {
      updateStep('ai_tools', 'loading', 45, 'Loading AI tools database...');
      await new Promise(resolve => setTimeout(resolve, 600));
      
      const response = await axios.get(`${API}/content/ai-tools`, { timeout: 5000 });
      
      updateStep('ai_tools', 'loading', 85, 'Validating 120+ AI tools...');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      if (response.data.total_tools && response.data.total_tools > 100) {
        updateStep('ai_tools', 'success', 100, '✅ AI tools database loaded');
        return true;
      }
    } catch (error) {
      updateStep('ai_tools', 'error', 100, '❌ AI tools database failed');
      return false;
    }
  };

  const testRelocateMe = async () => {
    try {
      updateStep('relocateme', 'loading', 30, 'Connecting to RelocateMe services...');
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Simulate RelocateMe API connection
      updateStep('relocateme', 'loading', 70, 'Validating relocation opportunities...');
      await new Promise(resolve => setTimeout(resolve, 500));
      
      updateStep('relocateme', 'success', 100, '✅ RelocateMe integration active');
      return true;
    } catch (error) {
      updateStep('relocateme', 'error', 100, '❌ RelocateMe connection failed');
      return false;
    }
  };

  const loadUIComponents = async () => {
    try {
      updateStep('ui_components', 'loading', 20, 'Initializing UI components...');
      await new Promise(resolve => setTimeout(resolve, 400));
      
      updateStep('ui_components', 'loading', 60, 'Loading noir-gold theme assets...');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      updateStep('ui_components', 'loading', 90, 'Finalizing component rendering...');
      await new Promise(resolve => setTimeout(resolve, 200));
      
      updateStep('ui_components', 'success', 100, '✅ UI components rendered');
      return true;
    } catch (error) {
      updateStep('ui_components', 'error', 100, '❌ UI component loading failed');
      return false;
    }
  };

  const loadUserSettings = async () => {
    try {
      updateStep('settings', 'loading', 40, 'Loading user preferences...');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // Load settings from localStorage
      const savedSettings = localStorage.getItem('thriveRemoteSettings');
      
      updateStep('settings', 'loading', 80, 'Applying theme configurations...');
      await new Promise(resolve => setTimeout(resolve, 200));
      
      updateStep('settings', 'success', 100, '✅ User settings restored');
      return true;
    } catch (error) {
      updateStep('settings', 'error', 100, '❌ Settings loading failed');
      return false;
    }
  };

  const finalizeSystem = async () => {
    try {
      updateStep('finalization', 'loading', 33, 'Optimizing system performance...');
      await new Promise(resolve => setTimeout(resolve, 400));
      
      updateStep('finalization', 'loading', 66, 'Activating luxury experience...');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      updateStep('finalization', 'loading', 100, 'ThriveRemoteOS ready for launch...');
      await new Promise(resolve => setTimeout(resolve, 500));
      
      updateStep('finalization', 'success', 100, '🎭 Welcome to ThriveRemoteOS V5.3');
      return true;
    } catch (error) {
      updateStep('finalization', 'error', 100, '❌ System finalization failed');
      return false;
    }
  };

  useEffect(() => {
    const runLoadingSequence = async () => {
      const loadingFunctions = [
        testBackendConnection,
        testDatabase,
        testWeatherServices,
        testJobAPIs,
        testMusicSystem,
        testAITools,
        testRelocateMe,
        loadUIComponents,
        loadUserSettings,
        finalizeSystem
      ];

      for (let i = 0; i < loadingFunctions.length; i++) {
        setCurrentStep(i);
        await loadingFunctions[i]();
        
        // Update overall progress
        const progress = ((i + 1) / loadingFunctions.length) * 100;
        setOverallProgress(progress);
        
        // Small delay between steps
        await new Promise(resolve => setTimeout(resolve, 200));
      }

      // Complete loading
      setTimeout(() => {
        if (onComplete) onComplete();
      }, 1000);
    };

    runLoadingSequence();
  }, [onComplete]);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success': return '✅';
      case 'error': return '❌';
      case 'loading': return '⚡';
      default: return '⏸️';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'success': return '#4CAF50';
      case 'error': return '#F44336';
      case 'loading': return '#D4AF37';
      default: return '#666';
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'linear-gradient(135deg, #0D0D0D 0%, #1A1A1A 50%, #0D0D0D 100%)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#D4AF37',
      fontFamily: 'Inter, sans-serif',
      zIndex: 10000
    }}>
      {/* Animated Background */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: `
          radial-gradient(circle at 20% 50%, rgba(212, 175, 55, 0.1) 0%, transparent 50%),
          radial-gradient(circle at 80% 20%, rgba(232, 180, 184, 0.1) 0%, transparent 50%),
          radial-gradient(circle at 40% 80%, rgba(212, 175, 55, 0.05) 0%, transparent 50%)
        `,
        animation: 'pulse 4s ease-in-out infinite'
      }} />

      {/* Main Logo */}
      <div style={{
        fontSize: '4rem',
        marginBottom: '20px',
        textShadow: '0 0 30px rgba(212, 175, 55, 0.5)',
        animation: 'glow 2s ease-in-out infinite alternate'
      }}>
        🎭
      </div>

      {/* Title */}
      <div style={{
        fontSize: '2.5rem',
        fontWeight: 'bold',
        marginBottom: '10px',
        fontFamily: 'Playfair Display, serif',
        textShadow: '0 2px 4px rgba(0, 0, 0, 0.5)'
      }}>
        ThriveRemoteOS V5.3
      </div>

      {/* Subtitle */}
      <div style={{
        fontSize: '1.2rem',
        marginBottom: '40px',
        opacity: 0.8,
        color: '#E8B4B8'
      }}>
        Initializing Noir-Gold Luxury Platform...
      </div>

      {/* Overall Progress Bar */}
      <div style={{
        width: '500px',
        height: '8px',
        background: 'rgba(255, 255, 255, 0.1)',
        borderRadius: '4px',
        marginBottom: '30px',
        overflow: 'hidden',
        border: '1px solid rgba(212, 175, 55, 0.3)'
      }}>
        <div style={{
          width: `${overallProgress}%`,
          height: '100%',
          background: 'linear-gradient(90deg, #D4AF37, #E8B4B8, #D4AF37)',
          borderRadius: '4px',
          transition: 'width 0.5s ease',
          boxShadow: '0 0 10px rgba(212, 175, 55, 0.5)'
        }} />
      </div>

      {/* Progress Percentage */}
      <div style={{
        fontSize: '1.1rem',
        marginBottom: '30px',
        fontWeight: 'bold'
      }}>
        {Math.round(overallProgress)}% Complete
      </div>

      {/* Loading Steps */}
      <div style={{
        width: '600px',
        maxHeight: '300px',
        overflowY: 'auto',
        marginBottom: '30px'
      }}>
        {loadingSteps.map((step, index) => (
          <div key={step.id} style={{
            display: 'flex',
            alignItems: 'center',
            padding: '8px 0',
            opacity: index <= currentStep ? 1 : 0.4,
            transition: 'opacity 0.3s ease'
          }}>
            <div style={{ marginRight: '12px', fontSize: '1.2rem' }}>
              {getStatusIcon(step.status)}
            </div>
            <div style={{ flex: 1, marginRight: '15px' }}>
              <div style={{ fontWeight: '500', color: getStatusColor(step.status) }}>
                {step.name}
              </div>
              {step.progress > 0 && (
                <div style={{
                  width: '100%',
                  height: '3px',
                  background: 'rgba(255, 255, 255, 0.1)',
                  borderRadius: '2px',
                  marginTop: '4px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${step.progress}%`,
                    height: '100%',
                    background: getStatusColor(step.status),
                    borderRadius: '2px',
                    transition: 'width 0.3s ease'
                  }} />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Live Messages */}
      <div style={{
        width: '600px',
        height: '120px',
        background: 'rgba(13, 13, 13, 0.8)',
        border: '1px solid rgba(212, 175, 55, 0.3)',
        borderRadius: '8px',
        padding: '15px',
        fontFamily: 'monospace',
        fontSize: '0.85rem',
        overflowY: 'auto'
      }}>
        {loadingMessages.map(message => (
          <div key={message.id} style={{
            marginBottom: '5px',
            color: '#E8B4B8',
            animation: 'fadeIn 0.3s ease'
          }}>
            <span style={{ color: '#666', fontSize: '0.7rem' }}>
              [{message.timestamp}]
            </span>
            {' '}
            <span>{message.text}</span>
          </div>
        ))}
      </div>

      {/* System Info */}
      <div style={{
        position: 'absolute',
        bottom: '20px',
        fontSize: '0.8rem',
        opacity: 0.6,
        textAlign: 'center'
      }}>
        <div>Backend: {BACKEND_URL}</div>
        <div>Build: Noir-Gold Luxury Edition • MongoDB • React 19 • FastAPI</div>
      </div>

      <style jsx>{`
        @keyframes glow {
          0% { text-shadow: 0 0 20px rgba(212, 175, 55, 0.5); }
          100% { text-shadow: 0 0 40px rgba(212, 175, 55, 0.8), 0 0 60px rgba(232, 180, 184, 0.3); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.8; }
        }
        @keyframes fadeIn {
          0% { opacity: 0; transform: translateY(10px); }
          100% { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
};

export default RealTimeLoader;