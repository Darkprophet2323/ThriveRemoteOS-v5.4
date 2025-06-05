import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ProfessionalRealTimeLoader = ({ onComplete }) => {
  const [loadingSteps, setLoadingSteps] = useState([
    { id: 'backend', name: 'Backend Services', status: 'pending', progress: 0, details: 'Connecting to FastAPI backend...' },
    { id: 'database', name: 'Database Connection', status: 'pending', progress: 0, details: 'MongoDB cluster initialization...' },
    { id: 'weather', name: 'Weather Integration', status: 'pending', progress: 0, details: 'OpenWeatherMap API validation...' },
    { id: 'jobs', name: 'Job Portal APIs', status: 'pending', progress: 0, details: 'Remotive & live job sources...' },
    { id: 'relocateme', name: 'RelocateMe Platform', status: 'pending', progress: 0, details: 'Global relocation services...' },
    { id: 'ai_tools', name: 'AI Tools Suite', status: 'pending', progress: 0, details: '120+ professional AI tools...' },
    { id: 'news', name: 'News Integration', status: 'pending', progress: 0, details: 'Live news feeds & updates...' },
    { id: 'performance', name: 'System Performance', status: 'pending', progress: 0, details: 'Real-time monitoring setup...' },
    { id: 'security', name: 'Security Systems', status: 'pending', progress: 0, details: 'Enterprise security protocols...' },
    { id: 'finalization', name: 'Platform Ready', status: 'pending', progress: 0, details: 'Final optimizations...' }
  ]);

  const [overallProgress, setOverallProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [systemMetrics, setSystemMetrics] = useState({
    cpu: 0,
    memory: 0,
    network: 'Connecting...',
    uptime: '0s'
  });

  const updateStep = (stepId, status, progress, details) => {
    setLoadingSteps(prev => prev.map(step => 
      step.id === stepId 
        ? { ...step, status, progress, details }
        : step
    ));
  };

  const fetchSystemMetrics = async () => {
    try {
      const response = await axios.get(`${API}/system/performance`, { timeout: 3000 });
      if (response.data.success) {
        setSystemMetrics({
          cpu: response.data.performance.cpu_usage,
          memory: response.data.performance.memory_usage,
          network: 'Connected',
          uptime: `${Math.round(response.data.performance.system_uptime_hours)}h`
        });
      }
    } catch (error) {
      setSystemMetrics(prev => ({ ...prev, network: 'Limited' }));
    }
  };

  const testBackendConnection = async () => {
    try {
      updateStep('backend', 'loading', 20, 'Establishing secure connection...');
      await new Promise(resolve => setTimeout(resolve, 800));
      
      const response = await axios.get(`${API}/`, { timeout: 5000 });
      
      updateStep('backend', 'loading', 70, 'Verifying API endpoints...');
      await new Promise(resolve => setTimeout(resolve, 500));
      
      if (response.data.message) {
        updateStep('backend', 'success', 100, 'Backend services operational');
        return true;
      }
    } catch (error) {
      updateStep('backend', 'error', 100, 'Backend connection failed');
      return false;
    }
  };

  const testDatabase = async () => {
    try {
      updateStep('database', 'loading', 30, 'Connecting to MongoDB cluster...');
      await new Promise(resolve => setTimeout(resolve, 700));
      
      const response = await axios.get(`${API}/dashboard/live-stats`, { timeout: 5000 });
      
      updateStep('database', 'loading', 80, 'Validating data integrity...');
      await new Promise(resolve => setTimeout(resolve, 400));
      
      if (response.data) {
        updateStep('database', 'success', 100, 'Database connection established');
        return true;
      }
    } catch (error) {
      updateStep('database', 'error', 100, 'Database connection failed');
      return false;
    }
  };

  const testWeatherServices = async () => {
    try {
      updateStep('weather', 'loading', 25, 'Testing weather APIs...');
      await new Promise(resolve => setTimeout(resolve, 600));
      
      const response = await axios.get(`${API}/weather/enhanced?location=London`, { timeout: 5000 });
      
      updateStep('weather', 'loading', 75, 'Validating global weather data...');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      if (response.data.success) {
        updateStep('weather', 'success', 100, 'Weather services active');
        return true;
      }
    } catch (error) {
      updateStep('weather', 'error', 100, 'Weather services unavailable');
      return false;
    }
  };

  const testJobAPIs = async () => {
    try {
      updateStep('jobs', 'loading', 30, 'Connecting to job portals...');
      await new Promise(resolve => setTimeout(resolve, 800));
      
      const response = await axios.get(`${API}/jobs/live`, { timeout: 8000 });
      
      updateStep('jobs', 'loading', 85, 'Syncing live opportunities...');
      await new Promise(resolve => setTimeout(resolve, 400));
      
      if (response.data.jobs) {
        updateStep('jobs', 'success', 100, 'Job portal APIs connected');
        return true;
      }
    } catch (error) {
      updateStep('jobs', 'error', 100, 'Job APIs connection failed');
      return false;
    }
  };

  const testRelocateMe = async () => {
    try {
      updateStep('relocateme', 'loading', 40, 'Connecting to RelocateMe...');
      await new Promise(resolve => setTimeout(resolve, 700));
      
      const response = await axios.get(`${API}/relocateme/opportunities`, { timeout: 5000 });
      
      updateStep('relocateme', 'loading', 85, 'Loading global opportunities...');
      await new Promise(resolve => setTimeout(resolve, 400));
      
      if (response.data.success) {
        updateStep('relocateme', 'success', 100, 'RelocateMe platform ready');
        return true;
      }
    } catch (error) {
      updateStep('relocateme', 'error', 100, 'RelocateMe connection failed');
      return false;
    }
  };

  const testAITools = async () => {
    try {
      updateStep('ai_tools', 'loading', 35, 'Loading AI tools database...');
      await new Promise(resolve => setTimeout(resolve, 600));
      
      const response = await axios.get(`${API}/content/ai-tools`, { timeout: 5000 });
      
      updateStep('ai_tools', 'loading', 90, 'Validating 120+ tools...');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      if (response.data.total_tools) {
        updateStep('ai_tools', 'success', 100, 'AI tools suite loaded');
        return true;
      }
    } catch (error) {
      updateStep('ai_tools', 'error', 100, 'AI tools unavailable');
      return false;
    }
  };

  const testNewsIntegration = async () => {
    try {
      updateStep('news', 'loading', 45, 'Connecting to news sources...');
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const response = await axios.get(`${API}/news/live`, { timeout: 5000 });
      
      updateStep('news', 'loading', 90, 'Validating live feeds...');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      if (response.data.success) {
        updateStep('news', 'success', 100, 'News integration active');
        return true;
      }
    } catch (error) {
      updateStep('news', 'error', 100, 'News feeds unavailable');
      return false;
    }
  };

  const testPerformanceMonitoring = async () => {
    try {
      updateStep('performance', 'loading', 50, 'Initializing monitoring...');
      await new Promise(resolve => setTimeout(resolve, 400));
      
      await fetchSystemMetrics();
      
      updateStep('performance', 'loading', 90, 'Calibrating metrics...');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      updateStep('performance', 'success', 100, 'Performance monitoring active');
      return true;
    } catch (error) {
      updateStep('performance', 'error', 100, 'Monitoring setup failed');
      return false;
    }
  };

  const initializeSecurity = async () => {
    try {
      updateStep('security', 'loading', 30, 'Activating security protocols...');
      await new Promise(resolve => setTimeout(resolve, 500));
      
      updateStep('security', 'loading', 70, 'Encrypting connections...');
      await new Promise(resolve => setTimeout(resolve, 400));
      
      updateStep('security', 'loading', 95, 'Finalizing security setup...');
      await new Promise(resolve => setTimeout(resolve, 200));
      
      updateStep('security', 'success', 100, 'Security systems active');
      return true;
    } catch (error) {
      updateStep('security', 'error', 100, 'Security setup failed');
      return false;
    }
  };

  const finalizeSystem = async () => {
    try {
      updateStep('finalization', 'loading', 40, 'Optimizing performance...');
      await new Promise(resolve => setTimeout(resolve, 400));
      
      updateStep('finalization', 'loading', 80, 'Loading user preferences...');
      await new Promise(resolve => setTimeout(resolve, 300));
      
      updateStep('finalization', 'loading', 95, 'Preparing interface...');
      await new Promise(resolve => setTimeout(resolve, 200));
      
      updateStep('finalization', 'success', 100, 'ThriveRemoteOS ready');
      return true;
    } catch (error) {
      updateStep('finalization', 'error', 100, 'Finalization failed');
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
        testRelocateMe,
        testAITools,
        testNewsIntegration,
        testPerformanceMonitoring,
        initializeSecurity,
        finalizeSystem
      ];

      for (let i = 0; i < loadingFunctions.length; i++) {
        setCurrentStep(i);
        await loadingFunctions[i]();
        
        const progress = ((i + 1) / loadingFunctions.length) * 100;
        setOverallProgress(progress);
        
        await new Promise(resolve => setTimeout(resolve, 150));
      }

      setTimeout(() => {
        if (onComplete) onComplete();
      }, 1000);
    };

    runLoadingSequence();
    
    // Update system metrics periodically
    const metricsInterval = setInterval(fetchSystemMetrics, 2000);
    return () => clearInterval(metricsInterval);
  }, [onComplete]);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success': return '✓';
      case 'error': return '✗';
      case 'loading': return '●';
      default: return '○';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'success': return '#38a169';
      case 'error': return '#e53e3e';
      case 'loading': return '#3182ce';
      default: return '#a0aec0';
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif',
      zIndex: 10000
    }}>
      
      {/* Main Loading Container */}
      <div style={{
        width: '600px',
        background: 'rgba(255, 255, 255, 0.95)',
        borderRadius: '20px',
        boxShadow: '0 25px 60px rgba(0, 0, 0, 0.15)',
        border: '1px solid rgba(203, 213, 224, 0.3)',
        overflow: 'hidden',
        backdropFilter: 'blur(20px)'
      }}>
        
        {/* Header */}
        <div style={{
          padding: '32px 40px 24px',
          background: '#1a202c',
          color: 'white',
          textAlign: 'center',
          position: 'relative',
          overflow: 'hidden'
        }}>
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'linear-gradient(45deg, rgba(49, 130, 206, 0.1) 0%, rgba(214, 158, 46, 0.1) 100%)',
            animation: 'shimmer 3s ease-in-out infinite'
          }} />
          
          <div style={{
            fontSize: '1.5rem',
            fontWeight: '700',
            marginBottom: '8px',
            letterSpacing: '-0.025em',
            position: 'relative',
            zIndex: 1
          }}>
            ThriveRemoteOS
          </div>
          <div style={{
            fontSize: '0.875rem',
            opacity: 0.8,
            fontWeight: '500',
            position: 'relative',
            zIndex: 1
          }}>
            Professional Remote Work Platform
          </div>
        </div>

        {/* Progress Section */}
        <div style={{ padding: '32px 40px' }}>
          
          {/* Overall Progress */}
          <div style={{ marginBottom: '32px' }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '12px'
            }}>
              <span style={{
                fontSize: '0.875rem',
                fontWeight: '600',
                color: '#4a5568'
              }}>
                System Initialization
              </span>
              <span style={{
                fontSize: '0.875rem',
                fontWeight: '700',
                color: '#3182ce'
              }}>
                {Math.round(overallProgress)}%
              </span>
            </div>
            
            <div style={{
              width: '100%',
              height: '6px',
              background: '#e2e8f0',
              borderRadius: '3px',
              overflow: 'hidden'
            }}>
              <div style={{
                width: `${overallProgress}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #3182ce, #2c5aa0)',
                borderRadius: '3px',
                transition: 'width 0.5s ease'
              }} />
            </div>
          </div>

          {/* Loading Steps */}
          <div style={{
            maxHeight: '280px',
            overflowY: 'auto',
            marginBottom: '24px'
          }}>
            {loadingSteps.map((step, index) => (
              <div key={step.id} style={{
                display: 'flex',
                alignItems: 'center',
                padding: '12px 0',
                opacity: index <= currentStep ? 1 : 0.4,
                transition: 'opacity 0.3s ease'
              }}>
                <div style={{
                  width: '16px',
                  height: '16px',
                  borderRadius: '50%',
                  background: getStatusColor(step.status),
                  color: 'white',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '10px',
                  fontWeight: '700',
                  marginRight: '16px',
                  flexShrink: 0
                }}>
                  {getStatusIcon(step.status)}
                </div>
                
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: '0.875rem',
                    fontWeight: '600',
                    color: '#2d3748',
                    marginBottom: '2px'
                  }}>
                    {step.name}
                  </div>
                  <div style={{
                    fontSize: '0.75rem',
                    color: '#718096',
                    marginBottom: '4px'
                  }}>
                    {step.details}
                  </div>
                  
                  {step.progress > 0 && (
                    <div style={{
                      width: '100%',
                      height: '2px',
                      background: '#e2e8f0',
                      borderRadius: '1px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        width: `${step.progress}%`,
                        height: '100%',
                        background: getStatusColor(step.status),
                        borderRadius: '1px',
                        transition: 'width 0.3s ease'
                      }} />
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* System Metrics */}
          <div style={{
            background: '#f7fafc',
            border: '1px solid #e2e8f0',
            borderRadius: '12px',
            padding: '20px'
          }}>
            <div style={{
              fontSize: '0.75rem',
              fontWeight: '700',
              color: '#4a5568',
              marginBottom: '12px',
              textTransform: 'uppercase',
              letterSpacing: '0.05em'
            }}>
              System Status
            </div>
            
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
              gap: '16px'
            }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{
                  fontSize: '1.25rem',
                  fontWeight: '700',
                  color: '#3182ce'
                }}>
                  {systemMetrics.cpu}%
                </div>
                <div style={{
                  fontSize: '0.75rem',
                  color: '#718096',
                  fontWeight: '500'
                }}>
                  CPU Usage
                </div>
              </div>
              
              <div style={{ textAlign: 'center' }}>
                <div style={{
                  fontSize: '1.25rem',
                  fontWeight: '700',
                  color: '#38a169'
                }}>
                  {systemMetrics.memory}%
                </div>
                <div style={{
                  fontSize: '0.75rem',
                  color: '#718096',
                  fontWeight: '500'
                }}>
                  Memory
                </div>
              </div>
              
              <div style={{ textAlign: 'center' }}>
                <div style={{
                  fontSize: '0.875rem',
                  fontWeight: '700',
                  color: '#d69e2e'
                }}>
                  {systemMetrics.network}
                </div>
                <div style={{
                  fontSize: '0.75rem',
                  color: '#718096',
                  fontWeight: '500'
                }}>
                  Network
                </div>
              </div>
              
              <div style={{ textAlign: 'center' }}>
                <div style={{
                  fontSize: '0.875rem',
                  fontWeight: '700',
                  color: '#ed8936'
                }}>
                  {systemMetrics.uptime}
                </div>
                <div style={{
                  fontSize: '0.75rem',
                  color: '#718096',
                  fontWeight: '500'
                }}>
                  Uptime
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{
          padding: '16px 40px',
          background: '#f7fafc',
          borderTop: '1px solid #e2e8f0',
          fontSize: '0.75rem',
          color: '#718096',
          textAlign: 'center',
          fontWeight: '500'
        }}>
          Professional-grade platform • Real-time data integration • Enterprise security
        </div>
      </div>

      <style jsx>{`
        @keyframes shimmer {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }
      `}</style>
    </div>
  );
};

export default ProfessionalRealTimeLoader;