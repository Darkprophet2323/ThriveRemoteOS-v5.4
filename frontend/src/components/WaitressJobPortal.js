import React, { useState, useEffect } from 'react';
import axios from 'axios';

const WaitressJobPortal = () => {
  const [aiTools, setAITools] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('ai-tools');

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch AI Tools
        const toolsResponse = await axios.get(`${BACKEND_URL}/api/content/ai-tools`);
        setAITools(toolsResponse.data);

        // Fetch Live Jobs
        const jobsResponse = await axios.get(`${BACKEND_URL}/api/jobs/live`);
        setJobs(jobsResponse.data.jobs || []);

        setLoading(false);
      } catch (error) {
        console.error('Error fetching portal data:', error);
        setLoading(false);
      }
    };

    fetchData();
  }, [BACKEND_URL]);

  if (loading) {
    return (
      <div style={{ padding: '30px', textAlign: 'center' }}>
        <h3>🤖 Loading AI Career Portal...</h3>
        <p>Preparing your comprehensive job application toolkit...</p>
      </div>
    );
  }

  return (
    <div style={{ padding: '25px', height: '100%', overflow: 'auto' }}>
      <div style={{ marginBottom: '25px' }}>
        <h2 style={{ 
          fontFamily: 'Playfair Display, serif', 
          color: '#0D0D0D', 
          marginBottom: '10px',
          borderBottom: '2px solid #D4AF37',
          paddingBottom: '10px'
        }}>
          🎭 AI Career Acceleration Portal
        </h2>
        <p style={{ color: '#666', fontSize: '1rem' }}>
          Your comprehensive platform for AI-powered job searching with <strong>{aiTools?.total_tools || 120}+ premium tools</strong>
        </p>
      </div>

      {/* Tab Navigation */}
      <div style={{ 
        display: 'flex', 
        gap: '10px', 
        marginBottom: '25px',
        borderBottom: '1px solid rgba(212, 175, 55, 0.3)'
      }}>
        <button
          onClick={() => setActiveTab('ai-tools')}
          style={{
            padding: '10px 20px',
            border: 'none',
            borderBottom: activeTab === 'ai-tools' ? '3px solid #D4AF37' : '3px solid transparent',
            background: 'transparent',
            color: activeTab === 'ai-tools' ? '#D4AF37' : '#666',
            fontWeight: activeTab === 'ai-tools' ? '600' : '400',
            cursor: 'pointer',
            fontSize: '1rem'
          }}
        >
          🤖 AI Tools ({aiTools?.total_tools || 120})
        </button>
        <button
          onClick={() => setActiveTab('live-jobs')}
          style={{
            padding: '10px 20px',
            border: 'none',
            borderBottom: activeTab === 'live-jobs' ? '3px solid #D4AF37' : '3px solid transparent',
            background: 'transparent',
            color: activeTab === 'live-jobs' ? '#D4AF37' : '#666',
            fontWeight: activeTab === 'live-jobs' ? '600' : '400',
            cursor: 'pointer',
            fontSize: '1rem'
          }}
        >
          💼 Live Jobs ({jobs.length})
        </button>
      </div>

      {/* AI Tools Tab */}
      {activeTab === 'ai-tools' && (
        <div>
          <div style={{ 
            background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.1), rgba(232, 180, 184, 0.1))',
            padding: '20px',
            borderRadius: '12px',
            marginBottom: '25px',
            border: '1px solid rgba(212, 175, 55, 0.3)'
          }}>
            <h3 style={{ margin: '0 0 10px 0', color: '#D4AF37' }}>🎯 Quick Actions</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
              <button 
                onClick={() => window.open('https://aiapply.co/', '_blank')}
                style={{
                  background: 'linear-gradient(135deg, #D4AF37, #FFD700)',
                  color: '#0D0D0D',
                  border: 'none',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease'
                }}
              >
                🚀 AI Apply Tool
              </button>
              <button 
                onClick={() => window.open('https://resumeworded.com/', '_blank')}
                style={{
                  background: 'linear-gradient(135deg, #E8B4B8, #F7E7CE)',
                  color: '#0D0D0D',
                  border: 'none',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease'
                }}
              >
                📝 Resume AI Review
              </button>
              <button 
                onClick={() => window.open('https://grow.google/certificates/interview-warmup/', '_blank')}
                style={{
                  background: 'linear-gradient(135deg, #B8860B, #D4AF37)',
                  color: '#0D0D0D',
                  border: 'none',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease'
                }}
              >
                🎯 Interview Prep
              </button>
            </div>
          </div>

          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', 
            gap: '20px' 
          }}>
            {Object.entries(aiTools?.categories || {}).map(([key, category]) => (
              <div key={`category-${key}`} style={{
                background: 'rgba(248, 246, 244, 0.8)',
                borderRadius: '12px',
                padding: '20px',
                border: '1px solid rgba(212, 175, 55, 0.2)',
                transition: 'all 0.3s ease'
              }}>
                <h4 style={{ 
                  color: '#D4AF37', 
                  marginBottom: '15px',
                  fontFamily: 'Playfair Display, serif'
                }}>
                  {category.name} ({category.count} tools)
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {category.tools?.slice(0, 4).map((tool, index) => (
                    <div key={`${key}-tool-${index}`} style={{
                      background: 'rgba(255, 255, 255, 0.9)',
                      padding: '12px',
                      borderRadius: '8px',
                      border: '1px solid rgba(212, 175, 55, 0.2)',
                      transition: 'all 0.3s ease'
                    }}>
                      <div style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        alignItems: 'flex-start',
                        marginBottom: '8px'
                      }}>
                        <strong style={{ color: '#0D0D0D', fontSize: '0.95rem' }}>
                          {tool.name}
                        </strong>
                        <a 
                          href={tool.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          style={{
                            background: 'linear-gradient(135deg, #D4AF37, #FFD700)',
                            color: '#0D0D0D',
                            padding: '4px 8px',
                            borderRadius: '6px',
                            textDecoration: 'none',
                            fontSize: '0.8rem',
                            fontWeight: '500'
                          }}
                        >
                          Visit →
                        </a>
                      </div>
                      <p style={{ 
                        margin: '0', 
                        color: '#666', 
                        fontSize: '0.85rem',
                        lineHeight: '1.4'
                      }}>
                        {tool.description}
                      </p>
                    </div>
                  ))}
                  {category.count > 4 && (
                    <p style={{ 
                      textAlign: 'center', 
                      color: '#D4AF37', 
                      fontStyle: 'italic',
                      margin: '10px 0 0 0'
                    }}>
                      +{category.count - 4} more tools available
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div style={{ 
            marginTop: '30px', 
            padding: '20px', 
            background: 'linear-gradient(135deg, rgba(13, 13, 13, 0.05), rgba(45, 45, 45, 0.05))',
            borderRadius: '12px',
            border: '1px solid rgba(212, 175, 55, 0.3)'
          }}>
            <h4 style={{ color: '#D4AF37', marginBottom: '15px' }}>🌟 Featured Premium Tools</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '15px' }}>
              {aiTools?.featured_tools?.map((tool, index) => (
                <div key={`featured-${tool}-${index}`} style={{
                  background: 'linear-gradient(135deg, #D4AF37, #FFD700)',
                  color: '#0D0D0D',
                  padding: '15px',
                  borderRadius: '8px',
                  textAlign: 'center',
                  fontWeight: '600',
                  fontSize: '0.9rem'
                }}>
                  {tool}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Live Jobs Tab */}
      {activeTab === 'live-jobs' && (
        <div>
          <div style={{ 
            background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.1), rgba(232, 180, 184, 0.1))',
            padding: '20px',
            borderRadius: '12px',
            marginBottom: '25px',
            border: '1px solid rgba(212, 175, 55, 0.3)'
          }}>
            <h3 style={{ margin: '0 0 10px 0', color: '#D4AF37' }}>💼 Live Opportunities</h3>
            <p style={{ margin: '0', color: '#666' }}>
              <strong>{jobs.length} active positions</strong> from multiple sources including Remotive API and curated opportunities
            </p>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {jobs.slice(0, 15).map((job, index) => (
              <div key={`${job.id || job.title}-${index}`} style={{
                background: 'rgba(248, 246, 244, 0.9)',
                borderRadius: '12px',
                padding: '20px',
                border: '1px solid rgba(212, 175, 55, 0.2)',
                transition: 'all 0.3s ease'
              }}>
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'flex-start',
                  marginBottom: '12px'
                }}>
                  <h4 style={{ 
                    margin: '0',
                    color: '#0D0D0D',
                    fontFamily: 'Playfair Display, serif',
                    fontSize: '1.1rem'
                  }}>
                    {job.title}
                  </h4>
                  <div style={{
                    background: 'linear-gradient(135deg, #D4AF37, #FFD700)',
                    color: '#0D0D0D',
                    padding: '6px 12px',
                    borderRadius: '8px',
                    fontWeight: '600',
                    fontSize: '0.85rem'
                  }}>
                    {job.salary}
                  </div>
                </div>
                
                <div style={{ marginBottom: '12px' }}>
                  <p style={{ margin: '0 0 8px 0', color: '#333', fontWeight: '500' }}>
                    <strong>{job.company}</strong> • {job.location}
                  </p>
                  <p style={{ margin: '0', color: '#666', fontSize: '0.9rem', lineHeight: '1.4' }}>
                    {job.description}
                  </p>
                </div>

                {job.skills && job.skills.length > 0 && (
                  <div style={{ marginBottom: '15px' }}>
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                      {job.skills.slice(0, 4).map((skill, skillIndex) => (
                        <span key={skillIndex} style={{
                          background: 'rgba(232, 180, 184, 0.3)',
                          color: '#0D0D0D',
                          padding: '4px 8px',
                          borderRadius: '6px',
                          fontSize: '0.8rem',
                          fontWeight: '500'
                        }}>
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center'
                }}>
                  <a 
                    href={job.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    style={{
                      background: 'linear-gradient(135deg, #D4AF37, #FFD700)',
                      color: '#0D0D0D',
                      padding: '10px 20px',
                      borderRadius: '8px',
                      textDecoration: 'none',
                      fontWeight: '600',
                      transition: 'all 0.3s ease'
                    }}
                  >
                    Apply Now →
                  </a>
                  <span style={{ 
                    fontSize: '0.8rem', 
                    color: '#D4AF37',
                    opacity: 0.8,
                    fontStyle: 'italic'
                  }}>
                    via {job.source}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {jobs.length > 15 && (
            <div style={{ 
              textAlign: 'center', 
              marginTop: '25px',
              padding: '20px',
              background: 'rgba(212, 175, 55, 0.1)',
              borderRadius: '12px'
            }}>
              <p style={{ margin: '0', color: '#D4AF37', fontWeight: '500' }}>
                +{jobs.length - 15} more opportunities available. 
                <br />
                <span style={{ fontSize: '0.9rem', opacity: 0.8 }}>
                  Use AI Apply tool for automated applications across all listings.
                </span>
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default WaitressJobPortal;
