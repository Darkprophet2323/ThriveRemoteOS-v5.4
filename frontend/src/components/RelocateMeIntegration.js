import React, { useState, useEffect } from 'react';
import LiveDataService from './LiveDataIntegration';

const RelocateMeIntegration = () => {
  const [relocateData, setRelocateData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedOpportunity, setSelectedOpportunity] = useState(null);
  const [showApplication, setShowApplication] = useState(false);

  useEffect(() => {
    fetchRelocateOpportunities();
  }, []);

  const fetchRelocateOpportunities = async () => {
    try {
      const data = await LiveDataService.getRelocateOpportunities();
      setRelocateData(data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching relocate opportunities:', error);
      setLoading(false);
    }
  };

  const handleApplyClick = (opportunity) => {
    setSelectedOpportunity(opportunity);
    setShowApplication(true);
  };

  const handleApplicationSubmit = () => {
    // In real implementation, this would submit to RelocateMe API
    alert('Application submitted successfully! RelocateMe will contact you within 48 hours.');
    setShowApplication(false);
    setSelectedOpportunity(null);
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
          <div style={{ fontSize: '2rem', marginBottom: '10px' }}>🌍</div>
          <div>Loading global relocation opportunities...</div>
        </div>
      </div>
    );
  }

  if (!relocateData?.success) {
    return (
      <div className="app-content">
        <h3>🌍 RelocateMe - Global Opportunities</h3>
        <div style={{
          padding: '20px',
          background: 'rgba(244, 67, 54, 0.1)',
          borderRadius: '10px',
          border: '1px solid rgba(244, 67, 54, 0.3)',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '2rem', marginBottom: '10px' }}>⚠️</div>
          <div>Unable to connect to RelocateMe services</div>
          <button 
            className="luxury-btn"
            onClick={fetchRelocateOpportunities}
            style={{ marginTop: '10px', fontSize: '0.8rem' }}
          >
            🔄 Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-content">
      {!showApplication ? (
        <>
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h3 style={{ margin: 0 }}>🌍 RelocateMe - Global Tech Opportunities</h3>
            <button 
              className="luxury-btn"
              onClick={fetchRelocateOpportunities}
              style={{ fontSize: '0.8rem', padding: '8px 16px' }}
            >
              🔄 Refresh
            </button>
          </div>

          {/* Stats Overview */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
            gap: '15px',
            marginBottom: '25px'
          }}>
            <div style={{
              background: 'rgba(76, 175, 80, 0.1)',
              padding: '15px',
              borderRadius: '10px',
              border: '1px solid rgba(76, 175, 80, 0.3)',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '1.5rem', marginBottom: '5px' }}>🎯</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--charcoal-noir)', marginBottom: '2px' }}>Available</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#4CAF50' }}>
                {relocateData.total_opportunities}
              </div>
            </div>
            
            <div style={{
              background: 'rgba(33, 150, 243, 0.1)',
              padding: '15px',
              borderRadius: '10px',
              border: '1px solid rgba(33, 150, 243, 0.3)',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '1.5rem', marginBottom: '5px' }}>✈️</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--charcoal-noir)', marginBottom: '2px' }}>Active</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#2196F3' }}>
                {relocateData.active_relocations}
              </div>
            </div>
            
            <div style={{
              background: 'rgba(255, 152, 0, 0.1)',
              padding: '15px',
              borderRadius: '10px',
              border: '1px solid rgba(255, 152, 0, 0.3)',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '1.5rem', marginBottom: '5px' }}>🏆</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--charcoal-noir)', marginBottom: '2px' }}>Success</div>
              <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#FF9800' }}>
                {relocateData.success_stories}
              </div>
            </div>
          </div>

          {/* Featured Countries */}
          <div style={{
            background: 'rgba(212, 175, 55, 0.1)',
            padding: '15px',
            borderRadius: '10px',
            border: '1px solid rgba(212, 175, 55, 0.3)',
            marginBottom: '25px'
          }}>
            <h4 style={{ color: 'var(--warm-gold)', marginBottom: '10px' }}>🌟 Featured Countries</h4>
            <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
              {relocateData.featured_countries.map(country => (
                <span key={country} style={{
                  background: 'rgba(212, 175, 55, 0.2)',
                  color: 'var(--warm-gold)',
                  padding: '4px 12px',
                  borderRadius: '6px',
                  fontSize: '0.9rem',
                  fontWeight: '500'
                }}>
                  {country}
                </span>
              ))}
            </div>
          </div>

          {/* Opportunities List */}
          <div className="relocate-opportunities">
            {relocateData.opportunities.map((opportunity) => (
              <div key={opportunity.id} style={{
                background: 'rgba(248, 246, 244, 0.8)',
                borderRadius: '12px',
                padding: '20px',
                border: '1px solid rgba(212, 175, 55, 0.2)',
                marginBottom: '15px',
                transition: 'all 0.3s ease'
              }}>
                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '15px' }}>
                  <div style={{ flex: 1 }}>
                    <h4 style={{ margin: 0, color: 'var(--noir-black)', marginBottom: '5px' }}>
                      {opportunity.title}
                    </h4>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '15px', fontSize: '0.9rem', color: 'var(--charcoal-noir)' }}>
                      <span><strong>{opportunity.company}</strong></span>
                      <span>📍 {opportunity.location}</span>
                      <span style={{ color: 'var(--warm-gold)', fontWeight: '600' }}>💰 {opportunity.salary}</span>
                    </div>
                  </div>
                  <div style={{
                    background: 'rgba(76, 175, 80, 0.2)',
                    color: '#4CAF50',
                    padding: '4px 8px',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    fontWeight: '500'
                  }}>
                    Visa Supported
                  </div>
                </div>

                {/* Relocation Package */}
                <div style={{ marginBottom: '15px' }}>
                  <h5 style={{ color: 'var(--warm-gold)', marginBottom: '8px' }}>📦 Relocation Package</h5>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '8px' }}>
                    <div style={{ fontSize: '0.85rem' }}>
                      ✈️ Moving Allowance: <strong>{opportunity.relocation_package.moving_allowance}</strong>
                    </div>
                    <div style={{ fontSize: '0.85rem' }}>
                      🏠 Housing: <strong>{opportunity.relocation_package.temporary_housing}</strong>
                    </div>
                    <div style={{ fontSize: '0.85rem' }}>
                      📋 Visa Support: <strong>{opportunity.relocation_package.visa_support ? 'Yes' : 'No'}</strong>
                    </div>
                    <div style={{ fontSize: '0.85rem' }}>
                      👨‍👩‍👧‍👦 Family Support: <strong>{opportunity.relocation_package.family_support ? 'Yes' : 'No'}</strong>
                    </div>
                  </div>
                </div>

                {/* Benefits */}
                <div style={{ marginBottom: '15px' }}>
                  <h5 style={{ color: 'var(--warm-gold)', marginBottom: '8px' }}>🎁 Benefits & Support</h5>
                  <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    {opportunity.benefits.map((benefit, index) => (
                      <span key={index} style={{
                        background: 'rgba(33, 150, 243, 0.1)',
                        color: '#2196F3',
                        padding: '3px 8px',
                        borderRadius: '4px',
                        fontSize: '0.8rem'
                      }}>
                        {benefit}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Requirements */}
                <div style={{ marginBottom: '15px' }}>
                  <h5 style={{ color: 'var(--warm-gold)', marginBottom: '8px' }}>📋 Requirements</h5>
                  <div style={{ fontSize: '0.85rem', color: 'var(--charcoal-noir)' }}>
                    {opportunity.requirements.join(' • ')}
                  </div>
                </div>

                {/* Actions */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--charcoal-noir)' }}>
                    ⏰ Application Deadline: <strong>{new Date(opportunity.deadline).toLocaleDateString()}</strong>
                  </div>
                  
                  <div style={{ display: 'flex', gap: '10px' }}>
                    <button
                      onClick={() => handleApplyClick(opportunity)}
                      className="luxury-btn"
                      style={{ fontSize: '0.9rem', padding: '8px 16px' }}
                    >
                      🚀 Apply via RelocateMe
                    </button>
                    
                    <button
                      style={{
                        background: 'rgba(33, 150, 243, 0.1)',
                        color: '#2196F3',
                        border: '1px solid rgba(33, 150, 243, 0.3)',
                        padding: '8px 16px',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '0.9rem',
                        fontWeight: '500'
                      }}
                    >
                      📄 Learn More
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      ) : (
        /* Application Form */
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h3 style={{ margin: 0 }}>🚀 Apply via RelocateMe</h3>
            <button
              onClick={() => setShowApplication(false)}
              style={{
                background: 'rgba(158, 158, 158, 0.1)',
                color: '#9E9E9E',
                border: '1px solid rgba(158, 158, 158, 0.3)',
                padding: '6px 12px',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '0.8rem'
              }}
            >
              ✖️ Close
            </button>
          </div>

          {selectedOpportunity && (
            <div style={{
              background: 'rgba(248, 246, 244, 0.8)',
              padding: '20px',
              borderRadius: '12px',
              border: '1px solid rgba(212, 175, 55, 0.3)',
              marginBottom: '20px'
            }}>
              <h4 style={{ color: 'var(--noir-black)', marginBottom: '10px' }}>
                {selectedOpportunity.title}
              </h4>
              <div style={{ fontSize: '0.9rem', color: 'var(--charcoal-noir)' }}>
                <strong>{selectedOpportunity.company}</strong> • {selectedOpportunity.location} • {selectedOpportunity.salary}
              </div>
            </div>
          )}

          <div style={{
            background: 'rgba(248, 246, 244, 0.5)',
            padding: '25px',
            borderRadius: '12px',
            border: '1px solid rgba(212, 175, 55, 0.2)'
          }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
              <input
                type="text"
                placeholder="Full Name"
                style={{
                  padding: '12px',
                  borderRadius: '8px',
                  border: '1px solid rgba(212, 175, 55, 0.3)',
                  background: 'rgba(255, 255, 255, 0.9)',
                  fontSize: '0.9rem'
                }}
              />
              
              <input
                type="email"
                placeholder="Email Address"
                style={{
                  padding: '12px',
                  borderRadius: '8px',
                  border: '1px solid rgba(212, 175, 55, 0.3)',
                  background: 'rgba(255, 255, 255, 0.9)',
                  fontSize: '0.9rem'
                }}
              />
              
              <input
                type="text"
                placeholder="Current Location"
                style={{
                  padding: '12px',
                  borderRadius: '8px',
                  border: '1px solid rgba(212, 175, 55, 0.3)',
                  background: 'rgba(255, 255, 255, 0.9)',
                  fontSize: '0.9rem'
                }}
              />
              
              <textarea
                placeholder="Why are you interested in relocating? Tell us about your experience..."
                rows="4"
                style={{
                  padding: '12px',
                  borderRadius: '8px',
                  border: '1px solid rgba(212, 175, 55, 0.3)',
                  background: 'rgba(255, 255, 255, 0.9)',
                  fontSize: '0.9rem',
                  fontFamily: 'inherit',
                  resize: 'vertical'
                }}
              />
              
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.85rem' }}>
                <input type="checkbox" id="terms" />
                <label htmlFor="terms" style={{ color: 'var(--charcoal-noir)' }}>
                  I agree to RelocateMe's terms and conditions and privacy policy
                </label>
              </div>
              
              <button
                onClick={handleApplicationSubmit}
                className="luxury-btn"
                style={{ alignSelf: 'flex-start', padding: '12px 24px' }}
              >
                🚀 Submit Application
              </button>
            </div>
          </div>

          <div style={{
            marginTop: '20px',
            padding: '15px',
            background: 'rgba(33, 150, 243, 0.1)',
            borderRadius: '10px',
            border: '1px solid rgba(33, 150, 243, 0.3)'
          }}>
            <h5 style={{ color: '#2196F3', marginBottom: '10px' }}>📞 What happens next?</h5>
            <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '0.85rem', color: 'var(--charcoal-noir)' }}>
              <li>RelocateMe team reviews your application (24-48 hours)</li>
              <li>Initial phone/video screening with relocation specialist</li>
              <li>Company introduction and interview coordination</li>
              <li>Visa and relocation planning support</li>
              <li>End-to-end assistance throughout the process</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default RelocateMeIntegration;