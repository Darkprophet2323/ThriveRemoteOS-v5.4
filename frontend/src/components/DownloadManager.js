import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const DownloadManager = () => {
  const [downloads, setDownloads] = useState([]);
  const [newDownload, setNewDownload] = useState({ url: '', filename: '', category: 'general' });
  const [loading, setLoading] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);

  useEffect(() => {
    fetchDownloads();
  }, []);

  const fetchDownloads = async () => {
    try {
      const response = await axios.get(`${API}/downloads`);
      setDownloads(response.data.downloads || []);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching downloads:', error);
      setLoading(false);
    }
  };

  const startDownload = async () => {
    if (!newDownload.url) return;

    try {
      const response = await axios.post(`${API}/downloads/start`, newDownload);
      if (response.data.success) {
        setNewDownload({ url: '', filename: '', category: 'general' });
        setShowAddForm(false);
        fetchDownloads();
        
        // Simulate download progress
        simulateDownloadProgress(response.data.download_id);
      }
    } catch (error) {
      console.error('Error starting download:', error);
    }
  };

  const simulateDownloadProgress = async (downloadId) => {
    for (let progress = 0; progress <= 100; progress += 10) {
      setTimeout(async () => {
        try {
          await axios.put(`${API}/downloads/${downloadId}/progress`, {
            progress,
            status: progress === 100 ? 'completed' : 'downloading'
          });
          fetchDownloads();
        } catch (error) {
          console.error('Error updating progress:', error);
        }
      }, progress * 100);
    }
  };

  const cancelDownload = async (downloadId) => {
    try {
      await axios.delete(`${API}/downloads/${downloadId}`);
      fetchDownloads();
    } catch (error) {
      console.error('Error cancelling download:', error);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return '✅';
      case 'downloading': return '⏳';
      case 'failed': return '❌';
      case 'cancelled': return '🚫';
      default: return '⏸️';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return '#4CAF50';
      case 'downloading': return '#2196F3';
      case 'failed': return '#F44336';
      case 'cancelled': return '#FF9800';
      default: return '#9E9E9E';
    }
  };

  if (loading) {
    return (
      <div className="app-content">
        <h3>📥 Loading Download Manager...</h3>
      </div>
    );
  }

  return (
    <div className="app-content">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h3>📥 Download Manager</h3>
        <button 
          className="luxury-btn"
          onClick={() => setShowAddForm(!showAddForm)}
          style={{ fontSize: '0.9rem', padding: '8px 16px' }}
        >
          {showAddForm ? '✖️ Cancel' : '➕ New Download'}
        </button>
      </div>

      {showAddForm && (
        <div style={{
          background: 'rgba(248, 246, 244, 0.8)',
          padding: '20px',
          borderRadius: '12px',
          border: '1px solid rgba(212, 175, 55, 0.3)',
          marginBottom: '20px'
        }}>
          <h4 style={{ color: 'var(--warm-gold)', marginBottom: '15px' }}>🔗 Add New Download</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <input
              type="text"
              placeholder="Download URL (e.g., https://example.com/file.pdf)"
              value={newDownload.url}
              onChange={(e) => setNewDownload({ ...newDownload, url: e.target.value })}
              style={{
                padding: '10px',
                borderRadius: '8px',
                border: '1px solid rgba(212, 175, 55, 0.3)',
                background: 'rgba(255, 255, 255, 0.9)',
                fontSize: '0.9rem'
              }}
            />
            <input
              type="text"
              placeholder="Filename (optional - auto-detected from URL)"
              value={newDownload.filename}
              onChange={(e) => setNewDownload({ ...newDownload, filename: e.target.value })}
              style={{
                padding: '10px',
                borderRadius: '8px',
                border: '1px solid rgba(212, 175, 55, 0.3)',
                background: 'rgba(255, 255, 255, 0.9)',
                fontSize: '0.9rem'
              }}
            />
            <select
              value={newDownload.category}
              onChange={(e) => setNewDownload({ ...newDownload, category: e.target.value })}
              style={{
                padding: '10px',
                borderRadius: '8px',
                border: '1px solid rgba(212, 175, 55, 0.3)',
                background: 'rgba(255, 255, 255, 0.9)',
                fontSize: '0.9rem'
              }}
            >
              <option value="general">📄 General</option>
              <option value="documents">📝 Documents</option>
              <option value="images">🖼️ Images</option>
              <option value="videos">🎥 Videos</option>
              <option value="audio">🎵 Audio</option>
              <option value="software">💾 Software</option>
            </select>
            <button 
              className="luxury-btn"
              onClick={startDownload}
              style={{ alignSelf: 'flex-start' }}
            >
              🚀 Start Download
            </button>
          </div>
        </div>
      )}

      <div className="downloads-list">
        {downloads.length === 0 ? (
          <div style={{
            textAlign: 'center',
            padding: '40px',
            color: 'var(--charcoal-noir)',
            opacity: 0.7
          }}>
            <div style={{ fontSize: '3rem', marginBottom: '10px' }}>📥</div>
            <p>No downloads yet</p>
            <p style={{ fontSize: '0.9rem' }}>Click "New Download" to start downloading files</p>
          </div>
        ) : (
          downloads.map((download) => (
            <div key={download.id} style={{
              background: 'rgba(248, 246, 244, 0.8)',
              borderRadius: '12px',
              padding: '15px',
              border: '1px solid rgba(212, 175, 55, 0.2)',
              marginBottom: '10px',
              transition: 'all 0.3s ease'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                    <span style={{ fontSize: '1.2rem' }}>{getStatusIcon(download.status)}</span>
                    <h4 style={{ margin: 0, color: 'var(--noir-black)' }}>{download.filename}</h4>
                    <span style={{
                      background: 'rgba(212, 175, 55, 0.2)',
                      color: 'var(--warm-gold)',
                      padding: '2px 8px',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      fontWeight: '500'
                    }}>
                      {download.category}
                    </span>
                  </div>

                  <div style={{ marginBottom: '10px' }}>
                    <div style={{
                      fontSize: '0.8rem',
                      color: 'var(--charcoal-noir)',
                      marginBottom: '5px'
                    }}>
                      Status: <span style={{ color: getStatusColor(download.status), fontWeight: '500' }}>
                        {download.status.charAt(0).toUpperCase() + download.status.slice(1)}
                      </span> • Progress: {Math.round(download.progress || 0)}%
                    </div>
                    
                    <div style={{
                      width: '100%',
                      height: '6px',
                      background: 'rgba(212, 175, 55, 0.2)',
                      borderRadius: '3px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        width: `${download.progress || 0}%`,
                        height: '100%',
                        background: 'linear-gradient(135deg, var(--warm-gold), var(--rich-gold))',
                        transition: 'width 0.3s ease'
                      }}></div>
                    </div>
                  </div>

                  <div style={{ fontSize: '0.8rem', color: 'var(--charcoal-noir)', opacity: 0.8 }}>
                    <div>📅 Started: {new Date(download.created_date).toLocaleString()}</div>
                    {download.completed_date && (
                      <div>✅ Completed: {new Date(download.completed_date).toLocaleString()}</div>
                    )}
                    <div>🔗 URL: {download.url}</div>
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginLeft: '15px' }}>
                  {download.status === 'downloading' && (
                    <button
                      onClick={() => cancelDownload(download.id)}
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
                      🚫 Cancel
                    </button>
                  )}
                  
                  {download.status === 'completed' && (
                    <button
                      style={{
                        background: 'rgba(76, 175, 80, 0.1)',
                        color: '#4CAF50',
                        border: '1px solid rgba(76, 175, 80, 0.3)',
                        padding: '6px 12px',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '0.8rem',
                        fontWeight: '500'
                      }}
                    >
                      📂 Open
                    </button>
                  )}
                  
                  <button
                    onClick={() => cancelDownload(download.id)}
                    style={{
                      background: 'rgba(158, 158, 158, 0.1)',
                      color: '#9E9E9E',
                      border: '1px solid rgba(158, 158, 158, 0.3)',
                      padding: '6px 12px',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      fontWeight: '500'
                    }}
                  >
                    🗑️ Remove
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      <div style={{
        marginTop: '20px',
        padding: '15px',
        background: 'rgba(212, 175, 55, 0.1)',
        borderRadius: '10px',
        border: '1px solid rgba(212, 175, 55, 0.3)'
      }}>
        <h4 style={{ color: 'var(--warm-gold)', marginBottom: '10px' }}>📊 Download Statistics</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '15px' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'var(--noir-black)' }}>
              {downloads.length}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--charcoal-noir)' }}>Total Downloads</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#4CAF50' }}>
              {downloads.filter(d => d.status === 'completed').length}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--charcoal-noir)' }}>Completed</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#2196F3' }}>
              {downloads.filter(d => d.status === 'downloading').length}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--charcoal-noir)' }}>In Progress</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#FF9800' }}>
              {downloads.filter(d => d.status === 'failed' || d.status === 'cancelled').length}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--charcoal-noir)' }}>Failed/Cancelled</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DownloadManager;