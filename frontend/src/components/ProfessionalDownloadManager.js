import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const ProfessionalDownloadManager = () => {
  const [downloads, setDownloads] = useState([]);
  const [newDownload, setNewDownload] = useState({ url: '', filename: '', category: 'documents' });
  const [loading, setLoading] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('created_date');

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
        setNewDownload({ url: '', filename: '', category: 'documents' });
        setShowAddForm(false);
        fetchDownloads();
        simulateDownloadProgress(response.data.download_id);
      }
    } catch (error) {
      console.error('Error starting download:', error);
    }
  };

  const simulateDownloadProgress = async (downloadId) => {
    for (let progress = 0; progress <= 100; progress += Math.random() * 15 + 5) {
      setTimeout(async () => {
        try {
          await axios.put(`${API}/downloads/${downloadId}/progress`, {
            progress: Math.min(progress, 100),
            status: progress >= 100 ? 'completed' : 'downloading'
          });
          fetchDownloads();
        } catch (error) {
          console.error('Error updating progress:', error);
        }
      }, progress * 50);
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
      case 'completed': return '✓';
      case 'downloading': return '↓';
      case 'failed': return '✗';
      case 'cancelled': return '○';
      default: return '○';
    }
  };

  const getCategoryIcon = (category) => {
    const icons = {
      documents: 'description',
      images: 'image',
      videos: 'videocam',
      audio: 'music_note',
      software: 'code',
      general: 'file_copy'
    };
    return icons[category] || 'file_copy';
  };

  const filteredDownloads = downloads.filter(download => {
    if (filter === 'all') return true;
    return download.status === filter || download.category === filter;
  });

  const sortedDownloads = [...filteredDownloads].sort((a, b) => {
    switch (sortBy) {
      case 'created_date':
        return new Date(b.created_date) - new Date(a.created_date);
      case 'filename':
        return a.filename.localeCompare(b.filename);
      case 'progress':
        return (b.progress || 0) - (a.progress || 0);
      default:
        return 0;
    }
  });

  if (loading) {
    return (
      <div className="app-content">
        <div className="flex items-center justify-center" style={{ minHeight: '200px' }}>
          <div className="loading">
            <div className="text-lg font-semibold text-secondary mb-2">Loading Downloads</div>
            <div className="text-sm text-muted">Fetching download history...</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-content">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-primary mb-1">Download Manager</h1>
          <p className="text-sm text-muted">Manage and track your downloads</p>
        </div>
        <button 
          className="btn-primary"
          onClick={() => setShowAddForm(!showAddForm)}
        >
          {showAddForm ? 'Cancel' : 'Add Download'}
        </button>
      </div>

      {/* Add Download Form */}
      {showAddForm && (
        <div className="card mb-6">
          <div className="card-header">
            <h3 className="text-lg font-semibold">Add New Download</h3>
          </div>
          <div className="card-content">
            <div className="grid gap-4">
              <div className="form-group">
                <label className="form-label">Download URL</label>
                <input
                  type="url"
                  className="form-input"
                  placeholder="https://example.com/file.pdf"
                  value={newDownload.url}
                  onChange={(e) => setNewDownload({ ...newDownload, url: e.target.value })}
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="form-group">
                  <label className="form-label">Filename (optional)</label>
                  <input
                    type="text"
                    className="form-input"
                    placeholder="Auto-detected from URL"
                    value={newDownload.filename}
                    onChange={(e) => setNewDownload({ ...newDownload, filename: e.target.value })}
                  />
                </div>
                
                <div className="form-group">
                  <label className="form-label">Category</label>
                  <select
                    className="form-select"
                    value={newDownload.category}
                    onChange={(e) => setNewDownload({ ...newDownload, category: e.target.value })}
                  >
                    <option value="documents">Documents</option>
                    <option value="images">Images</option>
                    <option value="videos">Videos</option>
                    <option value="audio">Audio</option>
                    <option value="software">Software</option>
                    <option value="general">General</option>
                  </select>
                </div>
              </div>
              
              <button 
                className="btn-primary"
                onClick={startDownload}
                disabled={!newDownload.url}
              >
                Start Download
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Filters and Controls */}
      <div className="card mb-6">
        <div className="card-content p-4">
          <div className="flex justify-between items-center">
            <div className="flex gap-3">
              <select
                className="form-select"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                style={{ width: '160px' }}
              >
                <option value="all">All Downloads</option>
                <option value="completed">Completed</option>
                <option value="downloading">In Progress</option>
                <option value="failed">Failed</option>
                <option value="documents">Documents</option>
                <option value="images">Images</option>
                <option value="videos">Videos</option>
              </select>
              
              <select
                className="form-select"
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                style={{ width: '140px' }}
              >
                <option value="created_date">Date Added</option>
                <option value="filename">Filename</option>
                <option value="progress">Progress</option>
              </select>
            </div>
            
            <div className="text-sm text-muted">
              {filteredDownloads.length} of {downloads.length} downloads
            </div>
          </div>
        </div>
      </div>

      {/* Downloads List */}
      <div className="grid gap-3">
        {sortedDownloads.length === 0 ? (
          <div className="card">
            <div className="card-content text-center py-12">
              <div className="text-4xl mb-4">📥</div>
              <h3 className="text-lg font-semibold text-secondary mb-2">No downloads found</h3>
              <p className="text-sm text-muted">
                {filter === 'all' 
                  ? 'Click "Add Download" to start downloading files'
                  : `No downloads match the current filter: ${filter}`
                }
              </p>
            </div>
          </div>
        ) : (
          sortedDownloads.map((download) => (
            <div key={download.id} className="card">
              <div className="card-content p-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4 flex-1">
                    {/* File Icon */}
                    <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-secondary">
                      <span className="material-icons-outlined text-primary text-lg">
                        {getCategoryIcon(download.category)}
                      </span>
                    </div>
                    
                    {/* Download Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-semibold text-primary truncate">{download.filename}</h4>
                        <span className={`px-2 py-1 rounded text-xs font-medium
                          ${download.status === 'completed' ? 'bg-green-100 text-green-800' :
                            download.status === 'downloading' ? 'bg-blue-100 text-blue-800' :
                            download.status === 'failed' ? 'bg-red-100 text-red-800' :
                            'bg-gray-100 text-gray-800'}`}>
                          {getStatusIcon(download.status)} {download.status}
                        </span>
                        <span className="px-2 py-1 rounded text-xs font-medium bg-secondary text-secondary">
                          {download.category}
                        </span>
                      </div>
                      
                      {/* Progress Bar */}
                      {download.status === 'downloading' && (
                        <div className="mb-2">
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-xs text-muted">Progress</span>
                            <span className="text-xs font-medium text-primary">
                              {Math.round(download.progress || 0)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${download.progress || 0}%` }}
                            ></div>
                          </div>
                        </div>
                      )}
                      
                      {/* URL and Metadata */}
                      <div className="text-xs text-muted">
                        <div className="truncate mb-1">URL: {download.url}</div>
                        <div className="flex gap-4">
                          <span>Added: {new Date(download.created_date).toLocaleDateString()}</span>
                          {download.completed_date && (
                            <span>Completed: {new Date(download.completed_date).toLocaleDateString()}</span>
                          )}
                          {download.size && <span>Size: {download.size}</span>}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex flex-col gap-2 ml-4">
                    {download.status === 'downloading' && (
                      <button
                        onClick={() => cancelDownload(download.id)}
                        className="btn-subtle text-xs px-3 py-1"
                      >
                        Cancel
                      </button>
                    )}
                    
                    {download.status === 'completed' && (
                      <button className="btn-secondary text-xs px-3 py-1">
                        Open
                      </button>
                    )}
                    
                    <button
                      onClick={() => cancelDownload(download.id)}
                      className="btn-subtle text-xs px-3 py-1"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Statistics */}
      <div className="card mt-6">
        <div className="card-header">
          <h3 className="text-lg font-semibold">Download Statistics</h3>
        </div>
        <div className="card-content">
          <div className="grid grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">{downloads.length}</div>
              <div className="text-sm text-muted">Total Downloads</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {downloads.filter(d => d.status === 'completed').length}
              </div>
              <div className="text-sm text-muted">Completed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {downloads.filter(d => d.status === 'downloading').length}
              </div>
              <div className="text-sm text-muted">In Progress</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {downloads.filter(d => d.status === 'failed' || d.status === 'cancelled').length}
              </div>
              <div className="text-sm text-muted">Failed</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfessionalDownloadManager;