import React, { useState, useEffect } from 'react';

const VaultApp = () => {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'list'
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('modified'); // 'name', 'modified', 'size'

  useEffect(() => {
    loadVaultFiles();
  }, []);

  const loadVaultFiles = () => {
    const saved = localStorage.getItem('thriveOS_vault_files');
    if (saved) {
      setFiles(JSON.parse(saved));
    }
  };

  const deleteFile = (fileId) => {
    const updatedFiles = files.filter(f => f.id !== fileId);
    setFiles(updatedFiles);
    localStorage.setItem('thriveOS_vault_files', JSON.stringify(updatedFiles));
    if (selectedFile && selectedFile.id === fileId) {
      setSelectedFile(null);
    }
  };

  const downloadFile = (file) => {
    const blob = new Blob([file.content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = file.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' bytes';
    if (bytes < 1024 * 1024) return Math.round(bytes / 1024) + ' KB';
    return Math.round(bytes / (1024 * 1024)) + ' MB';
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  };

  const getFileIcon = (file) => {
    if (file.type === 'document') return '📄';
    if (file.name.includes('.txt')) return '📝';
    if (file.name.includes('.md')) return '📋';
    return '📄';
  };

  const filteredFiles = files
    .filter(file => file.name.toLowerCase().includes(searchTerm.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === 'name') return a.name.localeCompare(b.name);
      if (sortBy === 'size') return b.size - a.size;
      if (sortBy === 'modified') return new Date(b.modified) - new Date(a.modified);
      return 0;
    });

  return (
    <div style={{ 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
      background: 'linear-gradient(135deg, rgba(248, 246, 244, 0.95), rgba(253, 245, 230, 0.9))',
      fontFamily: 'Inter, sans-serif'
    }}>
      {/* Vault Header */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(13, 13, 13, 0.05), rgba(212, 175, 55, 0.1))',
        padding: '15px',
        borderBottom: '1px solid rgba(212, 175, 55, 0.3)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '1.4rem' }}>🏛️</span>
          <h3 style={{ margin: 0, color: '#D4AF37', fontFamily: 'Playfair Display, serif' }}>
            Noir File Vault
          </h3>
          <span style={{ 
            background: 'rgba(212, 175, 55, 0.2)', 
            padding: '4px 8px', 
            borderRadius: '4px', 
            fontSize: '0.75rem',
            color: '#0D0D0D'
          }}>
            {files.length} files
          </span>
        </div>
        
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <button
            onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
            style={{
              background: 'rgba(212, 175, 55, 0.2)',
              border: '1px solid rgba(212, 175, 55, 0.4)',
              borderRadius: '6px',
              padding: '6px 10px',
              color: '#0D0D0D',
              cursor: 'pointer',
              fontSize: '0.8rem'
            }}
          >
            {viewMode === 'grid' ? '📋 List' : '⊞ Grid'}
          </button>
          <button
            onClick={loadVaultFiles}
            style={{
              background: 'linear-gradient(135deg, #D4AF37, #FFD700)',
              border: 'none',
              borderRadius: '6px',
              padding: '6px 12px',
              color: '#0D0D0D',
              cursor: 'pointer',
              fontSize: '0.8rem',
              fontWeight: '500'
            }}
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Search and Sort Controls */}
      <div style={{
        padding: '10px 15px',
        background: 'rgba(212, 175, 55, 0.05)',
        borderBottom: '1px solid rgba(212, 175, 55, 0.2)',
        display: 'flex',
        gap: '15px',
        alignItems: 'center'
      }}>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.8rem', color: '#666' }}>🔍</span>
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search files..."
            style={{
              flex: 1,
              padding: '6px 10px',
              border: '1px solid rgba(212, 175, 55, 0.3)',
              borderRadius: '4px',
              fontSize: '0.8rem',
              background: 'rgba(255, 255, 255, 0.8)'
            }}
          />
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.8rem', color: '#666' }}>Sort:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            style={{
              padding: '4px 8px',
              border: '1px solid rgba(212, 175, 55, 0.3)',
              borderRadius: '4px',
              fontSize: '0.8rem',
              background: 'rgba(255, 255, 255, 0.8)'
            }}
          >
            <option value="modified">Modified</option>
            <option value="name">Name</option>
            <option value="size">Size</option>
          </select>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, display: 'flex' }}>
        {/* File List/Grid */}
        <div style={{ 
          flex: selectedFile ? 2 : 1, 
          padding: '15px',
          overflow: 'auto'
        }}>
          {filteredFiles.length === 0 ? (
            <div style={{ 
              textAlign: 'center', 
              padding: '60px 20px',
              color: '#666'
            }}>
              <div style={{ fontSize: '3rem', marginBottom: '20px' }}>📁</div>
              <h3 style={{ color: '#D4AF37', marginBottom: '10px' }}>Vault is Empty</h3>
              <p>No files found. Create documents in Text Atelier to populate your vault.</p>
            </div>
          ) : (
            <div style={{
              display: viewMode === 'grid' ? 'grid' : 'flex',
              gridTemplateColumns: viewMode === 'grid' ? 'repeat(auto-fill, minmax(180px, 1fr))' : 'none',
              flexDirection: viewMode === 'list' ? 'column' : 'row',
              gap: '12px'
            }}>
              {filteredFiles.map(file => (
                <div
                  key={file.id}
                  onClick={() => setSelectedFile(selectedFile?.id === file.id ? null : file)}
                  style={{
                    background: selectedFile?.id === file.id 
                      ? 'linear-gradient(135deg, rgba(212, 175, 55, 0.2), rgba(232, 180, 184, 0.2))'
                      : 'rgba(255, 255, 255, 0.8)',
                    border: selectedFile?.id === file.id
                      ? '2px solid rgba(212, 175, 55, 0.5)'
                      : '1px solid rgba(212, 175, 55, 0.2)',
                    borderRadius: '8px',
                    padding: viewMode === 'grid' ? '15px' : '12px',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    display: 'flex',
                    flexDirection: viewMode === 'grid' ? 'column' : 'row',
                    alignItems: viewMode === 'grid' ? 'center' : 'flex-start',
                    gap: viewMode === 'grid' ? '8px' : '12px'
                  }}
                  onMouseEnter={(e) => {
                    if (selectedFile?.id !== file.id) {
                      e.target.style.background = 'rgba(212, 175, 55, 0.1)';
                      e.target.style.transform = 'translateY(-2px)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (selectedFile?.id !== file.id) {
                      e.target.style.background = 'rgba(255, 255, 255, 0.8)';
                      e.target.style.transform = 'translateY(0)';
                    }
                  }}
                >
                  <div style={{ 
                    fontSize: viewMode === 'grid' ? '2rem' : '1.5rem',
                    marginBottom: viewMode === 'grid' ? '5px' : '0'
                  }}>
                    {getFileIcon(file)}
                  </div>
                  
                  <div style={{ 
                    textAlign: viewMode === 'grid' ? 'center' : 'left',
                    flex: viewMode === 'list' ? 1 : 'none'
                  }}>
                    <div style={{ 
                      fontWeight: '500', 
                      fontSize: '0.85rem',
                      color: '#0D0D0D',
                      marginBottom: '3px',
                      wordBreak: 'break-word'
                    }}>
                      {file.name}
                    </div>
                    <div style={{ 
                      fontSize: '0.7rem', 
                      color: '#666',
                      marginBottom: '2px'
                    }}>
                      {formatDate(file.modified)}
                    </div>
                    <div style={{ 
                      fontSize: '0.7rem', 
                      color: '#999'
                    }}>
                      {formatFileSize(file.size)}
                    </div>
                  </div>

                  {viewMode === 'list' && (
                    <div style={{ display: 'flex', gap: '5px' }}>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          downloadFile(file);
                        }}
                        style={{
                          background: 'rgba(212, 175, 55, 0.2)',
                          border: '1px solid rgba(212, 175, 55, 0.4)',
                          borderRadius: '4px',
                          padding: '4px 8px',
                          fontSize: '0.7rem',
                          cursor: 'pointer'
                        }}
                      >
                        📥
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteFile(file.id);
                        }}
                        style={{
                          background: 'rgba(220, 20, 60, 0.1)',
                          border: '1px solid rgba(220, 20, 60, 0.3)',
                          borderRadius: '4px',
                          padding: '4px 8px',
                          fontSize: '0.7rem',
                          cursor: 'pointer',
                          color: '#DC143C'
                        }}
                      >
                        🗑️
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* File Preview */}
        {selectedFile && (
          <div style={{
            flex: 1,
            borderLeft: '1px solid rgba(212, 175, 55, 0.3)',
            background: 'rgba(13, 13, 13, 0.02)',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{
              padding: '15px',
              borderBottom: '1px solid rgba(212, 175, 55, 0.2)',
              background: 'rgba(212, 175, 55, 0.1)'
            }}>
              <h4 style={{ margin: '0 0 8px 0', color: '#D4AF37', fontSize: '0.9rem' }}>
                {getFileIcon(selectedFile)} {selectedFile.name}
              </h4>
              <div style={{ fontSize: '0.75rem', color: '#666' }}>
                <div>Modified: {formatDate(selectedFile.modified)}</div>
                <div>Size: {formatFileSize(selectedFile.size)}</div>
                <div>Type: Document</div>
              </div>
              
              <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                <button
                  onClick={() => downloadFile(selectedFile)}
                  style={{
                    background: 'linear-gradient(135deg, #D4AF37, #FFD700)',
                    border: 'none',
                    borderRadius: '4px',
                    padding: '6px 10px',
                    fontSize: '0.75rem',
                    cursor: 'pointer',
                    color: '#0D0D0D'
                  }}
                >
                  📥 Download
                </button>
                <button
                  onClick={() => deleteFile(selectedFile.id)}
                  style={{
                    background: 'rgba(220, 20, 60, 0.1)',
                    border: '1px solid rgba(220, 20, 60, 0.3)',
                    borderRadius: '4px',
                    padding: '6px 10px',
                    fontSize: '0.75rem',
                    cursor: 'pointer',
                    color: '#DC143C'
                  }}
                >
                  🗑️ Delete
                </button>
              </div>
            </div>
            
            <div style={{ 
              flex: 1, 
              padding: '15px',
              overflow: 'auto',
              fontSize: '0.85rem',
              lineHeight: '1.6',
              fontFamily: 'Cormorant Garamond, serif',
              whiteSpace: 'pre-wrap',
              background: 'rgba(255, 255, 255, 0.5)'
            }}>
              {selectedFile.content || 'No content available.'}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default VaultApp;
