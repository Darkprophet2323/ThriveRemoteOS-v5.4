import React, { useState, useEffect } from 'react';
import axios from 'axios';

const NotepadApp = () => {
  const [content, setContent] = useState('');
  const [fileName, setFileName] = useState('');
  const [savedFiles, setSavedFiles] = useState([]);
  const [currentFile, setCurrentFile] = useState(null);
  const [saveStatus, setSaveStatus] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

  useEffect(() => {
    // Load saved files from localStorage (simulating vault)
    const saved = localStorage.getItem('thriveOS_vault_files');
    if (saved) {
      setSavedFiles(JSON.parse(saved));
    }
  }, []);

  const saveToVault = () => {
    if (!fileName.trim()) {
      setShowSaveDialog(true);
      return;
    }

    const file = {
      id: Date.now().toString(),
      name: fileName,
      content: content,
      created: new Date().toISOString(),
      modified: new Date().toISOString(),
      type: 'document',
      size: content.length
    };

    const updatedFiles = [...savedFiles.filter(f => f.name !== fileName), file];
    setSavedFiles(updatedFiles);
    localStorage.setItem('thriveOS_vault_files', JSON.stringify(updatedFiles));
    
    setCurrentFile(file);
    setSaveStatus(`✅ Saved to Vault: ${fileName}`);
    setTimeout(() => setSaveStatus(''), 3000);
    setShowSaveDialog(false);
  };

  const loadFromVault = (file) => {
    setContent(file.content);
    setFileName(file.name);
    setCurrentFile(file);
    setSaveStatus(`📂 Loaded: ${file.name}`);
    setTimeout(() => setSaveStatus(''), 3000);
  };

  const newDocument = () => {
    setContent('');
    setFileName('');
    setCurrentFile(null);
    setSaveStatus('📝 New Document');
    setTimeout(() => setSaveStatus(''), 2000);
  };

  const deleteFromVault = (fileId) => {
    const updatedFiles = savedFiles.filter(f => f.id !== fileId);
    setSavedFiles(updatedFiles);
    localStorage.setItem('thriveOS_vault_files', JSON.stringify(updatedFiles));
    setSaveStatus('🗑️ File deleted from Vault');
    setTimeout(() => setSaveStatus(''), 3000);
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' bytes';
    return Math.round(bytes / 1024) + ' KB';
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString() + ' ' + 
           new Date(dateString).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  };

  return (
    <div style={{ 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
      background: 'linear-gradient(135deg, rgba(248, 246, 244, 0.95), rgba(253, 245, 230, 0.9))',
      fontFamily: 'Inter, sans-serif'
    }}>
      {/* Notepad Header */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(13, 13, 13, 0.05), rgba(212, 175, 55, 0.1))',
        padding: '12px 15px',
        borderBottom: '1px solid rgba(212, 175, 55, 0.3)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '1.2rem' }}>📝</span>
          <h3 style={{ margin: 0, color: '#D4AF37', fontFamily: 'Playfair Display, serif' }}>
            Noir Text Atelier
          </h3>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={newDocument}
            style={{
              background: 'rgba(212, 175, 55, 0.2)',
              border: '1px solid rgba(212, 175, 55, 0.4)',
              borderRadius: '6px',
              padding: '6px 12px',
              color: '#0D0D0D',
              cursor: 'pointer',
              fontSize: '0.8rem',
              fontWeight: '500'
            }}
          >
            📄 New
          </button>
          <button
            onClick={saveToVault}
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
            💾 Save to Vault
          </button>
        </div>
      </div>

      {/* File Name Input & Status */}
      <div style={{ 
        padding: '8px 15px', 
        background: 'rgba(212, 175, 55, 0.05)',
        borderBottom: '1px solid rgba(212, 175, 55, 0.2)',
        display: 'flex',
        alignItems: 'center',
        gap: '15px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <label style={{ fontSize: '0.8rem', color: '#666', fontWeight: '500' }}>File:</label>
          <input
            type="text"
            value={fileName}
            onChange={(e) => setFileName(e.target.value)}
            placeholder="Enter filename..."
            style={{
              padding: '4px 8px',
              border: '1px solid rgba(212, 175, 55, 0.3)',
              borderRadius: '4px',
              fontSize: '0.8rem',
              width: '200px',
              background: 'rgba(255, 255, 255, 0.8)'
            }}
          />
        </div>
        {saveStatus && (
          <div style={{ 
            fontSize: '0.8rem', 
            color: saveStatus.includes('✅') ? '#32CD32' : '#D4AF37',
            fontWeight: '500'
          }}>
            {saveStatus}
          </div>
        )}
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex' }}>
        {/* Text Editor */}
        <div style={{ flex: 2, display: 'flex', flexDirection: 'column' }}>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Start writing your document here...

This is the Noir Text Atelier - your sophisticated writing environment.

Features:
• Automatic saving to Vault
• Document management
• Elegant noir-gold theme
• Professional typography

Press Save to Vault to store your work securely."
            style={{
              flex: 1,
              padding: '20px',
              border: 'none',
              outline: 'none',
              resize: 'none',
              fontSize: '0.9rem',
              lineHeight: '1.6',
              fontFamily: 'Cormorant Garamond, serif',
              background: 'transparent',
              color: '#0D0D0D'
            }}
          />
          
          {/* Document Stats */}
          <div style={{
            padding: '8px 20px',
            background: 'rgba(212, 175, 55, 0.05)',
            borderTop: '1px solid rgba(212, 175, 55, 0.2)',
            fontSize: '0.75rem',
            color: '#666',
            display: 'flex',
            justifyContent: 'space-between'
          }}>
            <span>Characters: {content.length}</span>
            <span>Words: {content.split(/\s+/).filter(word => word.length > 0).length}</span>
            <span>Lines: {content.split('\n').length}</span>
          </div>
        </div>

        {/* Vault Sidebar */}
        <div style={{ 
          width: '200px', 
          borderLeft: '1px solid rgba(212, 175, 55, 0.3)',
          background: 'rgba(13, 13, 13, 0.02)',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <div style={{
            padding: '10px',
            borderBottom: '1px solid rgba(212, 175, 55, 0.2)',
            background: 'rgba(212, 175, 55, 0.1)',
            fontWeight: '600',
            fontSize: '0.8rem',
            color: '#D4AF37'
          }}>
            📁 Vault Files ({savedFiles.length})
          </div>
          
          <div style={{ flex: 1, overflow: 'auto' }}>
            {savedFiles.length === 0 ? (
              <div style={{ 
                padding: '20px', 
                textAlign: 'center', 
                color: '#666',
                fontSize: '0.8rem'
              }}>
                No files in vault yet.
                <br /><br />
                Save a document to get started.
              </div>
            ) : (
              savedFiles.map(file => (
                <div key={file.id} style={{
                  padding: '8px',
                  borderBottom: '1px solid rgba(212, 175, 55, 0.1)',
                  cursor: 'pointer',
                  transition: 'background 0.2s ease',
                  fontSize: '0.75rem'
                }}
                onMouseEnter={(e) => e.target.style.background = 'rgba(212, 175, 55, 0.1)'}
                onMouseLeave={(e) => e.target.style.background = 'transparent'}>
                  <div 
                    onClick={() => loadFromVault(file)}
                    style={{ marginBottom: '4px' }}
                  >
                    <div style={{ fontWeight: '500', color: '#0D0D0D', marginBottom: '2px' }}>
                      📄 {file.name}
                    </div>
                    <div style={{ color: '#666', fontSize: '0.7rem' }}>
                      {formatDate(file.modified)}
                    </div>
                    <div style={{ color: '#999', fontSize: '0.7rem' }}>
                      {formatFileSize(file.size)}
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteFromVault(file.id);
                    }}
                    style={{
                      background: 'rgba(220, 20, 60, 0.1)',
                      border: '1px solid rgba(220, 20, 60, 0.3)',
                      borderRadius: '3px',
                      padding: '2px 6px',
                      fontSize: '0.6rem',
                      color: '#DC143C',
                      cursor: 'pointer',
                      marginTop: '4px'
                    }}
                  >
                    🗑️ Delete
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Save Dialog */}
      {showSaveDialog && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: 'linear-gradient(135deg, rgba(248, 246, 244, 0.98), rgba(253, 245, 230, 0.95))',
            padding: '25px',
            borderRadius: '12px',
            border: '2px solid rgba(212, 175, 55, 0.3)',
            backdropFilter: 'blur(20px)',
            minWidth: '300px'
          }}>
            <h4 style={{ margin: '0 0 15px 0', color: '#D4AF37' }}>💾 Save Document</h4>
            <p style={{ margin: '0 0 15px 0', fontSize: '0.9rem', color: '#666' }}>
              Enter a filename for your document:
            </p>
            <input
              type="text"
              value={fileName}
              onChange={(e) => setFileName(e.target.value)}
              placeholder="my-document.txt"
              style={{
                width: '100%',
                padding: '8px 12px',
                border: '1px solid rgba(212, 175, 55, 0.3)',
                borderRadius: '6px',
                fontSize: '0.9rem',
                marginBottom: '15px',
                background: 'rgba(255, 255, 255, 0.8)'
              }}
              autoFocus
              onKeyPress={(e) => e.key === 'Enter' && saveToVault()}
            />
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => setShowSaveDialog(false)}
                style={{
                  background: 'rgba(128, 128, 128, 0.2)',
                  border: '1px solid rgba(128, 128, 128, 0.3)',
                  borderRadius: '6px',
                  padding: '8px 16px',
                  cursor: 'pointer',
                  fontSize: '0.8rem'
                }}
              >
                Cancel
              </button>
              <button
                onClick={saveToVault}
                disabled={!fileName.trim()}
                style={{
                  background: fileName.trim() ? 'linear-gradient(135deg, #D4AF37, #FFD700)' : 'rgba(128, 128, 128, 0.3)',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '8px 16px',
                  color: '#0D0D0D',
                  cursor: fileName.trim() ? 'pointer' : 'not-allowed',
                  fontSize: '0.8rem',
                  fontWeight: '500'
                }}
              >
                💾 Save to Vault
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NotepadApp;
