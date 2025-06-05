import React, { useState, useEffect } from 'react';

const QuantumTerminal = () => {
  const [commands, setCommands] = useState([]);
  const [currentCommand, setCurrentCommand] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  const terminalCommands = {
    help: {
      description: 'Show available commands',
      output: `Available Commands:
• help - Show this help message
• status - Display system status
• weather [location] - Get weather information
• jobs - List available remote jobs
• downloads - Show download manager
• clear - Clear terminal screen
• whoami - Display current user info
• neofetch - System information
• ping [host] - Test network connectivity`
    },
    status: {
      description: 'Display system status',
      output: `System Status:
├─ OS: ThriveRemoteOS v5.3
├─ Kernel: Professional Edition
├─ Memory: 85% available
├─ CPU: 12% usage
├─ Network: Connected
├─ Services: All operational
└─ Uptime: 2h 34m`
    },
    whoami: {
      description: 'Display current user',
      output: 'user@thriveremote-pro'
    },
    neofetch: {
      description: 'System information',
      output: `        ██████████████        user@thriveremote-pro
      ██                ██      ─────────────────────
    ██                    ██    OS: ThriveRemoteOS v5.3
  ██                        ██  Kernel: Professional Edition
██            ████            ██ Uptime: 2 hours, 34 mins
██          ██    ██          ██ Packages: 847 (npm), 142 (pip)
██          ██    ██          ██ Shell: bash 5.1.4
██            ████            ██ Resolution: 1920x1080
  ██                        ██  Theme: Sleek Professional Dark
    ██                    ██    CPU: Intel i7-12700K (16 cores)
      ██                ██      Memory: 2.8GB / 16GB (17%)
        ██████████████        
                              
Professional Remote Work Platform`
    },
    clear: {
      description: 'Clear terminal screen',
      output: null
    }
  };

  useEffect(() => {
    // Welcome message
    setCommands([
      {
        type: 'output',
        content: `Welcome to ThriveRemoteOS Professional Terminal v5.3
Type 'help' for available commands.`,
        timestamp: new Date().toLocaleTimeString()
      }
    ]);
  }, []);

  const executeCommand = async (cmd) => {
    const trimmedCmd = cmd.trim().toLowerCase();
    const [command, ...args] = trimmedCmd.split(' ');

    // Add command to history
    setCommands(prev => [...prev, {
      type: 'command',
      content: `$ ${cmd}`,
      timestamp: new Date().toLocaleTimeString()
    }]);

    setIsProcessing(true);

    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 300));

    let output = '';

    switch (command) {
      case 'help':
        output = terminalCommands.help.output;
        break;
      case 'status':
        output = terminalCommands.status.output;
        break;
      case 'whoami':
        output = terminalCommands.whoami.output;
        break;
      case 'neofetch':
        output = terminalCommands.neofetch.output;
        break;
      case 'weather':
        const location = args.join(' ') || 'New York';
        output = `Weather for ${location}:
🌤️ Partly Cloudy, 22°C
💧 Humidity: 65%
💨 Wind: 15 km/h NE
📊 Pressure: 1013 hPa`;
        break;
      case 'jobs':
        output = `Available Remote Jobs:
┌─ Senior React Developer - $95k-125k - Remote Global
├─ DevOps Engineer - $90k-130k - Remote US/EU  
├─ Product Manager - $100k-140k - Remote Worldwide
├─ UX Designer - $70k-95k - Remote North America
└─ Data Scientist - $85k-115k - Remote Global

Use 'jobs --details' for more information.`;
        break;
      case 'downloads':
        output = `Download Manager Status:
┌─ Active Downloads: 2
├─ Completed: 15
├─ Failed: 0
└─ Total Size: 2.3 GB

Use GUI Download Manager for detailed view.`;
        break;
      case 'ping':
        const host = args[0] || 'google.com';
        output = `PING ${host} (172.217.164.110): 56 data bytes
64 bytes from 172.217.164.110: icmp_seq=0 ttl=55 time=12.4 ms
64 bytes from 172.217.164.110: icmp_seq=1 ttl=55 time=11.8 ms
64 bytes from 172.217.164.110: icmp_seq=2 ttl=55 time=13.2 ms

--- ${host} ping statistics ---
3 packets transmitted, 3 received, 0% packet loss
round-trip min/avg/max/stddev = 11.8/12.5/13.2/0.6 ms`;
        break;
      case 'clear':
        setCommands([]);
        setIsProcessing(false);
        return;
      case '':
        setIsProcessing(false);
        return;
      default:
        output = `Command not found: ${command}
Type 'help' for available commands.`;
    }

    // Add output to terminal
    setCommands(prev => [...prev, {
      type: 'output',
      content: output,
      timestamp: new Date().toLocaleTimeString()
    }]);

    setIsProcessing(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !isProcessing) {
      executeCommand(currentCommand);
      setCurrentCommand('');
    }
  };

  return (
    <div className="app-content" style={{ 
      background: '#0f0f0f', 
      color: '#00ff00', 
      fontFamily: 'SF Mono, Monaco, Consolas, monospace',
      padding: '16px',
      height: '100%',
      overflow: 'hidden'
    }}>
      {/* Terminal Header */}
      <div style={{
        borderBottom: '1px solid #333',
        paddingBottom: '8px',
        marginBottom: '16px',
        color: '#ccc',
        fontSize: '0.875rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            background: '#ff5f57'
          }}></div>
          <div style={{
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            background: '#ffbd2e'
          }}></div>
          <div style={{
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            background: '#28ca42'
          }}></div>
          <span style={{ marginLeft: '12px', fontWeight: '500' }}>
            ThriveRemoteOS Terminal - Professional Edition
          </span>
        </div>
      </div>

      {/* Terminal Output */}
      <div style={{
        height: 'calc(100% - 80px)',
        overflowY: 'auto',
        marginBottom: '16px',
        lineHeight: '1.4'
      }}>
        {commands.map((cmd, index) => (
          <div key={index} style={{ marginBottom: '8px' }}>
            {cmd.type === 'command' ? (
              <div style={{ color: '#00ff00', fontWeight: '600' }}>
                {cmd.content}
              </div>
            ) : (
              <pre style={{
                color: '#ccc',
                margin: 0,
                fontFamily: 'inherit',
                whiteSpace: 'pre-wrap',
                wordWrap: 'break-word'
              }}>
                {cmd.content}
              </pre>
            )}
          </div>
        ))}
        
        {/* Current Input Line */}
        <div style={{ display: 'flex', alignItems: 'center', color: '#00ff00' }}>
          <span style={{ marginRight: '8px', fontWeight: '600' }}>$</span>
          <input
            type="text"
            value={currentCommand}
            onChange={(e) => setCurrentCommand(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isProcessing}
            style={{
              background: 'transparent',
              border: 'none',
              outline: 'none',
              color: '#00ff00',
              fontFamily: 'inherit',
              fontSize: 'inherit',
              flex: 1,
              caretColor: '#00ff00'
            }}
            placeholder={isProcessing ? "Processing..." : "Enter command..."}
            autoFocus
          />
          {isProcessing && (
            <span style={{ marginLeft: '8px', color: '#ff6b35' }}>⚡</span>
          )}
        </div>
      </div>

      {/* Terminal Footer */}
      <div style={{
        borderTop: '1px solid #333',
        paddingTop: '8px',
        fontSize: '0.75rem',
        color: '#666',
        display: 'flex',
        justifyContent: 'space-between'
      }}>
        <span>Press Enter to execute commands</span>
        <span>{new Date().toLocaleTimeString()}</span>
      </div>
    </div>
  );
};

export default QuantumTerminal;