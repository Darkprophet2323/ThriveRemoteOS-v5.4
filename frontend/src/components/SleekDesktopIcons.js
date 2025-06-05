import React from 'react';

const SleekDesktopIcons = ({ onIconClick, sounds }) => {
  const icons = [
    {
      id: 'ai-portal',
      icon: 'psychology',
      label: 'ai portal',
      component: 'WaitressJobPortal',
      size: { width: 500, height: 400 }
    },
    {
      id: 'relocate',
      icon: 'flight_takeoff',
      label: 'relocate',
      component: 'RelocateMeIntegration',
      size: { width: 550, height: 450 }
    },
    {
      id: 'downloads',
      icon: 'cloud_download',
      label: 'downloads',
      component: 'ProfessionalDownloadManager',
      size: { width: 600, height: 500 }
    },
    {
      id: 'weather',
      icon: 'wb_sunny',
      label: 'weather',
      component: 'ProfessionalWeatherWidget',
      size: { width: 500, height: 450 }
    },
    {
      id: 'vault',
      icon: 'folder_open',
      label: 'vault',
      component: 'VaultApp',
      size: { width: 450, height: 350 }
    },
    {
      id: 'text-editor',
      icon: 'edit_note',
      label: 'text editor',
      component: 'NotepadApp',
      size: { width: 500, height: 400 }
    },
    {
      id: 'terminal',
      icon: 'terminal',
      label: 'terminal',
      component: 'QuantumTerminal',
      size: { width: 400, height: 300 }
    },
    {
      id: 'settings',
      icon: 'tune',
      label: 'settings',
      component: 'ProfessionalSettings',
      size: { width: 600, height: 500 }
    }
  ];

  return (
    <div className="desktop-icons">
      {icons.map((iconData) => (
        <div
          key={iconData.id}
          className="desktop-icon"
          onClick={() => {
            if (sounds?.playClick) sounds.playClick();
            onIconClick(
              iconData.label,
              iconData.icon,
              iconData.component,
              iconData.size.width,
              iconData.size.height
            );
          }}
          onMouseEnter={() => {
            if (sounds?.playHover) sounds.playHover();
          }}
        >
          <span className="icon material-icons-outlined">
            {iconData.icon}
          </span>
          <span className="label">{iconData.label}</span>
        </div>
      ))}
    </div>
  );
};

export default SleekDesktopIcons;