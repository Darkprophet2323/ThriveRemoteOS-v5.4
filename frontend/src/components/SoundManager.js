import React, { createContext, useContext, useRef, useEffect } from 'react';

const SoundContext = createContext();

export const useSounds = () => {
  const context = useContext(SoundContext);
  if (!context) {
    throw new Error('useSounds must be used within a SoundProvider');
  }
  return context;
};

export const SoundProvider = ({ children }) => {
  const soundsRef = useRef({});

  useEffect(() => {
    // Preload luxury sound effects
    soundsRef.current = {
      windowOpen: new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhByiN0e/BaiMFLYXQ8tiJNggSaLrl5Z9OEAwRpu7pnF8WCzqR3fC2eysJKHfH8N6TPQUWXsrq66hUEwlZq+XjwWknBR6KxfL0VxYPT6nj6bFJGRUkHKQdKjON3fCdTBYOU7Ti57CKH5QDzA0CwMJNDATyH5QP0QIDwMNJBgTyH5QHrONJBgT2J5UFrtlnDw=='), // Subtle chime
      windowClose: new Audio('data:audio/wav;base64,UklGRuACAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YZwCAAC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEC4uLi4uLi4QEBAuLi4uLi4uEBAQLi4uLi4uLhAQEA='), // Soft close
      click: new Audio('data:audio/wav;base64,UklGRjQDAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YRAD'), // Click sound
      hover: new Audio('data:audio/wav;base64,UklGRjQDAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YRAD'), // Hover sound
      success: new Audio('data:audio/wav;base64,UklGRjQDAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YRAD') // Success sound
    };

    // Set volume for all sounds to 30% for subtle luxury experience
    Object.values(soundsRef.current).forEach(sound => {
      sound.volume = 0.3;
    });
  }, []);

  const playSound = (soundName) => {
    try {
      const sound = soundsRef.current[soundName];
      if (sound) {
        sound.currentTime = 0; // Reset to beginning
        sound.play().catch(e => {
          // Ignore autoplay restrictions
          console.log('Sound autoplay prevented:', e);
        });
      }
    } catch (error) {
      console.log('Sound play error:', error);
    }
  };

  const sounds = {
    playWindowOpen: () => playSound('windowOpen'),
    playWindowClose: () => playSound('windowClose'),
    playClick: () => playSound('click'),
    playHover: () => playSound('hover'),
    playSuccess: () => playSound('success')
  };

  return (
    <SoundContext.Provider value={sounds}>
      {children}
    </SoundContext.Provider>
  );
};
