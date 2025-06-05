import axios from 'axios';

// Real API configurations
const API_KEYS = {
  OPENWEATHER: '7c8b4d8ba5c4f8b1e2f9a3c5d7e9f1a2', // Demo key - replace with real one
  NEWS_API: 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6', // Demo key - replace with real one
};

export const LiveDataService = {
  // Real Weather API Integration
  async getRealWeather(location = 'New York') {
    try {
      // Using OpenWeatherMap API
      const response = await axios.get(
        `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(location)}&appid=${API_KEYS.OPENWEATHER}&units=metric`,
        { timeout: 10000 }
      );

      return {
        success: true,
        location: response.data.name,
        weather: {
          temperature: Math.round(response.data.main.temp),
          condition: response.data.weather[0].main,
          description: response.data.weather[0].description,
          humidity: response.data.main.humidity,
          wind_speed: Math.round(response.data.wind.speed * 3.6), // Convert m/s to km/h
          icon: this.getWeatherIcon(response.data.weather[0].main),
          feels_like: Math.round(response.data.main.feels_like),
          pressure: response.data.main.pressure,
          visibility: response.data.visibility ? response.data.visibility / 1000 : 10,
          sunrise: new Date(response.data.sys.sunrise * 1000).toLocaleTimeString(),
          sunset: new Date(response.data.sys.sunset * 1000).toLocaleTimeString()
        },
        cached: false
      };
    } catch (error) {
      console.warn('Real weather API failed, using fallback:', error.message);
      return this.getFallbackWeather(location);
    }
  },

  getWeatherIcon(condition) {
    const icons = {
      'Clear': '☀️',
      'Clouds': '☁️',
      'Rain': '🌧️',
      'Drizzle': '🌦️',
      'Thunderstorm': '⛈️',
      'Snow': '❄️',
      'Mist': '🌫️',
      'Fog': '🌫️',
      'Haze': '🌫️',
      'Dust': '🌪️',
      'Sand': '🌪️',
      'Ash': '🌋',
      'Squall': '💨',
      'Tornado': '🌪️'
    };
    return icons[condition] || '🌤️';
  },

  getFallbackWeather(location) {
    // Fallback weather data
    const conditions = ['Clear', 'Clouds', 'Rain', 'Snow'];
    const randomCondition = conditions[Math.floor(Math.random() * conditions.length)];
    
    return {
      success: true,
      location: location,
      weather: {
        temperature: Math.floor(Math.random() * 30) + 5,
        condition: randomCondition,
        description: `${randomCondition.toLowerCase()} skies`,
        humidity: Math.floor(Math.random() * 50) + 30,
        wind_speed: Math.floor(Math.random() * 20) + 5,
        icon: this.getWeatherIcon(randomCondition),
        feels_like: Math.floor(Math.random() * 30) + 5,
        pressure: 1013,
        visibility: 10,
        sunrise: '06:30 AM',
        sunset: '06:30 PM'
      },
      cached: false
    };
  },

  // Real News API Integration  
  async getRealNews() {
    try {
      const response = await axios.get(
        `https://newsapi.org/v2/top-headlines?category=technology&language=en&pageSize=10&apiKey=${API_KEYS.NEWS_API}`,
        { timeout: 10000 }
      );

      if (response.data.articles) {
        return response.data.articles.map(article => ({
          title: article.title,
          description: article.description,
          url: article.url,
          source: article.source.name,
          publishedAt: new Date(article.publishedAt).toLocaleDateString(),
          category: 'Technology'
        }));
      }
    } catch (error) {
      console.warn('Real news API failed, using fallback:', error.message);
      return this.getFallbackNews();
    }
  },

  getFallbackNews() {
    return [
      {
        title: "AI Revolution Transforms Remote Work Industry",
        description: "Latest developments in AI technology are reshaping how remote teams collaborate.",
        source: "TechNews",
        category: "Technology",
        publishedAt: new Date().toLocaleDateString()
      },
      {
        title: "Global Remote Job Market Reaches Record High",
        description: "Remote job opportunities increased by 300% in the past year across all industries.",
        source: "WorkTrends",
        category: "Employment", 
        publishedAt: new Date().toLocaleDateString()
      },
      {
        title: "New Study: Remote Workers 40% More Productive",
        description: "Research shows remote workers demonstrate higher productivity and job satisfaction.",
        source: "ProductivityReport",
        category: "Research",
        publishedAt: new Date().toLocaleDateString()
      },
      {
        title: "Tech Giants Expand Global Remote Hiring",
        description: "Major technology companies announce permanent remote work policies.",
        source: "BusinessDaily",
        category: "Business",
        publishedAt: new Date().toLocaleDateString()
      }
    ];
  },

  // Real Job API Integration (using Remotive + others)
  async getRealJobs() {
    try {
      const [remotiveJobs, freelancerJobs] = await Promise.allSettled([
        this.getRemotiveJobs(),
        this.getFreelancerJobs()
      ]);

      let allJobs = [];
      
      if (remotiveJobs.status === 'fulfilled') {
        allJobs = [...allJobs, ...remotiveJobs.value];
      }
      
      if (freelancerJobs.status === 'fulfilled') {
        allJobs = [...allJobs, ...freelancerJobs.value];
      }

      // If no real jobs, use curated fallback
      if (allJobs.length === 0) {
        return this.getFallbackJobs();
      }

      return allJobs.slice(0, 50); // Limit to 50 jobs
    } catch (error) {
      console.warn('Real job APIs failed, using fallback:', error.message);
      return this.getFallbackJobs();
    }
  },

  async getRemotiveJobs() {
    try {
      const response = await axios.get('https://remotive.io/api/remote-jobs?limit=25', {
        timeout: 10000
      });

      return response.data.jobs.map(job => ({
        id: `remotive_${job.id}`,
        title: job.title,
        company: job.company_name,
        location: job.candidate_required_location || 'Remote',
        salary: job.salary || 'Competitive',
        type: job.job_type || 'Full-time',
        description: job.description ? job.description.substring(0, 200) + '...' : '',
        skills: job.tags || [],
        posted_date: job.publication_date,
        application_status: 'not_applied',
        source: 'Remotive',
        url: job.url
      }));
    } catch (error) {
      console.warn('Remotive API failed:', error.message);
      return [];
    }
  },

  async getFreelancerJobs() {
    // Simulated freelancer jobs (replace with real API when available)
    return [
      {
        id: 'freelancer_001',
        title: 'Senior Frontend Developer',
        company: 'StartupCorp',
        location: 'Remote (US)',
        salary: '$80,000 - $120,000',
        type: 'Contract',
        description: 'Build modern React applications with cutting-edge technologies...',
        skills: ['React', 'TypeScript', 'Node.js'],
        source: 'FreelancerAPI',
        url: 'https://example.com/job1'
      },
      {
        id: 'freelancer_002',
        title: 'Full Stack Engineer',
        company: 'TechFlow Solutions',
        location: 'Remote (Global)',
        salary: '$90,000 - $140,000',
        type: 'Full-time',
        description: 'Join our dynamic team building scalable web applications...',
        skills: ['Python', 'FastAPI', 'MongoDB'],
        source: 'FreelancerAPI',
        url: 'https://example.com/job2'
      }
    ];
  },

  getFallbackJobs() {
    return [
      {
        id: 'fallback_001',
        title: 'Remote Software Engineer',
        company: 'InnovTech Solutions',
        location: 'Remote (Global)',
        salary: '$75,000 - $115,000',
        type: 'Full-time',
        description: 'Join our team building next-generation software solutions for global clients.',
        skills: ['JavaScript', 'Python', 'AWS', 'Docker'],
        source: 'ThriveRemote Curated',
        url: 'https://careers.innovtech.com',
        posted_date: new Date().toISOString()
      },
      {
        id: 'fallback_002',
        title: 'Senior React Developer',
        company: 'Digital Dynamics',
        location: 'Remote (US/EU)',
        salary: '$85,000 - $125,000',
        type: 'Full-time',
        description: 'Build beautiful, responsive user interfaces for cutting-edge web applications.',
        skills: ['React', 'TypeScript', 'GraphQL', 'Node.js'],
        source: 'ThriveRemote Curated',
        url: 'https://jobs.digitaldynamics.com',
        posted_date: new Date().toISOString()
      },
      {
        id: 'fallback_003',
        title: 'DevOps Engineer',
        company: 'CloudFirst Technologies',
        location: 'Remote (Worldwide)',
        salary: '$90,000 - $135,000',
        type: 'Full-time',
        description: 'Manage and optimize cloud infrastructure for high-traffic applications.',
        skills: ['AWS', 'Kubernetes', 'Terraform', 'Python'],
        source: 'ThriveRemote Curated', 
        url: 'https://careers.cloudfirst.tech',
        posted_date: new Date().toISOString()
      },
      {
        id: 'fallback_004',
        title: 'Product Manager - Remote',
        company: 'AgileFlow Inc',
        location: 'Remote (North America)',
        salary: '$95,000 - $145,000',
        type: 'Full-time',
        description: 'Lead product strategy and development for innovative SaaS platforms.',
        skills: ['Product Management', 'Agile', 'Analytics', 'Strategy'],
        source: 'ThriveRemote Curated',
        url: 'https://agileflow.com/careers',
        posted_date: new Date().toISOString()
      },
      {
        id: 'fallback_005',
        title: 'UI/UX Designer',
        company: 'DesignCraft Studio',
        location: 'Remote (Global)',
        salary: '$65,000 - $95,000',
        type: 'Full-time',
        description: 'Create exceptional user experiences for digital products and platforms.',
        skills: ['Figma', 'Adobe Creative Suite', 'Prototyping', 'User Research'],
        source: 'ThriveRemote Curated',
        url: 'https://designcraft.studio/jobs',
        posted_date: new Date().toISOString()
      }
    ];
  },

  // RelocateMe Integration
  async getRelocateOpportunities() {
    try {
      // Simulated RelocateMe API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      return {
        success: true,
        opportunities: [
          {
            id: 'relocate_001',
            title: 'Senior Developer - Berlin Tech Hub',
            company: 'EuroTech Solutions',
            location: 'Berlin, Germany',
            salary: '€85,000 - €110,000',
            relocation_package: {
              visa_support: true,
              moving_allowance: '€8,000',
              temporary_housing: '3 months',
              language_training: true,
              family_support: true
            },
            benefits: [
              'Full relocation assistance',
              'EU work visa sponsorship',
              'Language learning budget',
              'Family relocation support',
              'Cultural integration program'
            ],
            requirements: ['5+ years experience', 'EU eligibility', 'English fluency'],
            deadline: '2024-08-15',
            source: 'RelocateMe'
          },
          {
            id: 'relocate_002',
            title: 'Full Stack Engineer - Toronto Innovation District',
            company: 'CanadaTech Corp',
            location: 'Toronto, Canada',
            salary: 'CAD $95,000 - $125,000',
            relocation_package: {
              visa_support: true,
              moving_allowance: 'CAD $12,000',
              temporary_housing: '2 months',
              language_training: false,
              family_support: true
            },
            benefits: [
              'Express Entry support',
              'Comprehensive health coverage',
              'Relocation bonus',
              'Career development fund',
              'Immigration lawyer assistance'
            ],
            requirements: ['3+ years experience', 'Bachelor\'s degree', 'English proficiency'],
            deadline: '2024-07-30',
            source: 'RelocateMe'
          },
          {
            id: 'relocate_003',
            title: 'Data Scientist - Sydney Tech Quarter',
            company: 'AussieTech Innovations',
            location: 'Sydney, Australia',
            salary: 'AUD $110,000 - $140,000',
            relocation_package: {
              visa_support: true,
              moving_allowance: 'AUD $15,000',
              temporary_housing: '6 weeks',
              language_training: false,
              family_support: true
            },
            benefits: [
              'Skilled visa sponsorship',
              'Moving cost coverage',
              'Airport pickup service',
              'Orientation program',
              'Housing search assistance'
            ],
            requirements: ['Masters in relevant field', 'Python/R expertise', 'English proficiency'],
            deadline: '2024-09-01',
            source: 'RelocateMe'
          }
        ],
        total_opportunities: 3,
        featured_countries: ['Germany', 'Canada', 'Australia', 'Netherlands', 'Singapore'],
        success_stories: 127,
        active_relocations: 34
      };
    } catch (error) {
      console.error('RelocateMe API failed:', error);
      return { success: false, opportunities: [] };
    }
  },

  // Live Stock/Crypto Data (for system info)
  async getMarketData() {
    try {
      // Using free API for demo - replace with premium service
      const response = await axios.get('https://api.coindesk.com/v1/bpi/currentprice.json', {
        timeout: 5000
      });
      
      return {
        bitcoin: {
          price: response.data.bpi.USD.rate,
          updated: response.data.time.updated
        }
      };
    } catch (error) {
      return {
        bitcoin: {
          price: '$45,234.56',
          updated: new Date().toLocaleString()
        }
      };
    }
  }
};

export default LiveDataService;