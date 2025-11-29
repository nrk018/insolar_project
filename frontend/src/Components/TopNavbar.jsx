import { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from '../utils/axiosConfig';
import { stopCamera } from '../utils/cameraUtils';

const TopNavbar = ({ isAdmin = false }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [name, setName] = useState('');
  const [profilePhoto, setProfilePhoto] = useState([]);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('/api/user/name');
        setName(response.data.name);
        const photoResponse = await axios.get('/api/user/profilePhoto');
        setProfilePhoto(photoResponse.data.profilePhoto);
      } catch (error) {
        console.error('Error fetching name or profile photo:', error);
      }
    };
    fetchData();
  }, []);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };
    if (isDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isDropdownOpen]);

  const handleLogout = async () => {
    try {
      // Stop camera before logging out - use Promise.race to ensure it doesn't hang
      const stopPromise = stopCamera();
      const timeoutPromise = new Promise(resolve => setTimeout(resolve, 2000));
      await Promise.race([stopPromise, timeoutPromise]);
      
      // Then proceed with logout
      await axios.post('/logout');
      navigate('/');
    } catch (error) {
      console.error('Error logging out:', error);
      // Navigate anyway even if camera stop fails
      navigate('/');
    }
  };

  // No need for extractNameFromImage - profilePhoto now contains full URL

  const navItems = isAdmin
    ? [
        { path: '/dashboard', label: 'Dashboard' },
        { path: '/history', label: 'History' },
        { path: '/employees', label: 'Employees' },
        { path: '/add-employee', label: 'Add Employee' },
        { path: '/camera', label: 'Camera Monitor' },
        { path: '/analyze', label: 'Image Analysis' },
        { path: '/settings', label: 'Settings' },
      ]
    : [
        { path: '/dashboard', label: 'Dashboard' },
        { path: '/history', label: 'History' },
        { path: '/settings', label: 'Settings' },
      ];

  return (
    <nav className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between px-4">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <img 
              src="/logo.png" 
              alt="Company Logo" 
              className="h-10 w-auto object-contain"
            />
          </div>
          <div className="hidden md:flex items-center gap-1">
            {navItems.map((item) => (
              <button
                key={item.path}
                onClick={() => navigate(item.path)}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  location.pathname === item.path
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-4">
          <span className="text-sm font-medium hidden sm:block">{name}</span>
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="flex items-center gap-2 rounded-full focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            >
              {profilePhoto.length > 0 ? (
                <img
                  src={profilePhoto[0]}
                  alt="Profile"
                  className="w-8 h-8 rounded-full border-2 border-border object-cover"
                />
              ) : (
                <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center">
                  <span className="text-xs font-medium">{name.charAt(0).toUpperCase()}</span>
                </div>
              )}
            </button>
            {isDropdownOpen && (
              <div className="absolute right-0 mt-2 w-48 rounded-md border bg-popover shadow-md z-50">
                <div className="p-1">
                  <button
                    onClick={handleLogout}
                    className="w-full text-left px-3 py-2 text-sm rounded-sm hover:bg-accent hover:text-accent-foreground transition-colors"
                  >
                    Logout
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default TopNavbar;

