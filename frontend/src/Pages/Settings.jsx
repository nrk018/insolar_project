import { useState, useEffect } from "react";
import axios from '../utils/axiosConfig';
import { useNavigate } from 'react-router-dom';

const Settings = () => {
  const [name, setName] = useState("");
  const [empID, setEmpID] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [department, setDepartment] = useState("");
  const [designation, setDesignation] = useState("");
  const [profilePic, setProfilePic] = useState(null);
  const [availableImages, setAvailableImages] = useState([]);
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('/api/user');
        const { name, employee_id, email, phone, department, designation, profilePhoto } = response.data;
        setName(name);
        setEmpID(employee_id);
        setEmail(email);
        setPhone(phone);
        setDepartment(department);
        setDesignation(designation);
        
        // Fetch user's uploaded images
        try {
          const imagesResponse = await axios.get('/api/user/images');
          const images = imagesResponse.data.images || [];
          setAvailableImages(images);
          
          // Set profile photo: use profilePhoto from /api/user if available, otherwise use first image from available images
          if (profilePhoto && profilePhoto.length > 0) {
            setProfilePic(profilePhoto[0]);
            console.log('[SETTINGS] Using profile photo from /api/user:', profilePhoto[0]);
          } else if (images.length > 0) {
            setProfilePic(images[0]);
            console.log('[SETTINGS] Using first image from available images:', images[0]);
          } else {
            setProfilePic(null);
            console.log('[SETTINGS] No profile photo available');
          }
        } catch (error) {
          console.error("Error fetching user images:", error);
          // Still try to use profilePhoto from /api/user if available
          if (profilePhoto && profilePhoto.length > 0) {
            setProfilePic(profilePhoto[0]);
          }
        }
      } catch (error) {
        console.error("Error fetching user data:", error);
      }
    };
    fetchData();
  }, []);

  const handleSaveChanges = () => {
    setIsLoading(true);
    const updateData = { name, phone };

    if (oldPassword && newPassword) {
      updateData.oldPassword = oldPassword;
      updateData.newPassword = newPassword;
    }

    axios.patch("/api/users/update", updateData)
      .then(response => {
        alert("Changes saved successfully!");
        setOldPassword("");
        setNewPassword("");
      })
      .catch(error => {
        console.error("Error saving changes:", error);
        alert(error.response?.data?.error || "Failed to update user details.");
      })
      .finally(() => setIsLoading(false));
  };

  const handleImageSelect = (imageUrl) => {
    setProfilePic(imageUrl);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">Manage your account settings and preferences.</p>
      </div>

      <div className="grid gap-6">
        <div className="rounded-lg border bg-card p-6 shadow-sm">
          <div className="flex items-center gap-6 mb-6">
            {profilePic ? (
              <img
                src={profilePic}
                alt="Profile"
                className="w-20 h-20 rounded-full border-2 border-border object-cover"
                onError={(e) => {
                  console.error('[SETTINGS] Failed to load profile image:', profilePic);
                  e.target.style.display = 'none';
                  // Show fallback
                  const parent = e.target.parentElement;
                  if (parent) {
                    const fallback = document.createElement('div');
                    fallback.className = 'w-20 h-20 rounded-full bg-muted flex items-center justify-center';
                    fallback.innerHTML = `<span class="text-2xl font-medium">${name.charAt(0).toUpperCase()}</span>`;
                    parent.appendChild(fallback);
                  }
                }}
                onLoad={() => {
                  console.log('[SETTINGS] Successfully loaded profile image:', profilePic);
                }}
              />
            ) : (
              <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center">
                <span className="text-2xl font-medium">{name.charAt(0).toUpperCase()}</span>
              </div>
            )}
            <div>
              <h3 className="text-2xl font-semibold">{name}</h3>
              <p className="text-sm text-muted-foreground">{designation}</p>
              <p className="text-sm text-muted-foreground">{empID}</p>
            </div>
          </div>
          
          {/* Available Images Selection */}
          {availableImages.length > 0 && (
            <div className="mt-4">
              <h4 className="text-sm font-medium mb-2">Select Profile Picture</h4>
              <div className="flex gap-2 flex-wrap">
                {availableImages.map((imageUrl, index) => (
                  <button
                    key={index}
                    onClick={() => handleImageSelect(imageUrl)}
                    className={`relative w-16 h-16 rounded-lg border-2 overflow-hidden transition-all ${
                      profilePic === imageUrl 
                        ? 'border-primary ring-2 ring-primary ring-offset-2' 
                        : 'border-border hover:border-primary/50'
                    }`}
                  >
                    <img
                      src={imageUrl}
                      alt={`Profile option ${index + 1}`}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        console.error('[SETTINGS] Failed to load image:', imageUrl);
                        e.target.style.display = 'none';
                      }}
                    />
                    {profilePic === imageUrl && (
                      <div className="absolute inset-0 bg-primary/20 flex items-center justify-center">
                        <span className="text-primary text-xs font-bold">✓</span>
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="rounded-lg border bg-card p-6 shadow-sm">
          <h2 className="text-xl font-semibold mb-4">Personal Information</h2>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium leading-none">Name</label>
            <input
              type="text"
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
            <div className="space-y-2">
              <label className="text-sm font-medium leading-none">Email</label>
            <input
              type="email"
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              value={email}
                disabled
            />
          </div>
            <div className="space-y-2">
              <label className="text-sm font-medium leading-none">Phone</label>
          <input
            type="text"
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            value={phone}
            onChange={(e) => setPhone(e.target.value)}
          />
        </div>
            <div className="space-y-2">
              <label className="text-sm font-medium leading-none">Department</label>
            <input
              type="text"
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background disabled:cursor-not-allowed disabled:opacity-50"
              value={department}
              disabled
            />
          </div>
            <div className="space-y-2">
              <label className="text-sm font-medium leading-none">Designation</label>
            <input
              type="text"
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background disabled:cursor-not-allowed disabled:opacity-50"
              value={designation}
              disabled
            />
            </div>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6 shadow-sm">
          <h2 className="text-xl font-semibold mb-4">Change Password</h2>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label className="text-sm font-medium leading-none">Old Password</label>
            <input
              type="password"
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              value={oldPassword}
              onChange={(e) => setOldPassword(e.target.value)}
            />
          </div>
            <div className="space-y-2">
              <label className="text-sm font-medium leading-none">New Password</label>
            <input
              type="password"
                className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
            />
            </div>
          </div>
        </div>

        <div className="flex justify-end">
        <button
          onClick={handleSaveChanges}
            disabled={isLoading}
            className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2 bg-primary text-primary-foreground hover:bg-primary/90"
        >
            {isLoading ? "Saving..." : "Save Changes"}
        </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;
