import { Route, Routes, useLocation } from "react-router-dom";
import SignIn from "./Pages/SignIn";
import SignUp from "./Pages/SignUp";
import Dashboard from "./Pages/Dashboard";
import History from "./Pages/History";
import Settings from "./Pages/Settings";
import CameraMonitor from "./Pages/CameraMonitor";
import ImageAnalysis from "./Pages/ImageAnalysis";
import AddEmployee from "./Pages/AddEmployee";
import Employees from "./Pages/Employees";
import TopNavbar from "./Components/TopNavbar";
import { useState, useEffect } from "react";
import axios from "./utils/axiosConfig";

const App = () => {
  const location = useLocation();
  const isAuthPage =
    location.pathname === "/" || location.pathname === "/signup";
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    // Check if user is admin (case-insensitive)
    const checkAdmin = async () => {
      try {
        const response = await axios.get("/api/user");
        const designation = response.data.designation;
        // Handle all variations: "admin", "Admin", "ADMIN", etc.
        const isAdminUser = designation && 
          designation.toString().toLowerCase().trim() === "admin";
        console.log("[ADMIN CHECK] Designation:", designation, "Is Admin:", isAdminUser);
        setIsAdmin(isAdminUser);
      } catch (error) {
        console.error("Error checking admin status:", error);
        setIsAdmin(false);
      }
    };
    if (!isAuthPage) {
      checkAdmin();
    }
  }, [isAuthPage]);

  return (
    <div className="min-h-screen bg-background">
      {!isAuthPage && <TopNavbar isAdmin={isAdmin} />}
      <Routes>
        <Route path="/" element={<SignIn />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/history" element={<History />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/camera" element={<CameraMonitor />} />
        <Route path="/analyze" element={<ImageAnalysis />} />
        <Route path="/add-employee" element={<AddEmployee />} />
        <Route path="/employees" element={<Employees />} />
      </Routes>
    </div>
  );
};

export default App;