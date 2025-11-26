import { useState } from "react";
import axios from '../utils/axiosConfig';
import { useNavigate } from 'react-router-dom';

const AddEmployee = () => {
  const navigate = useNavigate();
  
  const [newEmployeeName, setNewEmployeeName] = useState("");
  const [newEmployeeEmail, setNewEmployeeEmail] = useState("");
  const [newEmployeeID, setNewEmployeeID] = useState("");
  const [newEmployeeDepartment, setNewEmployeeDepartment] = useState("");
  const [newEmployeeDesignation, setNewEmployeeDesignation] = useState("");
  const [newEmployeePhone, setNewEmployeePhone] = useState("");
  const [newEmployeeImages, setNewEmployeeImages] = useState([]);
  const [isAddingEmployee, setIsAddingEmployee] = useState(false);
  const [addEmployeeMessage, setAddEmployeeMessage] = useState({ type: '', text: '' });

  const handleImageChange = (e) => {
    const files = Array.from(e.target.files);
    setNewEmployeeImages(files);
  };

  const handleAddEmployee = async () => {
    if (!newEmployeeName || !newEmployeeID || !newEmployeeDepartment || !newEmployeeDesignation) {
      setAddEmployeeMessage({ type: 'error', text: 'Please fill all required fields (Name, Employee ID, Department, Designation)' });
      return;
    }

    if (newEmployeeImages.length === 0) {
      setAddEmployeeMessage({ type: 'error', text: 'Please upload at least one image of the employee' });
      return;
    }

    // Validate email format if provided
    if (newEmployeeEmail && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(newEmployeeEmail)) {
      setAddEmployeeMessage({ type: 'error', text: 'Invalid email format' });
      return;
    }

    setIsAddingEmployee(true);
    setAddEmployeeMessage({ type: '', text: '' });

    try {
      const formData = new FormData();
      formData.append('name', newEmployeeName);
      if (newEmployeeEmail) {
        formData.append('email', newEmployeeEmail);
      }
      formData.append('employee_id', newEmployeeID);
      formData.append('department', newEmployeeDepartment);
      formData.append('designation', newEmployeeDesignation);
      if (newEmployeePhone) {
        formData.append('phone', newEmployeePhone);
      }
      
      newEmployeeImages.forEach((image) => {
        formData.append('profilePhotos', image);
      });

      const response = await axios.post('/api/admin/add-employee', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setAddEmployeeMessage({ 
        type: 'success', 
        text: `Employee added successfully! Default password: ${response.data.default_password}` 
      });
      
      // Reset form after 3 seconds and navigate to employees list
      setTimeout(() => {
        navigate('/employees');
      }, 2000);
    } catch (error) {
      setAddEmployeeMessage({ 
        type: 'error', 
        text: error.response?.data?.error || 'Failed to add employee' 
      });
    } finally {
      setIsAddingEmployee(false);
    }
  };

  const handleReset = () => {
    setNewEmployeeName("");
    setNewEmployeeEmail("");
    setNewEmployeeID("");
    setNewEmployeeDepartment("");
    setNewEmployeeDesignation("");
    setNewEmployeePhone("");
    setNewEmployeeImages([]);
    const fileInput = document.getElementById('employee-images');
    if (fileInput) fileInput.value = '';
    setAddEmployeeMessage({ type: '', text: '' });
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Add New Employee</h1>
        <p className="text-muted-foreground">
          Upload employee images and details. Face recognition embeddings will be generated automatically.
        </p>
      </div>

      <div className="rounded-lg border bg-card p-6 shadow-sm">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <label className="text-sm font-medium leading-none">
              Name <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              value={newEmployeeName}
              onChange={(e) => setNewEmployeeName(e.target.value)}
              placeholder="Employee Name"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium leading-none">
              Email (Optional)
            </label>
            <input
              type="email"
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              value={newEmployeeEmail}
              onChange={(e) => setNewEmployeeEmail(e.target.value)}
              placeholder="employee@example.com"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium leading-none">
              Employee ID <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              value={newEmployeeID}
              onChange={(e) => setNewEmployeeID(e.target.value)}
              placeholder="001"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium leading-none">
              Phone (Optional)
            </label>
            <input
              type="text"
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              value={newEmployeePhone}
              onChange={(e) => setNewEmployeePhone(e.target.value)}
              placeholder="1234567890"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium leading-none">
              Department <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              value={newEmployeeDepartment}
              onChange={(e) => setNewEmployeeDepartment(e.target.value)}
              placeholder="Engineering"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium leading-none">
              Designation <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              value={newEmployeeDesignation}
              onChange={(e) => setNewEmployeeDesignation(e.target.value)}
              placeholder="Engineer"
            />
          </div>

          <div className="space-y-2 md:col-span-2">
            <label className="text-sm font-medium leading-none">
              Employee Images <span className="text-red-500">*</span>
            </label>
            <input
              id="employee-images"
              type="file"
              multiple
              accept="image/*"
              onChange={handleImageChange}
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            <p className="text-xs text-muted-foreground">
              Upload multiple images (at least 1) for better face recognition accuracy. Supported formats: PNG, JPG, JPEG
            </p>
            {newEmployeeImages.length > 0 && (
              <p className="text-xs text-green-600">
                {newEmployeeImages.length} image(s) selected
              </p>
            )}
          </div>
        </div>

        {addEmployeeMessage.text && (
          <div className={`mt-4 p-3 rounded-md ${
            addEmployeeMessage.type === 'success' 
              ? 'bg-green-50 border border-green-200 text-green-800' 
              : 'bg-red-50 border border-red-200 text-red-800'
          }`}>
            <p className="text-sm">{addEmployeeMessage.text}</p>
          </div>
        )}

        <div className="flex justify-end gap-4 mt-6">
          <button
            onClick={handleReset}
            disabled={isAddingEmployee}
            className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2 bg-gray-500 text-white hover:bg-gray-600"
          >
            Reset
          </button>
          <button
            onClick={handleAddEmployee}
            disabled={isAddingEmployee}
            className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2 bg-green-600 text-white hover:bg-green-700"
          >
            {isAddingEmployee ? "Adding Employee..." : "Add Employee"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default AddEmployee;

