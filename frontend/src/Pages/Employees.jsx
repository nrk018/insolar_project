import { useState, useEffect } from "react";
import axios from '../utils/axiosConfig';

const Employees = () => {
  const [employees, setEmployees] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetchEmployees();
  }, []);

  const fetchEmployees = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/admin/employees');
      setEmployees(response.data);
      setError(null);
    } catch (err) {
      console.error("Error fetching employees:", err);
      setError(err.response?.data?.error || "Failed to load employees");
    } finally {
      setLoading(false);
    }
  };

  const extractNameFromImage = (imageName) => {
    if (!imageName) return '';
    return imageName.split('_')[0];
  };

  const filteredEmployees = employees.filter(emp => {
    const searchLower = searchTerm.toLowerCase();
    return (
      emp.name?.toLowerCase().includes(searchLower) ||
      emp.employee_id?.toLowerCase().includes(searchLower) ||
      emp.email?.toLowerCase().includes(searchLower) ||
      emp.department?.toLowerCase().includes(searchLower) ||
      emp.designation?.toLowerCase().includes(searchLower)
    );
  });

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="rounded-lg border bg-card p-12 text-center">
          <p className="text-muted-foreground">Loading employees...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <div className="rounded-lg border border-destructive bg-destructive/10 p-4">
          <p className="text-destructive">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Employees</h1>
        <p className="text-muted-foreground">
          View and manage all employees in the system.
        </p>
      </div>

      {/* Search Bar */}
      <div className="rounded-lg border bg-card p-4">
        <input
          type="text"
          placeholder="Search by name, ID, email, department, or designation..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
        />
      </div>

      {/* Employees List */}
      <div className="rounded-lg border bg-card shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                  Photo
                </th>
                <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                  Name
                </th>
                <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                  Employee ID
                </th>
                <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                  Email
                </th>
                <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                  Phone
                </th>
                <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                  Department
                </th>
                <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                  Designation
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredEmployees.length === 0 ? (
                <tr>
                  <td
                    colSpan={7}
                    className="p-8 text-center text-sm text-muted-foreground"
                  >
                    {searchTerm ? "No employees found matching your search." : "No employees found."}
                  </td>
                </tr>
              ) : (
                filteredEmployees.map((emp, idx) => (
                  <tr
                    key={idx}
                    className="border-b transition-colors hover:bg-muted/50"
                  >
                    <td className="p-4">
                      {emp.profilephoto && emp.profilephoto.length > 0 ? (
                        <img
                          src={emp.profilephoto[0]}
                          alt={emp.name}
                          className="w-12 h-12 rounded-full border-2 border-border object-cover"
                          onError={(e) => {
                            e.target.style.display = 'none';
                            const fallback = e.target.parentElement.querySelector('.profile-fallback');
                            if (fallback) fallback.style.display = 'flex';
                          }}
                        />
                      ) : null}
                      <div
                        className="w-12 h-12 rounded-full bg-muted flex items-center justify-center profile-fallback"
                        style={{ display: emp.profilephoto && emp.profilephoto.length > 0 ? 'none' : 'flex' }}
                      >
                        <span className="text-lg font-medium">
                          {emp.name?.charAt(0)?.toUpperCase() || '?'}
                        </span>
                      </div>
                    </td>
                    <td className="p-4 text-sm font-medium">{emp.name || "—"}</td>
                    <td className="p-4 text-sm font-mono">{emp.employee_id || "—"}</td>
                    <td className="p-4 text-sm">{emp.email || "—"}</td>
                    <td className="p-4 text-sm">{emp.phone || "—"}</td>
                    <td className="p-4 text-sm">{emp.department || "—"}</td>
                    <td className="p-4 text-sm">{emp.designation || "—"}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Summary */}
      <div className="text-sm text-muted-foreground">
        Showing {filteredEmployees.length} of {employees.length} employee(s)
      </div>
    </div>
  );
};

export default Employees;

