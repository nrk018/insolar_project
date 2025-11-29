import { useEffect, useState } from "react";

const History = () => {
  const [defaulters, setDefaulters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [resettingStreak, setResettingStreak] = useState({});
  const [editingWorker, setEditingWorker] = useState(null);
  const [editForm, setEditForm] = useState({ name: '', email: '', mobile: '' });
  const [saving, setSaving] = useState(false);
  
  // Initialize selectedDate to today
  const getTodayDate = () => {
    const today = new Date();
    return today.toISOString().split('T')[0];
  };
  
  const [selectedDate, setSelectedDate] = useState(getTodayDate);

  // Calculate min and max dates (today and 7 days ago) - use useMemo to avoid recalculation
  const { maxDate, minDateStr } = (() => {
    const today = new Date();
    const max = today.toISOString().split('T')[0];
    const min = new Date(today);
    min.setDate(today.getDate() - 7);
    const minStr = min.toISOString().split('T')[0];
    return { maxDate: max, minDateStr: minStr };
  })();

  // Format date for display
  const formatDateDisplay = (dateStr) => {
    try {
      if (!dateStr) return "";
      const date = new Date(dateStr);
      if (isNaN(date.getTime())) return dateStr;
      const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
      const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
      return `${days[date.getDay()]}, ${months[date.getMonth()]} ${date.getDate()}, ${date.getFullYear()}`;
    } catch (e) {
      console.error("Error formatting date:", e);
      return dateStr;
    }
  };

  useEffect(() => {
    const fetchDefaulters = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`/api/ppe/defaulters?date=${selectedDate}`, {
          credentials: "include",
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Failed to load defaulters");
        }
        setDefaulters(Array.isArray(data) ? data : []);
      } catch (e) {
        console.error("Error fetching defaulters:", e);
        setError(e.message);
        setDefaulters([]);
      } finally {
        setLoading(false);
      }
    };
    fetchDefaulters();
  }, [selectedDate]);

  const handleEdit = (worker) => {
    setEditingWorker(worker);
    setEditForm({
      name: worker.name || '',
      email: worker.email || '',
      mobile: worker.mobile || ''
    });
  };

  const handleSaveEdit = async () => {
    if (!editingWorker) return;
    
    setSaving(true);
    try {
      const res = await fetch(`/api/ppe/update-worker`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: "include",
        body: JSON.stringify({
          worker_id: editingWorker.worker_id,
          name: editForm.name,
          email: editForm.email,
          mobile: editForm.mobile
        }),
      });
      
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Failed to update worker");
      }
      
      // Refresh the data
      const refreshRes = await fetch(`/api/ppe/defaulters?date=${selectedDate}`, {
        credentials: "include",
      });
      const refreshData = await refreshRes.json();
      if (refreshRes.ok) {
        setDefaulters(Array.isArray(refreshData) ? refreshData : []);
      }
      
      setEditingWorker(null);
      alert('Worker details updated successfully');
    } catch (e) {
      console.error("Error updating worker:", e);
      alert(`Failed to update worker: ${e.message}`);
    } finally {
      setSaving(false);
    }
  };

  const handleResetStreak = async (workerId, date) => {
    const key = `${workerId}_${date}`;
    setResettingStreak(prev => ({ ...prev, [key]: true }));
    
    try {
      // Ensure date is in YYYY-MM-DD format
      const dateStr = date ? (typeof date === 'string' ? date.split('T')[0] : date) : selectedDate;
      
      console.log(`[FRONTEND] Resetting streak for worker_id=${workerId}, date=${dateStr}`);
      
      const res = await fetch(`/api/ppe/reset-streak`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: "include",
        body: JSON.stringify({ worker_id: workerId, date: dateStr }),
      });
      
      // Check if response has content before parsing JSON
      const contentType = res.headers.get("content-type");
      let data;
      
      if (contentType && contentType.includes("application/json")) {
        const text = await res.text();
        if (!text || text.trim() === '') {
          throw new Error("Empty response from server");
        }
        try {
          data = JSON.parse(text);
        } catch (parseError) {
          console.error("JSON parse error:", parseError, "Response text:", text);
          throw new Error(`Invalid JSON response: ${text.substring(0, 100)}`);
        }
      } else {
        const text = await res.text();
        throw new Error(text || "Server returned non-JSON response");
      }
      
      if (!res.ok) {
        throw new Error(data.error || data.message || `Server error: ${res.status}`);
      }
      
      // Refresh the data
      const refreshRes = await fetch(`/api/ppe/defaulters?date=${selectedDate}`, {
        credentials: "include",
      });
      const refreshData = await refreshRes.json();
      if (refreshRes.ok) {
        setDefaulters(Array.isArray(refreshData) ? refreshData : []);
      }
    } catch (e) {
      console.error("Error resetting streak:", e);
      alert(`Failed to reset streak: ${e.message}`);
    } finally {
      setResettingStreak(prev => ({ ...prev, [key]: false }));
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">PPE Violations History</h1>
        <p className="text-muted-foreground">
          Workers with PPE violations for the selected date.
        </p>
      </div>

      {/* Date Selection */}
      <div className="flex items-center gap-4 p-4 rounded-lg border bg-card">
        <label htmlFor="date-select" className="text-sm font-medium">
          Select Date:
        </label>
        <input
          id="date-select"
          type="date"
          value={selectedDate || ""}
          min={minDateStr}
          max={maxDate}
          onChange={(e) => {
            if (e.target.value) {
              setSelectedDate(e.target.value);
            }
          }}
          className="px-3 py-2 border rounded-md text-sm"
        />
        <div className="text-sm text-muted-foreground">
          {formatDateDisplay(selectedDate)}
        </div>
      </div>

      {loading ? (
        <div className="rounded-lg border bg-card p-12 text-center">
          <p className="text-muted-foreground">Loading violations...</p>
        </div>
      ) : error ? (
        <div className="rounded-lg border border-destructive bg-destructive/10 p-4">
          <p className="text-destructive">{error}</p>
        </div>
      ) : (
        <div className="rounded-lg border bg-card shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                    Worker ID
                  </th>
                  <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                    Worker Name
                  </th>
                  <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                    Violation Items
                  </th>
                  <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                    Streak
                  </th>
                  <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                    Last Detected
                  </th>
                  <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {defaulters.length === 0 ? (
                  <tr>
                    <td
                      colSpan={6}
                      className="p-8 text-center text-sm text-muted-foreground"
                    >
                      No PPE violations found for this date.
                    </td>
                  </tr>
                ) : (
                  defaulters.map((d, idx) => {
                    // Determine which PPE items were violated
                    const violationItems = [];
                    if (d.helmet_status === "No") violationItems.push("Helmet");
                    if (d.gloves_status === "No") violationItems.push("Gloves");
                    if (d.vests_status === "No") violationItems.push("Jacket");
                    if (d.boots_status === "No") violationItems.push("Boots");
                    const violationText = violationItems.length > 0 
                      ? violationItems.join(", ")
                      : "—";

                    const resetKey = `${d.worker_id}_${d.date}`;
                    const isResetting = resettingStreak[resetKey] || false;

                    // Format last detected timestamp
                    let lastDetectedDisplay = "—";
                    if (d.last_detected_at) {
                      try {
                        const timestampStr = d.last_detected_at;
                        const utcString = timestampStr.endsWith('Z') ? timestampStr : timestampStr + 'Z';
                        const detectedAt = new Date(utcString);
                        
                        if (!isNaN(detectedAt.getTime())) {
                          const timeString = detectedAt.toLocaleTimeString('en-IN', { 
                            hour: '2-digit', 
                            minute: '2-digit',
                            second: '2-digit',
                            hour12: true,
                            timeZone: 'Asia/Kolkata'
                          });
                          const dateString = detectedAt.toLocaleDateString('en-IN', {
                            month: 'short',
                            day: 'numeric',
                            timeZone: 'Asia/Kolkata'
                          });
                          lastDetectedDisplay = `${dateString}, ${timeString}`;
                          if (d.camera_source) {
                            lastDetectedDisplay += ` (${d.camera_source === 'rtsp' ? 'RTSP' : 'Webcam'})`;
                          }
                        }
                      } catch (e) {
                        console.error('Error formatting timestamp:', e);
                      }
                    }

                    return (
                      <tr
                        key={idx}
                        className="border-b transition-colors hover:bg-muted/50"
                      >
                        <td className="p-4 text-sm font-mono">{d.worker_id}</td>
                        <td className="p-4 text-sm">{d.name || "—"}</td>
                        <td className="p-4 text-sm text-red-600 font-medium">
                          {violationText}
                        </td>
                        <td className="p-4 text-sm font-semibold text-red-600">
                          {d.streak || 0}
                        </td>
                        <td className="p-4 text-sm text-muted-foreground">
                          {lastDetectedDisplay}
                        </td>
                        <td className="p-4 text-sm">
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => handleEdit(d)}
                              className="px-3 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600"
                            >
                              Edit
                            </button>
                            <button
                              onClick={() => {
                                if (window.confirm(`Reset streak for ${d.name || d.worker_id}?`)) {
                                  handleResetStreak(d.worker_id, d.date);
                                }
                              }}
                              disabled={isResetting}
                              className="px-3 py-1 text-xs bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
                            >
                              {isResetting ? "Resetting..." : "Reset Streak"}
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Edit Modal */}
      {editingWorker && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-2xl font-bold mb-4">Edit Worker Details</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Worker ID</label>
                <input
                  type="text"
                  value={editingWorker.worker_id}
                  disabled
                  className="w-full px-3 py-2 border rounded-md bg-gray-100 text-gray-600"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Name</label>
                <input
                  type="text"
                  value={editForm.name}
                  onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
                  className="w-full px-3 py-2 border rounded-md"
                  placeholder="Worker name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Email</label>
                <input
                  type="email"
                  value={editForm.email}
                  onChange={(e) => setEditForm({ ...editForm, email: e.target.value })}
                  className="w-full px-3 py-2 border rounded-md"
                  placeholder="worker@example.com"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Mobile</label>
                <input
                  type="tel"
                  value={editForm.mobile}
                  onChange={(e) => setEditForm({ ...editForm, mobile: e.target.value })}
                  className="w-full px-3 py-2 border rounded-md"
                  placeholder="Phone number"
                />
              </div>
            </div>
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setEditingWorker(null)}
                className="px-4 py-2 text-sm border rounded-md hover:bg-gray-100"
                disabled={saving}
              >
                Cancel
              </button>
              <button
                onClick={handleSaveEdit}
                disabled={saving}
                className="px-4 py-2 text-sm bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {saving ? "Saving..." : "Save"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default History;
