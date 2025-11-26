import { useEffect, useState } from "react";
import PPEStatus from "../Components/PPEStatus";

const Dashboard = () => {
  const [workers, setWorkers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("/api/ppe/workers", {
          credentials: "include",
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Failed to load PPE data");
        }
        setWorkers(data);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">PPE Safety Dashboard</h1>
        <p className="text-muted-foreground">
          Live overview of worker PPE compliance and recent violations.
        </p>
      </div>

      {loading ? (
        <div className="rounded-lg border bg-card p-12 text-center">
          <p className="text-muted-foreground">Loading PPE records...</p>
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
                    Date
                  </th>
                  <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                    PPE Status
                  </th>
                  <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                    Daily Violations
                  </th>
                  <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                    Total Violations
                  </th>
                  <th className="h-12 px-4 text-left text-xs font-medium text-muted-foreground">
                    Streak
                  </th>
                </tr>
              </thead>
              <tbody>
                {workers.length === 0 ? (
                  <tr>
                    <td
                      colSpan={6}
                      className="p-8 text-center text-sm text-muted-foreground"
                    >
                      No PPE records found.
                    </td>
                  </tr>
                ) : (
                  workers.map((w, idx) => (
                    <tr
                      key={idx}
                      className="border-b transition-colors hover:bg-muted/50"
                    >
                      <td className="p-4 text-sm font-mono">{w.worker_id}</td>
                      <td className="p-4 text-sm">
                        {w.date || "—"}
                      </td>
                      <td className="p-4">
                        <PPEStatus
                          ppeCompliant={w.daily_violations === 0}
                          ppeItems={{
                            helmet: w.helmet_status === "Yes",
                            gloves: w.gloves_status === "Yes",
                            boots: w.boots_status === "Yes",
                            jacket: w.vests_status === "Yes",
                          }}
                        />
                      </td>
                      <td className="p-4 text-sm">{w.daily_violations}</td>
                      <td className="p-4 text-sm">{w.total_violations}</td>
                      <td className="p-4 text-sm">{w.streak}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
