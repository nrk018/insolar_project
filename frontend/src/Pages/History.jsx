import { useEffect, useState } from "react";

const History = () => {
  const [defaulters, setDefaulters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDefaulters = async () => {
      try {
        const res = await fetch("/api/ppe/defaulters?minStreak=3", {
          credentials: "include",
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Failed to load defaulters");
        }
        setDefaulters(data);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    fetchDefaulters();
  }, []);

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">PPE Defaulters</h1>
        <p className="text-muted-foreground">
          Workers with repeated PPE violations and high streaks.
        </p>
      </div>

      {loading ? (
        <div className="rounded-lg border bg-card p-12 text-center">
          <p className="text-muted-foreground">Loading defaulters...</p>
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
                {defaulters.length === 0 ? (
                  <tr>
                    <td
                      colSpan={5}
                      className="p-8 text-center text-sm text-muted-foreground"
                    >
                      No PPE defaulters found.
                    </td>
                  </tr>
                ) : (
                  defaulters.map((d, idx) => (
                    <tr
                      key={idx}
                      className="border-b transition-colors hover:bg-muted/50"
                    >
                      <td className="p-4 text-sm font-mono">{d.worker_id}</td>
                      <td className="p-4 text-sm">{d.date || "—"}</td>
                      <td className="p-4 text-sm">{d.daily_violations}</td>
                      <td className="p-4 text-sm">{d.total_violations}</td>
                      <td className="p-4 text-sm font-semibold text-red-600">
                        {d.streak}
                      </td>
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

export default History;
