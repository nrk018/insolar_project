import { supabase } from "./database.js";

// Helper to get today/yesterday in YYYY-MM-DD
function getIsoDate(offsetDays = 0) {
  const d = new Date(Date.now() + offsetDays * 24 * 60 * 60 * 1000);
  return d.toISOString().split("T")[0];
}

export async function resolveWorkerFromName(name) {
  const { data, error } = await supabase
    .from("users")
    .select("employee_id, name, email, phone, department")
    .eq("name", name)
    .limit(1);

  if (error) {
    throw new Error(`PPE: failed to resolve worker: ${error.message || error}`);
  }
  if (!data || data.length === 0) {
    return null;
  }

  const user = data[0];
  const workerId = `W${String(user.employee_id).padStart(3, "0")}`;

  return {
    worker_id: workerId,
    name: user.name,
    email: user.email,
    mobile: user.phone,
    department: user.department,
  };
}

export async function upsertWorkerDetails(worker) {
  const { error } = await supabase.from("worker_details").upsert(
    {
      worker_id: worker.worker_id,
      name: worker.name,
      email: worker.email || null,
      mobile: worker.mobile || null,
      state: worker.state || "Uttar Pradesh",
    },
    { onConflict: "worker_id" }
  );

  if (error) {
    throw new Error(`PPE: failed to upsert worker_details: ${error.message || error}`);
  }
}

export async function recordPpeEvent({
  worker,
  ppeCompliant,
  ppeItems,
  ppeConfidence,
}) {
  const today = getIsoDate(0);
  const yesterday = getIsoDate(-1);

  const helmet = ppeItems?.helmet === true;
  const gloves = ppeItems?.gloves === true;
  const vests = ppeItems?.jacket === true;
  const boots = ppeItems?.boots === true;

  const dailyViolations = [helmet, gloves, vests, boots].filter((ok) => !ok).length;

  // Previous totals / streak
  const { data: prevRows, error: prevErr } = await supabase
    .from("workers")
    .select("total_violations, streak")
    .eq("worker_id", worker.worker_id)
    .order("date", { ascending: false })
    .limit(1);

  if (prevErr) {
    throw new Error(`PPE: failed to read previous stats: ${prevErr.message || prevErr}`);
  }

  const prev =
    prevRows && prevRows.length > 0
      ? prevRows[0]
      : { total_violations: 0, streak: 0 };

  const totalViolations = prev.total_violations + dailyViolations;

  // Yesterday violations
  const { data: yRows, error: yErr } = await supabase
    .from("workers")
    .select("daily_violations")
    .eq("worker_id", worker.worker_id)
    .eq("date", yesterday)
    .limit(1);

  if (yErr) {
    throw new Error(`PPE: failed to read yesterday stats: ${yErr.message || yErr}`);
  }

  const yesterdayViol =
    yRows && yRows.length > 0 ? yRows[0].daily_violations || 0 : 0;

  const newStreak =
    dailyViolations > 0
      ? yesterdayViol > 0
        ? prev.streak + 1
        : 1
      : 0;

  const helmetStatus = helmet ? "Yes" : "No";
  const glovesStatus = gloves ? "Yes" : "No";
  const vestsStatus = vests ? "Yes" : "No";
  const bootsStatus = boots ? "Yes" : "No";

  const { error: upsertErr } = await supabase.from("workers").upsert(
    {
      worker_id: worker.worker_id,
      date: today,
      helmet_status: helmetStatus,
      gloves_status: glovesStatus,
      vests_status: vestsStatus,
      boots_status: bootsStatus,
      daily_violations: dailyViolations,
      total_violations: totalViolations,
      streak: newStreak,
      last_modified: new Date().toISOString(),
      // last_notified_date will be updated by notificationService when notification is sent
    },
    { onConflict: "worker_id,date" }
  );

  if (upsertErr) {
    throw new Error(`PPE: failed to upsert workers: ${upsertErr.message || upsertErr}`);
  }

  return {
    worker_id: worker.worker_id,
    name: worker.name,
    email: worker.email,
    mobile: worker.mobile,
    helmet_status: helmetStatus,
    gloves_status: glovesStatus,
    vests_status: vestsStatus,
    boots_status: bootsStatus,
    daily_violations: dailyViolations,
    total_violations: totalViolations,
    streak: newStreak,
    ppe_confidence: ppeConfidence || 0,
  };
}

export async function listPpeWorkers() {
  const { data, error } = await supabase
    .from("workers")
    .select(
      "worker_id, date, helmet_status, gloves_status, vests_status, boots_status, daily_violations, total_violations, streak"
    )
    .order("date", { ascending: false });

  if (error) {
    throw new Error(`PPE: failed to list workers: ${error.message || error}`);
  }
  return data || [];
}

export async function listDefaulters({ minStreak = 3 } = {}) {
  const { data, error } = await supabase
    .from("workers")
    .select(
      "worker_id, date, daily_violations, total_violations, streak"
    )
    .gte("streak", minStreak)
    .order("streak", { ascending: false });

  if (error) {
    throw new Error(`PPE: failed to list defaulters: ${error.message || error}`);
  }
  return data || [];
}




