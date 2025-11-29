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

  // Check if there's already a record for today (to prevent multiple updates)
  const { data: todayRows, error: todayErr } = await supabase
    .from("workers")
    .select("streak, total_violations, daily_violations")
    .eq("worker_id", worker.worker_id)
    .eq("date", today)
    .limit(1);

  if (todayErr) {
    throw new Error(`PPE: failed to read today's stats: ${todayErr.message || todayErr}`);
  }

  // If record already exists for today, use existing values (don't recalculate)
  if (todayRows && todayRows.length > 0) {
    const existingRecord = todayRows[0];
    // Keep existing streak and total_violations (already set once today)
    const newStreak = existingRecord.streak || 0;
    const totalViolations = existingRecord.total_violations || 0;
    const existingDailyViolations = existingRecord.daily_violations || 0;

    const helmetStatus = helmet ? "Yes" : "No";
    const glovesStatus = gloves ? "Yes" : "No";
    const vestsStatus = vests ? "Yes" : "No";
    const bootsStatus = boots ? "Yes" : "No";

    // Update PPE status but keep existing totals
    const { error: updateErr } = await supabase
      .from("workers")
      .update({
        helmet_status: helmetStatus,
        gloves_status: glovesStatus,
        vests_status: vestsStatus,
        boots_status: bootsStatus,
        daily_violations: existingDailyViolations, // Keep existing daily violations
        total_violations: totalViolations, // Keep existing total
        streak: newStreak, // Keep existing streak
        last_modified: new Date().toISOString(),
      })
      .eq("worker_id", worker.worker_id)
      .eq("date", today);

    if (updateErr) {
      throw new Error(`PPE: failed to update workers: ${updateErr.message || updateErr}`);
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
      daily_violations: existingDailyViolations,
      total_violations: totalViolations,
      streak: newStreak,
      ppe_confidence: ppeConfidence || 0,
    };
  }

  // New record for today - calculate totals from previous day
  const { data: prevRows, error: prevErr } = await supabase
    .from("workers")
    .select("total_violations")
    .eq("worker_id", worker.worker_id)
    .order("date", { ascending: false })
    .limit(1);

  if (prevErr) {
    throw new Error(`PPE: failed to read previous stats: ${prevErr.message || prevErr}`);
  }

  const prev =
    prevRows && prevRows.length > 0
      ? prevRows[0]
      : { total_violations: 0 };

  // Only add to total_violations for new day
  const totalViolations = prev.total_violations + dailyViolations;

  // Streak logic: 
  // - If new record for today: streak = 1 if violation, 0 if no violation
  // - Streak is per day (1 for each violation day), not cumulative
  const newStreak = dailyViolations > 0 ? 1 : 0;

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

export async function listDefaulters({ date, minStreak = 1 } = {}) {
  let query = supabase
    .from("workers")
    .select(
      "worker_id, date, helmet_status, gloves_status, vests_status, boots_status, total_violations, streak"
    );

  // If date is provided, filter by that date, otherwise get all
  if (date) {
    query = query.eq("date", date);
  }

  // Only show workers with violations (streak >= 1)
  query = query.gte("streak", minStreak)
    .order("streak", { ascending: false })
    .order("worker_id", { ascending: true });

  const { data, error } = await query;

  if (error) {
    throw new Error(`PPE: failed to list defaulters: ${error.message || error}`);
  }

  // Get worker names from worker_details table and latest detection timestamps
  if (data && data.length > 0) {
    const workerIds = [...new Set(data.map(d => d.worker_id))];
    
    // Get worker details (name, email, mobile)
    const { data: workerDetails, error: detailsError } = await supabase
      .from("worker_details")
      .select("worker_id, name, email, mobile")
      .in("worker_id", workerIds);

    // Get most recent detection timestamps for each worker
    // Query detection_events for the most recent detection per worker on the selected date
    const timestampMap = {};
    
    try {
      if (date) {
        // Get start and end of the selected date in UTC
        const selectedDate = new Date(date + 'T00:00:00Z');
        const nextDate = new Date(selectedDate);
        nextDate.setDate(nextDate.getDate() + 1);
        
        const startISO = selectedDate.toISOString();
        const endISO = nextDate.toISOString();
        
        // Get all detections for these workers on this date, then find the most recent per worker
        const { data: allDetections, error: detError } = await supabase
          .from("detection_events")
          .select("worker_id, detected_at, camera_source")
          .in("worker_id", workerIds)
          .gte("detected_at", startISO)
          .lt("detected_at", endISO)
          .order("detected_at", { ascending: false });
        
        if (!detError && allDetections) {
          // Group by worker_id and keep only the most recent (first one since sorted DESC)
          const seenWorkers = new Set();
          allDetections.forEach(det => {
            if (!seenWorkers.has(det.worker_id)) {
              timestampMap[det.worker_id] = {
                detected_at: det.detected_at,
                camera_source: det.camera_source || null
              };
              seenWorkers.add(det.worker_id);
            }
          });
        }
      } else {
        // If no date filter, get the most recent detection overall for each worker
        // Use a more efficient approach: get all recent detections and group by worker_id
        const { data: allDetections, error: detError } = await supabase
          .from("detection_events")
          .select("worker_id, detected_at, camera_source")
          .in("worker_id", workerIds)
          .order("detected_at", { ascending: false });
        
        if (!detError && allDetections) {
          // Group by worker_id and keep only the most recent (first one since sorted DESC)
          const seenWorkers = new Set();
          allDetections.forEach(det => {
            if (!seenWorkers.has(det.worker_id)) {
              timestampMap[det.worker_id] = {
                detected_at: det.detected_at,
                camera_source: det.camera_source || null
              };
              seenWorkers.add(det.worker_id);
            }
          });
        }
      }
    } catch (err) {
      // If detection_events table doesn't exist or query fails, continue without timestamps
      console.warn(`[listDefaulters] Could not fetch detection timestamps:`, err.message);
    }

    if (!detailsError && workerDetails) {
      const detailsMap = {};
      workerDetails.forEach(w => {
        detailsMap[w.worker_id] = {
          name: w.name,
          email: w.email,
          mobile: w.mobile
        };
      });

      // Add names, email, mobile and timestamps to the results
      return data.map(d => ({
        ...d,
        name: detailsMap[d.worker_id]?.name || null,
        email: detailsMap[d.worker_id]?.email || null,
        mobile: detailsMap[d.worker_id]?.mobile || null,
        last_detected_at: timestampMap[d.worker_id]?.detected_at || null,
        camera_source: timestampMap[d.worker_id]?.camera_source || null
      }));
    }
  }

  return data || [];
}




