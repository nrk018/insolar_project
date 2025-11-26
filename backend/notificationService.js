import { supabase } from "./database.js";
import axios from "axios";
import nodemailer from "nodemailer";

// Basic clone of Python-Field-Project notification semantics, simplified

async function logNotification({
  worker_id,
  type,
  recipient,
  message,
  status,
  reason,
}) {
  await supabase.from("notifications").insert({
    worker_id,
    type,
    recipient: recipient || "N/A",
    message,
    status,
    reason,
    timestamp: new Date().toISOString(),
  });
}

async function sendSms(toNumber, message, workerId) {
  const url = "https://www.smsidea.co.in/sendbulksms.aspx";

  const mobile = process.env.SMSIDEA_USERNAME;
  const password = process.env.SMSIDEA_PASSWORD;
  const senderid = process.env.SMSIDEA_SENDER_ID;

  const payload = {
    mobile,
    password,
    senderid,
    msgtype: "uc",
    message: [
      {
        text: message,
        to: toNumber,
        scheduledate: "",
      },
    ],
  };

  try {
    const response = await axios.get(url, {
      params: { data: JSON.stringify(payload) },
      timeout: 5000,
    });
    const text = String(response.data || "").trim();

    if (
      response.status === 200 &&
      (text.includes("000 : success") || text.includes("1 SMS Sent"))
    ) {
      await logNotification({
        worker_id: workerId,
        type: "SMS",
        recipient: toNumber,
        message,
        status: "Sent",
        reason: "Sent successfully",
      });
      return { ok: true, reason: "Sent successfully" };
    }

    await logNotification({
      worker_id: workerId,
      type: "SMS",
      recipient: toNumber,
      message,
      status: "Failed",
      reason: text || "Unknown error",
    });
    return { ok: false, reason: text || "Unknown error" };
  } catch (err) {
    const reason = err.message || "Network error";
    await logNotification({
      worker_id: workerId,
      type: "SMS",
      recipient: toNumber,
      message,
      status: "Failed",
      reason,
    });
    return { ok: false, reason };
  }
}

async function sendEmail({ workerId, workerName, recipients, subject, html }) {
  if (!recipients || recipients.length === 0) {
    return { ok: false, reason: "No recipients" };
  }

  const user = process.env.EMAIL_USER;
  const pass = process.env.EMAIL_PASS;

  const transporter = nodemailer.createTransport({
    host: "smtp.gmail.com",
    port: 587,
    secure: false,
    auth: { user, pass },
  });

  try {
    await transporter.sendMail({
      from: user,
      to: recipients.join(","),
      subject,
      html,
    });

    for (const r of recipients) {
      await logNotification({
        worker_id: workerId,
        type: "Email",
        recipient: r,
        message: subject,
        status: "Sent",
        reason: "Sent successfully",
      });
    }

    return { ok: true, reason: "Sent successfully" };
  } catch (err) {
    const reason = err.message || "SMTP error";
    for (const r of recipients) {
      await logNotification({
        worker_id: workerId,
        type: "Email",
        recipient: r,
        message: subject,
        status: "Failed",
        reason,
      });
    }
    return { ok: false, reason };
  }
}

async function updateLastNotifiedDate(workerId, today) {
  // Update the last_notified_date in the workers table for today's record
  const { error } = await supabase
    .from("workers")
    .update({ last_notified_date: today })
    .eq("worker_id", workerId)
    .eq("date", today);

  if (error) {
    // If column doesn't exist, log warning (backward compatibility)
    if (error.message && error.message.includes("column") && error.message.includes("does not exist")) {
      console.warn("last_notified_date column not found. Run migration: add_last_notified_date.sql");
    } else {
      console.error(`Failed to update last_notified_date for ${workerId}:`, error);
    }
  }
}

export async function maybeNotifyForPpe({
  worker,
  dailyViolations,
  totalViolations,
  streak,
}) {
  if (dailyViolations <= 0) {
    return { sms: false, email: false };
  }

  const today = new Date().toISOString().split("T")[0];

  // Check if worker has already been notified today
  // Note: If last_notified_date column doesn't exist yet, this will return null
  // and we'll proceed with notification (backward compatibility)
  const { data: workerRows, error: workerErr } = await supabase
    .from("workers")
    .select("last_notified_date")
    .eq("worker_id", worker.worker_id)
    .eq("date", today)
    .limit(1);

  if (workerErr) {
    // If column doesn't exist, log warning but continue (backward compatibility)
    if (workerErr.message && workerErr.message.includes("column") && workerErr.message.includes("does not exist")) {
      console.warn("last_notified_date column not found. Run migration: add_last_notified_date.sql");
    } else {
      console.error("Error checking last_notified_date:", workerErr);
    }
  }

  const alreadyNotifiedToday =
    workerRows &&
    workerRows.length > 0 &&
    workerRows[0] &&
    workerRows[0].last_notified_date === today;

  if (alreadyNotifiedToday) {
    console.log(
      `Worker ${worker.worker_id} already notified today (${today}). Skipping notification.`
    );
    return { sms: false, email: false, reason: "Already notified today" };
  }

  // Threshold from settings
  const { data: settingsRows } = await supabase
    .from("settings")
    .select("value")
    .eq("key", "email_threshold")
    .limit(1);
  const threshold =
    settingsRows && settingsRows.length > 0
      ? settingsRows[0].value
      : 4;

  // How many SMS already sent today? (for threshold check)
  const { data: smsCountRows } = await supabase
    .from("notifications")
    .select("id")
    .eq("worker_id", worker.worker_id)
    .eq("type", "SMS")
    .gte("timestamp", `${today}T00:00:00`)
    .lte("timestamp", `${today}T23:59:59`);

  const smsAlreadyToday = smsCountRows ? smsCountRows.length : 0;

  let smsOk = false;
  let emailOk = false;

  const msg = `Worker ${worker.name} (${worker.worker_id}) has ${dailyViolations} PPE violations today. Total: ${totalViolations}, Streak: ${streak}.`;

  // Send SMS if threshold not reached and mobile number available
  if (smsAlreadyToday < threshold && worker.mobile) {
    const smsRes = await sendSms(worker.mobile, msg, worker.worker_id);
    smsOk = smsRes.ok;
  }

  // Email to supervisors + worker
  const defaultRecipientsRaw = process.env.PPE_EMAIL_RECIPIENTS || "";
  const defaultRecipients = defaultRecipientsRaw
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean);

  const recipients = new Set(defaultRecipients);
  if (worker.email) {
    recipients.add(worker.email);
  }

  const subject = "PPE Violation Alert";
  const html = `
    <html>
      <body>
        <h2>PPE Violation Alert</h2>
        <p>${msg}</p>
      </body>
    </html>
  `;

  // Send email if recipients available
  if (recipients.size > 0) {
    const emailRes = await sendEmail({
      workerId: worker.worker_id,
      workerName: worker.name,
      recipients: Array.from(recipients),
      subject,
      html,
    });
    emailOk = emailRes.ok;
  }

  // Update last_notified_date if either SMS or Email was sent successfully
  if (smsOk || emailOk) {
    await updateLastNotifiedDate(worker.worker_id, today);
  }

  return { sms: smsOk, email: emailOk };
}




