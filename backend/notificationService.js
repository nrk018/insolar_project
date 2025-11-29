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

  console.log(`[EMAIL SEND] Attempting to send email for worker ${workerId} to ${recipients.length} recipient(s): ${recipients.join(", ")}`);

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

    console.log(`[EMAIL SEND SUCCESS] Email sent successfully for worker ${workerId} to ${recipients.length} recipient(s)`);

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
    console.error(`[EMAIL SEND FAILED] Failed to send email for worker ${workerId}: ${reason}`);
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
  ppeItems = {}, // PPE items status: { helmet: true/false, gloves: true/false, boots: true/false, jacket: true/false }
}) {
  if (dailyViolations <= 0) {
    return { sms: false, email: false };
  }

  const today = new Date().toISOString().split("T")[0];

  // FIRST: Check notifications table to see if email was already sent today
  // This is more reliable than checking last_notified_date due to race conditions
  const { data: emailNotifications, error: emailCheckErr } = await supabase
    .from("notifications")
    .select("id")
    .eq("worker_id", worker.worker_id)
    .eq("type", "Email")
    .eq("status", "Sent")
    .gte("timestamp", `${today}T00:00:00`)
    .lte("timestamp", `${today}T23:59:59`)
    .limit(1);

  if (emailCheckErr) {
    console.error("Error checking email notifications:", emailCheckErr);
  }

  // If email was already sent today, skip notification
  if (emailNotifications && emailNotifications.length > 0) {
    console.log(
      `[DUPLICATE PREVENTION] Worker ${worker.worker_id} already received email today (${today}). Found ${emailNotifications.length} sent email(s) in notifications table. Skipping.`
    );
    return { sms: false, email: false, reason: "Email already sent today" };
  }

  // SECOND: Check last_notified_date as additional safeguard
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
      `[DUPLICATE PREVENTION] Worker ${worker.worker_id} already notified today (${today}) - last_notified_date check. Skipping notification.`
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

  // Determine which PPE items are missing
  const missingItems = [];
  const itemNames = {
    helmet: "Helmet",
    gloves: "Gloves",
    boots: "Boots",
    jacket: "Jacket"
  };

  // Check each PPE item - if not detected (false or undefined), it's missing
  if (ppeItems.helmet !== true) missingItems.push("Helmet");
  if (ppeItems.gloves !== true) missingItems.push("Gloves");
  if (ppeItems.boots !== true) missingItems.push("Boots");
  if (ppeItems.jacket !== true) missingItems.push("Jacket");

  // Format violation message
  let violationMessage = "";
  if (missingItems.length === 1) {
    violationMessage = `1 violation of not wearing ${missingItems[0]}`;
  } else if (missingItems.length > 1) {
    const itemsList = missingItems.slice(0, -1).join(", ") + " and " + missingItems[missingItems.length - 1];
    violationMessage = `${missingItems.length} violations of not wearing ${itemsList}`;
  } else {
    violationMessage = "PPE violation detected";
  }

  const msg = `Worker ${worker.name} (${worker.worker_id}) has ${dailyViolations} PPE violations today. Total: ${totalViolations}, Streak: ${streak}.`;

  // Send SMS if threshold not reached and mobile number available
  if (smsAlreadyToday < threshold && worker.mobile) {
    const smsRes = await sendSms(worker.mobile, msg, worker.worker_id);
    smsOk = smsRes.ok;
  }

  // Email ONLY to the worker (person) - not to supervisors
  // Only send email if worker has an email address
  const recipients = new Set();
  if (worker.email) {
    recipients.add(worker.email);
  }

  // Personalized email message
  const currentTime = new Date().toLocaleString('en-US', { 
    weekday: 'long', 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric', 
    hour: '2-digit', 
    minute: '2-digit' 
  });

  // Format the main message as requested: "Hello [name]! Violation time, not wearing [items]"
  const notWearingList = missingItems.length > 0 ? missingItems.join(", ") : "Unknown items";
  const mainMessage = `Hello ${worker.name}! Violation time, not wearing ${notWearingList}`;

  const subject = "PPE Violation Alert";
  const html = `
    <html>
      <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
          <h2 style="color: #d32f2f;">PPE Violation Alert</h2>
          <p style="font-size: 16px;">${mainMessage}</p>
          <p style="font-size: 14px; color: #666; margin-top: 10px;">
            <strong>Time:</strong> ${currentTime}
          </p>
          <p style="font-size: 14px; color: #d32f2f; font-weight: bold; margin-top: 15px;">
            ${violationMessage}
          </p>
          <hr style="border: 1px solid #eee; margin: 20px 0;">
          <p style="color: #666; font-size: 14px;">
            Please ensure you wear all required PPE items: Helmet, Gloves, Boots, and Jacket for your safety.
          </p>
        </div>
      </body>
    </html>
  `;

  // Send email if recipients available
  // FINAL CHECK: One more verification right before sending to prevent race conditions
  if (recipients.size > 0) {
    // Final check of notifications table right before sending (prevents race condition)
    const { data: finalEmailCheck, error: finalCheckErr } = await supabase
      .from("notifications")
      .select("id")
      .eq("worker_id", worker.worker_id)
      .eq("type", "Email")
      .eq("status", "Sent")
      .gte("timestamp", `${today}T00:00:00`)
      .lte("timestamp", `${today}T23:59:59`)
      .limit(1);

    if (finalCheckErr) {
      console.error("[NOTIFICATION] Error in final email check:", finalCheckErr);
    }

    if (finalEmailCheck && finalEmailCheck.length > 0) {
      console.log(
        `[DUPLICATE PREVENTION] Final check: Worker ${worker.worker_id} already has sent email in notifications table. Skipping email send.`
      );
      emailOk = false; // Don't send, but don't return early (SMS might still be sent)
    } else {
      // All checks passed, send email
      console.log(`[NOTIFICATION] Sending email to ${recipients.size} recipient(s) for worker ${worker.worker_id}`);
      const emailRes = await sendEmail({
        workerId: worker.worker_id,
        workerName: worker.name,
        recipients: Array.from(recipients),
        subject,
        html,
      });
      emailOk = emailRes.ok;
    }
  }

  // Update last_notified_date if either SMS or Email was sent successfully
  if (smsOk || emailOk) {
    await updateLastNotifiedDate(worker.worker_id, today);
  }

  // Log final result
  if (emailOk) {
    console.log(`[NOTIFICATION COMPLETE] Successfully sent email notification for worker ${worker.worker_id} on ${today}`);
  } else if (recipients.size > 0) {
    console.log(`[NOTIFICATION SKIPPED] Email notification was skipped for worker ${worker.worker_id} on ${today} (likely duplicate prevention)`);
  }

  return { sms: smsOk, email: emailOk };
}

// Manual email send function - bypasses duplicate checks (for admin manual send)
export async function sendManualEmail({
  worker,
  ppeItems = {},
}) {
  const today = new Date().toISOString().split("T")[0];

  // Determine which PPE items are missing
  const missingItems = [];
  if (ppeItems.helmet !== true) missingItems.push("Helmet");
  if (ppeItems.gloves !== true) missingItems.push("Gloves");
  if (ppeItems.boots !== true) missingItems.push("Boots");
  if (ppeItems.jacket !== true) missingItems.push("Jacket");

  // Format violation message
  let violationMessage = "";
  if (missingItems.length === 1) {
    violationMessage = `1 violation of not wearing ${missingItems[0]}`;
  } else if (missingItems.length > 1) {
    const itemsList = missingItems.slice(0, -1).join(", ") + " and " + missingItems[missingItems.length - 1];
    violationMessage = `${missingItems.length} violations of not wearing ${itemsList}`;
  } else {
    violationMessage = "PPE violation detected";
  }

  // Email ONLY to the worker (person) - not to supervisors
  const recipients = new Set();
  if (worker.email) {
    recipients.add(worker.email);
  } else {
    return { ok: false, reason: "Worker has no email address" };
  }

  // Personalized email message
  const currentTime = new Date().toLocaleString('en-US', { 
    weekday: 'long', 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric', 
    hour: '2-digit', 
    minute: '2-digit' 
  });

  const notWearingList = missingItems.length > 0 ? missingItems.join(", ") : "Unknown items";
  const mainMessage = `Hello ${worker.name}! Violation time, not wearing ${notWearingList}`;

  const subject = "PPE Violation Alert (Manual)";
  const html = `
    <html>
      <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
          <h2 style="color: #d32f2f;">PPE Violation Alert</h2>
          <p style="font-size: 16px;">${mainMessage}</p>
          <p style="font-size: 14px; color: #666; margin-top: 10px;">
            <strong>Time:</strong> ${currentTime}
          </p>
          <p style="font-size: 14px; color: #d32f2f; font-weight: bold; margin-top: 15px;">
            ${violationMessage}
          </p>
          <hr style="border: 1px solid #eee; margin: 20px 0;">
          <p style="color: #666; font-size: 14px;">
            Please ensure you wear all required PPE items: Helmet, Gloves, Boots, and Jacket for your safety.
          </p>
        </div>
      </body>
    </html>
  `;

  // Send email (bypass duplicate checks for manual send)
  console.log(`[MANUAL EMAIL] Sending email to ${worker.name} (${worker.worker_id}) - bypassing duplicate checks`);
  const emailRes = await sendEmail({
    workerId: worker.worker_id,
    workerName: worker.name,
    recipients: Array.from(recipients),
    subject,
    html,
  });

  // Update last_notified_date if email was sent successfully
  if (emailRes.ok) {
    await updateLastNotifiedDate(worker.worker_id, today);
  }

  return { ok: emailRes.ok, reason: emailRes.reason };
}




