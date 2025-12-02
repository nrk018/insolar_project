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

// Helper function to format phone number for SMSIdea API
function formatPhoneNumber(phone) {
  if (!phone) return null;
  
  // Remove all non-digit characters (spaces, dashes, +, etc.)
  let cleaned = phone.toString().replace(/\D/g, '');
  
  // Remove country code if present (91 for India)
  if (cleaned.startsWith('91') && cleaned.length === 12) {
    cleaned = cleaned.substring(2);
  }
  
  // Validate: Should be 10 digits and start with 6, 7, 8, or 9
  if (cleaned.length === 10 && /^[6-9]/.test(cleaned)) {
    return cleaned;
  }
  
  // If not valid, return null
  console.warn(`[SMS] Invalid phone number format: ${phone} (cleaned: ${cleaned})`);
  return null;
}

async function sendSms(toNumber, message, workerId) {
  const url = "https://www.smsidea.co.in/sendbulksms.aspx";

  // Format phone number before sending
  const formattedNumber = formatPhoneNumber(toNumber);
  if (!formattedNumber) {
    const reason = `Invalid phone number format: ${toNumber}. Must be 10-digit Indian mobile number.`;
    console.error(`[SMS ERROR] ${reason}`);
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

  const mobile = process.env.SMSIDEA_USERNAME;
  const password = process.env.SMSIDEA_PASSWORD;
  const senderid = process.env.SMSIDEA_SENDER_ID;

  // Check if SMS credentials are configured
  if (!mobile || !password || !senderid) {
    const missing = [];
    if (!mobile) missing.push("SMSIDEA_USERNAME");
    if (!password) missing.push("SMSIDEA_PASSWORD");
    if (!senderid) missing.push("SMSIDEA_SENDER_ID");
    console.error(`[SMS ERROR] Missing SMS credentials in .env file: ${missing.join(", ")}`);
    await logNotification({
      worker_id: workerId,
      type: "SMS",
      recipient: toNumber,
      message,
      status: "Failed",
      reason: `Missing credentials: ${missing.join(", ")}`,
    });
    return { ok: false, reason: `Missing SMS credentials: ${missing.join(", ")}` };
  }

  console.log(`[SMS SEND] Original number: ${toNumber}, Formatted: ${formattedNumber}`);
  console.log(`[SMS SEND] Sending SMS to ${formattedNumber} for worker ${workerId}`);
  console.log(`[SMS SEND] Using sender ID: ${senderid}`);

  const payload = {
    mobile,
    password,
    senderid,
    msgtype: "uc",
    message: [
      {
        text: message,
        to: formattedNumber, // Use formatted number
        scheduledate: "",
      },
    ],
  };

  try {
    console.log(`[SMS SEND] Calling SMS API: ${url}`);
    console.log(`[SMS SEND] Payload:`, JSON.stringify(payload, null, 2));
    const response = await axios.get(url, {
      params: { data: JSON.stringify(payload) },
      timeout: 10000, // Increased timeout to 10 seconds
    });
    const text = String(response.data || "").trim();
    console.log(`[SMS RESPONSE] Status: ${response.status}, Response: ${text}`);
    console.log(`[SMS RESPONSE] Full response data:`, response.data);

    // Check for success indicators
    const successIndicators = [
      "000 : success",
      "1 SMS Sent",
      "success",
      "sent successfully",
      "message sent"
    ];
    
    const isSuccess = response.status === 200 && 
      successIndicators.some(indicator => text.toLowerCase().includes(indicator.toLowerCase()));

    if (isSuccess) {
      console.log(`[SMS SUCCESS] SMS sent successfully to ${formattedNumber} (original: ${toNumber})`);
      await logNotification({
        worker_id: workerId,
        type: "SMS",
        recipient: toNumber,
        message,
        status: "Sent",
        reason: `Sent successfully - API Response: ${text}`,
      });
      return { ok: true, reason: "Sent successfully" };
    }

    // Check for common error codes
    let errorReason = text || "Unknown error";
    if (text.includes("Invalid") || text.includes("invalid")) {
      errorReason = `Invalid credentials or parameters: ${text}`;
    } else if (text.includes("balance") || text.includes("Balance")) {
      errorReason = `Insufficient balance: ${text}`;
    } else if (text.includes("sender") || text.includes("Sender")) {
      errorReason = `Invalid sender ID: ${text}`;
    } else if (text.includes("mobile") || text.includes("Mobile")) {
      errorReason = `Invalid mobile number: ${text}`;
    }

    console.error(`[SMS FAILED] API returned error: ${errorReason}`);
    await logNotification({
      worker_id: workerId,
      type: "SMS",
      recipient: toNumber,
      message,
      status: "Failed",
      reason: errorReason,
    });
    return { ok: false, reason: errorReason };
  } catch (err) {
    const reason = err.message || "Network error";
    console.error(`[SMS ERROR] Exception while sending SMS: ${reason}`);
    console.error(`[SMS ERROR] Full error:`, err);
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

  // Remove duplicates from recipients array to ensure we only send one email per unique address
  const uniqueRecipients = [...new Set(recipients.map(r => r.toLowerCase().trim()))];
  
  if (uniqueRecipients.length === 0) {
    return { ok: false, reason: "No valid recipients after deduplication" };
  }

  const user = process.env.EMAIL_USER;
  const pass = process.env.EMAIL_PASS;

  console.log(`[EMAIL SEND] Attempting to send email for worker ${workerId} to ${uniqueRecipients.length} unique recipient(s): ${uniqueRecipients.join(", ")}`);

  const transporter = nodemailer.createTransport({
    host: "smtp.gmail.com",
    port: 587,
    secure: false,
    auth: { user, pass },
  });

  try {
    // Send ONE email to all unique recipients
    await transporter.sendMail({
      from: user,
      to: uniqueRecipients.join(","),
      subject,
      html,
    });

    console.log(`[EMAIL SEND SUCCESS] Email sent successfully for worker ${workerId} to ${uniqueRecipients.length} unique recipient(s)`);

    // Log notification for each unique recipient
    for (const r of uniqueRecipients) {
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
    for (const r of uniqueRecipients) {
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
  // Check for ANY email notification (Sent, Failed, or Pending) to prevent duplicates
  // This is more reliable than checking last_notified_date due to race conditions
  const { data: emailNotifications, error: emailCheckErr } = await supabase
    .from("notifications")
    .select("id, status, timestamp")
    .eq("worker_id", worker.worker_id)
    .eq("type", "Email")
    .gte("timestamp", `${today}T00:00:00`)
    .lte("timestamp", `${today}T23:59:59`)
    .order("timestamp", { ascending: false })
    .limit(5);

  if (emailCheckErr) {
    console.error("Error checking email notifications:", emailCheckErr);
  }

  // If email was already sent today (status = "Sent"), skip notification
  if (emailNotifications && emailNotifications.length > 0) {
    const sentEmails = emailNotifications.filter(n => n.status === "Sent");
    if (sentEmails.length > 0) {
      console.log(
        `[DUPLICATE PREVENTION] Worker ${worker.worker_id} already received email today (${today}). Found ${sentEmails.length} sent email(s) in notifications table. Skipping.`
      );
      return { sms: false, email: false, reason: "Email already sent today" };
    }
    // Also check if there's a recent pending email (within last 2 minutes) to prevent race conditions
    const recentPending = emailNotifications.filter(n => {
      if (n.status !== "Sent") {
        const notifTime = new Date(n.timestamp);
        const now = new Date();
        const diffMinutes = (now - notifTime) / (1000 * 60);
        return diffMinutes < 2; // Within last 2 minutes
      }
      return false;
    });
    if (recentPending.length > 0) {
      console.log(
        `[DUPLICATE PREVENTION] Worker ${worker.worker_id} has a recent email notification being processed. Skipping to prevent duplicates.`
      );
      return { sms: false, email: false, reason: "Email notification already in progress" };
    }
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

  // Check if SMS was already sent today (similar to email duplicate prevention)
  const { data: smsNotifications, error: smsCheckErr } = await supabase
    .from("notifications")
    .select("id, status, timestamp")
    .eq("worker_id", worker.worker_id)
    .eq("type", "SMS")
    .eq("status", "Sent")
    .gte("timestamp", `${today}T00:00:00`)
    .lte("timestamp", `${today}T23:59:59`)
    .limit(1);

  if (smsCheckErr) {
    console.error("Error checking SMS notifications:", smsCheckErr);
  }

  // Format SMS message to match email format: "Hello [name]! Violation time, not wearing [items]"
  const notWearingList = missingItems.length > 0 ? missingItems.join(", ") : "Unknown items";
  const smsMessage = `Hello ${worker.name}! Violation time, not wearing ${notWearingList}`;

  // Send SMS only if not already sent today and mobile number is available
  console.log(`[SMS CHECK] Worker ${worker.worker_id}: SMS already sent today: ${smsNotifications && smsNotifications.length > 0}, Has mobile: ${!!worker.mobile}, Mobile number: ${worker.mobile || 'N/A'}`);
  
  if (smsNotifications && smsNotifications.length > 0) {
    console.log(`[SMS SKIP] Worker ${worker.worker_id} already received SMS today (${today}). Skipping SMS.`);
  } else if (worker.mobile) {
    // Final check right before sending to prevent race conditions
    const { data: finalSmsCheck, error: finalSmsCheckErr } = await supabase
      .from("notifications")
      .select("id, status, timestamp")
      .eq("worker_id", worker.worker_id)
      .eq("type", "SMS")
      .eq("status", "Sent")
      .gte("timestamp", `${today}T00:00:00`)
      .lte("timestamp", `${today}T23:59:59`)
      .limit(1);

    if (finalSmsCheckErr) {
      console.error("[SMS] Error in final SMS check:", finalSmsCheckErr);
    }

    if (finalSmsCheck && finalSmsCheck.length > 0) {
      console.log(`[SMS SKIP] Final check: Worker ${worker.worker_id} already has sent SMS in notifications table. Skipping SMS send.`);
    } else {
      console.log(`[SMS SEND] Attempting to send SMS to ${worker.mobile} for worker ${worker.worker_id}`);
      const smsRes = await sendSms(worker.mobile, smsMessage, worker.worker_id);
      smsOk = smsRes.ok;
      if (smsOk) {
        console.log(`[SMS SUCCESS] SMS sent successfully to ${worker.mobile} for worker ${worker.worker_id}`);
      } else {
        console.error(`[SMS FAILED] Failed to send SMS to ${worker.mobile} for worker ${worker.worker_id}: ${smsRes.reason}`);
      }
    }
  } else {
    console.log(`[SMS SKIP] Worker ${worker.worker_id} has no mobile number`);
  }

  // Email ONLY to the worker (person) - not to supervisors
  // Only send email if worker has an email address
  // Normalize email addresses (lowercase, trim) to prevent duplicates
  const recipients = new Set();
  if (worker.email) {
    const normalizedEmail = worker.email.toLowerCase().trim();
    if (normalizedEmail) {
      recipients.add(normalizedEmail);
    }
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
  // Reuse notWearingList already declared above for SMS
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
    // Check for any sent emails today OR recent pending emails
    const { data: finalEmailCheck, error: finalCheckErr } = await supabase
      .from("notifications")
      .select("id, status, timestamp")
      .eq("worker_id", worker.worker_id)
      .eq("type", "Email")
      .gte("timestamp", `${today}T00:00:00`)
      .lte("timestamp", `${today}T23:59:59`)
      .order("timestamp", { ascending: false })
      .limit(5);

    if (finalCheckErr) {
      console.error("[NOTIFICATION] Error in final email check:", finalCheckErr);
    }

    let shouldSkip = false;
    if (finalEmailCheck && finalEmailCheck.length > 0) {
      // Check for sent emails
      const sentEmails = finalEmailCheck.filter(n => n.status === "Sent");
      if (sentEmails.length > 0) {
        console.log(
          `[DUPLICATE PREVENTION] Final check: Worker ${worker.worker_id} already has sent email in notifications table. Skipping email send.`
        );
        shouldSkip = true;
      } else {
        // Check for recent pending emails (within last 2 minutes)
        const recentPending = finalEmailCheck.filter(n => {
          if (n.status !== "Sent") {
            const notifTime = new Date(n.timestamp);
            const now = new Date();
            const diffMinutes = (now - notifTime) / (1000 * 60);
            return diffMinutes < 2;
          }
          return false;
        });
        if (recentPending.length > 0) {
          console.log(
            `[DUPLICATE PREVENTION] Final check: Worker ${worker.worker_id} has recent email notification being processed. Skipping email send.`
          );
          shouldSkip = true;
        }
      }
    }

    if (shouldSkip) {
      emailOk = false; // Don't send, but don't return early (SMS might still be sent)
    } else {
      // All checks passed, send email
      console.log(`[NOTIFICATION] Sending email to ${recipients.size} unique recipient(s) for worker ${worker.worker_id}`);
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
  // Normalize email addresses (lowercase, trim) to prevent duplicates
  const recipients = new Set();
  if (worker.email) {
    const normalizedEmail = worker.email.toLowerCase().trim();
    if (normalizedEmail) {
      recipients.add(normalizedEmail);
    }
  }
  
  if (recipients.size === 0) {
    return { ok: false, reason: "Worker has no valid email address" };
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

// Manual SMS send function - bypasses duplicate checks (for admin manual send)
export async function sendManualSms({
  worker,
  ppeItems = {},
}) {
  // Determine which PPE items are missing
  const missingItems = [];
  if (ppeItems.helmet !== true) missingItems.push("Helmet");
  if (ppeItems.gloves !== true) missingItems.push("Gloves");
  if (ppeItems.boots !== true) missingItems.push("Boots");
  if (ppeItems.jacket !== true) missingItems.push("Jacket");

  // Format SMS message to match email format: "Hello [name]! Violation time, not wearing [items]"
  const notWearingList = missingItems.length > 0 ? missingItems.join(", ") : "Unknown items";
  const smsMessage = `Hello ${worker.name}! Violation time, not wearing ${notWearingList}`;

  // Check if worker has mobile number
  if (!worker.mobile) {
    return { ok: false, reason: "Worker has no mobile number" };
  }

  // Format and validate phone number
  const formattedMobile = formatPhoneNumber(worker.mobile);
  if (!formattedMobile) {
    return { ok: false, reason: `Invalid phone number format: ${worker.mobile}. Must be 10-digit Indian mobile number.` };
  }

  // Send SMS (bypass duplicate checks for manual send)
  console.log(`[MANUAL SMS] Sending SMS to ${worker.name} (${worker.worker_id}) - bypassing duplicate checks`);
  console.log(`[MANUAL SMS] Original number: ${worker.mobile}, Formatted: ${formattedMobile}`);
  const smsRes = await sendSms(worker.mobile, smsMessage, worker.worker_id);

  return { ok: smsRes.ok, reason: smsRes.reason };
}




