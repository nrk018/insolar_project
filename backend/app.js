import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import bcrypt from "bcrypt";
import bodyParser from "body-parser";
import jwt from "jsonwebtoken";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";
import { createClient } from "@supabase/supabase-js";
import { exec } from "child_process";
import dotenv from "dotenv";
import multer from "multer";
import fs from "fs";
import {
    resolveWorkerFromName,
    upsertWorkerDetails,
    recordPpeEvent,
    listPpeWorkers,
    listDefaulters,
} from "./ppeService.js";
import { maybeNotifyForPpe } from "./notificationService.js";

dotenv.config();

const app = express();
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Supabase Connection
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_ANON_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

// Middleware
app.use(cors());
app.use(express.static(path.join(__dirname, "dist")));
app.set("view engine", "ejs");
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, "public")));
app.use(cookieParser());

// ✅ Ensure uploads folder exists
const uploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir);
}

// 🔹 Multer Configuration
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadDir);
    },
    filename: function (req, file, cb) {
        const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1E9)}`;
        const ext = path.extname(file.originalname);
        cb(null, `${file.fieldname}-${uniqueSuffix}${ext}`);
    }
});
const upload = multer({ storage });

// Middleware for token verification
const authenticateToken = (req, res, next) => {
    const token = req.cookies.token;
    if (!token) return res.sendStatus(401);

    jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
        if (err) return res.sendStatus(403);
        req.user = user;
        next();
    });
};

// ✅ User registration with Profile Photo Upload
app.post("/register", upload.array("profilePhotos"), async (req, res) => {
    try {
        const { name, email, phone, department, designation, password, employee_id } = req.body;

        // Create a new directory for the employee's profile photos
        const employeeDir = path.join(uploadDir, name);
        if (!fs.existsSync(employeeDir)) {
            fs.mkdirSync(employeeDir, { recursive: true }); // Ensure the directory is created
        }

        // Store uploaded file names
        const profilePhotos = req.files.map((file, index) => {
            const ext = path.extname(file.originalname);
            const newFileName = `${name}_${index}${ext}`;
            const newFilePath = path.join(employeeDir, newFileName);
            fs.renameSync(file.path, newFilePath);
            return newFileName;
        });

        // Check if user already exists
        const { data: existingUser, error: findError } = await supabase
            .from("users")
            .select("id")
            .or(`email.eq."${email}",phone.eq."${phone}"`);

        if (findError) {
            console.error("Supabase find user error:", findError);
            return res.status(500).json({ error: "Failed to check existing user", details: findError.message || findError });
        }

        if (existingUser && existingUser.length > 0) {
            return res.status(400).json({ error: "User already exists" });
        }

        // Hash password
        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(password, salt);

        const { error: insertError } = await supabase.from("users").insert([
            {
                employee_id: employee_id,
                name,
                email,
                phone,
                department,
                designation,
                profilephoto: `{${profilePhotos.join(",")}}`,
                password: hashedPassword,
                created_at: new Date(),
            },
        ]);

        if (insertError) {
            console.error("Supabase insert user error:", insertError);
            return res.status(500).json({ error: "Failed to register user", details: insertError.message || insertError });
        }

        // Generate JWT token
        const token = jwt.sign({ email, employee_id, designation }, process.env.JWT_SECRET);
        res.cookie("token", token, { httpOnly: true });

        // Pass designation to the frontend
        res.status(201).json({ message: "Registration Successful", designation: designation });

        // ✅ Trigger embeddings generation (keep this code AFTER successful registration)
        const scriptPath = path.join(__dirname, "../flaskServer/createEmbeddings.py");
        const uploadFolder = path.join(__dirname, "uploads", name);

        const command = `python "${scriptPath}" "${uploadFolder}"`;

        exec(command, (error, stdout, stderr) => {
            if (error) {
                console.error(`[Embedding Error]: ${error.message}`);
                return;
            }
            if (stderr) {
                console.error(`[Embedding stderr]: ${stderr}`);
            }
            console.log(`[Embeddings Log]: ${stdout}`);
        });
    } catch (error) {
        console.error("Error in registration", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

// ✅ Serve Uploaded Images
app.use("/uploads", express.static("uploads"));

// ✅ Get user's uploaded images
app.get("/api/user/images", authenticateToken, async (req, res) => {
    try {
        const { name } = req.user;
        
        if (!name) {
            return res.status(400).json({ error: "User name not found" });
        }
        
        const userUploadFolder = path.join(__dirname, "uploads", name);
        
        // Check if folder exists
        if (!fs.existsSync(userUploadFolder)) {
            return res.status(200).json({ images: [] });
        }
        
        // Get all image files (exclude embeddings.csv)
        const files = fs.readdirSync(userUploadFolder);
        const imageFiles = files.filter(file => {
            const ext = path.extname(file).toLowerCase();
            return ['.jpg', '.jpeg', '.png', '.gif', '.webp'].includes(ext);
        }).sort(); // Sort alphabetically
        
        // Create full URLs for images
        const imageUrls = imageFiles.map(file => `/uploads/${encodeURIComponent(name)}/${encodeURIComponent(file)}`);
        
        res.status(200).json({ 
            images: imageUrls,
            imageFiles: imageFiles 
        });
    } catch (error) {
        console.error("[USER IMAGES ERROR]", error);
        res.status(500).json({ error: "Failed to fetch user images" });
    }
});

// ✅ Admin: Add Employee (with image uploads)
app.post("/api/admin/add-employee", authenticateToken, upload.array("profilePhotos"), async (req, res) => {
    try {
        // Check if user is admin (req.user is set by authenticateToken middleware)
        // Check case-insensitively to handle "admin", "Admin", "ADMIN", etc.
        if (!req.user) {
            console.error("[ADMIN CHECK] req.user is missing");
            return res.status(403).json({ error: "Forbidden: Admin access required" });
        }
        
        // Case-insensitive admin check - handles "admin", "Admin", "ADMIN", etc.
        if (!req.user.designation) {
            console.error("[ADMIN CHECK] req.user.designation is missing. req.user:", req.user);
            return res.status(403).json({ error: "Forbidden: Admin access required" });
        }
        const userDesignation = req.user.designation.toString().toLowerCase().trim();
        if (userDesignation !== "admin") {
            console.error(`[ADMIN CHECK] User designation is "${req.user.designation}" (normalized: "${userDesignation}"), not admin`);
            return res.status(403).json({ error: "Forbidden: Admin access required" });
        }

        const { name, email, employee_id, department, designation, phone } = req.body;

        // Validate required fields (email is now optional)
        if (!name || !employee_id || !department || !designation) {
            return res.status(400).json({ error: "Missing required fields: name, employee_id, department, designation" });
        }

        // Validate email format if provided
        if (email) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(email)) {
                return res.status(400).json({ error: "Invalid email format" });
            }
        }

        // Check if employee_id already exists
        const { data: existingEmployee, error: findError } = await supabase
            .from("users")
            .select("employee_id, email")
            .or(`employee_id.eq."${employee_id}"${email ? `,email.eq."${email}"` : ''}`);

        if (findError) {
            console.error("Supabase find employee error:", findError);
            return res.status(500).json({ error: "Failed to check existing employee", details: findError.message });
        }

        if (existingEmployee && existingEmployee.length > 0) {
            const existing = existingEmployee[0];
            if (existing.employee_id === employee_id) {
                return res.status(400).json({ error: `Employee ID ${employee_id} already exists` });
            }
            if (email && existing.email === email) {
                return res.status(400).json({ error: `Email ${email} already exists` });
            }
        }

        // Handle image uploads
        const employeeDir = path.join(uploadDir, name);
        if (!fs.existsSync(employeeDir)) {
            fs.mkdirSync(employeeDir, { recursive: true });
        }

        let profilePhotos = [];
        if (req.files && req.files.length > 0) {
            profilePhotos = req.files.map((file, index) => {
                const ext = path.extname(file.originalname);
                const newFileName = `${name}_${index}${ext}`;
                const newFilePath = path.join(employeeDir, newFileName);
                fs.renameSync(file.path, newFilePath);
                return newFileName;
            });
        }

        // Generate default password (employee can change later)
        // Format: {name}@{employee_id} (lowercase, no spaces)
        const defaultPassword = `${name.toLowerCase().replace(/\s+/g, '')}@${employee_id}`;
        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(defaultPassword, salt);

        // Insert employee into users table
        const { error: insertError } = await supabase.from("users").insert([
            {
                employee_id: employee_id,
                name,
                email: email || null, // Email is now optional
                phone: phone || null, // Phone is optional
                department,
                designation,
                profilephoto: profilePhotos.length > 0 ? `{${profilePhotos.join(",")}}` : null,
                password: hashedPassword,
                created_at: new Date(),
            },
        ]);

        if (insertError) {
            console.error("Supabase insert employee error:", insertError);
            return res.status(500).json({ error: "Failed to add employee", details: insertError.message });
        }

        // Update worker_details table for PPE system
        const workerId = `W${String(employee_id).padStart(3, "0")}`;
        await upsertWorkerDetails({
            worker_id: workerId,
            name,
            email: email || null, // Email is optional
            mobile: phone || null,
            state: "Uttar Pradesh", // Default state
        });

        // Generate embeddings if images were uploaded
        if (profilePhotos.length > 0) {
            const scriptPath = path.join(__dirname, "../flaskServer/createEmbeddings.py");
            const uploadFolder = path.join(__dirname, "uploads", name);
            const command = `python "${scriptPath}" "${uploadFolder}"`;

            exec(command, (error, stdout, stderr) => {
                if (error) {
                    console.error(`[Embedding Error]: ${error.message}`);
                }
                if (stderr) {
                    console.error(`[Embedding stderr]: ${stderr}`);
                }
                console.log(`[Embeddings Log]: ${stdout}`);
            });
        }

        return res.status(201).json({
            message: "Employee added successfully",
            employee_id,
            name,
            email: email || null,
            default_password: defaultPassword, // Return default password for admin reference
            images_uploaded: profilePhotos.length,
        });
    } catch (error) {
        console.error("Error in add-employee:", error);
        return res.status(500).json({ error: "Internal Server Error", details: error.message });
    }
});

// ✅ Store Detection Event (called from Flask server)
app.post("/api/detections/event", async (req, res) => {
    try {
        const { worker_name, confidence, ppe_compliant, ppe_items, camera_source, snapshot_path } = req.body;

        if (!worker_name) {
            console.error("[DETECTION EVENT] Missing worker_name");
            return res.status(400).json({ error: "worker_name is required" });
        }

        // Resolve worker_id from name
        let worker_id = null;
        try {
            const worker = await resolveWorkerFromName(worker_name);
            if (worker) {
                worker_id = worker.worker_id;
            }
        } catch (err) {
            console.error("[DETECTION EVENT] Error resolving worker:", err);
            // Continue without worker_id if resolution fails
        }

        // Get current timestamp in ISO format (UTC)
        const currentTimestamp = new Date().toISOString();
        const localTime = new Date().toLocaleString();
        
        const detectionData = {
            worker_id: worker_id || null,
            worker_name,
            confidence: confidence || null,
            ppe_compliant: ppe_compliant !== undefined ? ppe_compliant : null,
            ppe_items: ppe_items || null,
            camera_source: camera_source || null,
            snapshot_path: snapshot_path || null,  // Path to snapshot image with annotations
            detected_at: currentTimestamp,
        };

        console.log("[DETECTION EVENT] Upserting:", worker_name);
        console.log("[DETECTION EVENT] Server time (UTC):", currentTimestamp);
        console.log("[DETECTION EVENT] Server time (local):", localTime);

        // Upsert: Update existing entry for this worker_name, or insert new one
        // This ensures only one entry per person, with the latest detection time
        // First, check if entry exists for this worker_name
        const { data: existing, error: checkError } = await supabase
            .from("detection_events")
            .select("id")
            .eq("worker_name", worker_name)
            .limit(1);

        let data, upsertError;
        if (existing && existing.length > 0) {
            // Update existing entry - update timestamp and other fields including snapshot_path
            const { data: updateData, error: updateError } = await supabase
                .from("detection_events")
                .update({
                    worker_id: worker_id || null,
                    confidence: confidence || null,
                    ppe_compliant: ppe_compliant !== undefined ? ppe_compliant : null,
                    ppe_items: ppe_items || null,
                    camera_source: camera_source || null,
                    snapshot_path: snapshot_path || null,  // Update snapshot_path
                    detected_at: currentTimestamp, // Use the same timestamp variable
                })
                .eq("worker_name", worker_name)
                .select();
            data = updateData;
            upsertError = updateError;
            console.log("[DETECTION EVENT] Updated existing entry for:", worker_name, "snapshot_path:", snapshot_path);
        } else {
            // Insert new entry
            const { data: insertData, error: insertError } = await supabase
                .from("detection_events")
                .insert([detectionData])
                .select();
            data = insertData;
            upsertError = insertError;
            console.log("[DETECTION EVENT] Inserted new entry for:", worker_name);
        }

        if (upsertError) {
            // If table doesn't exist, log warning but don't fail
            if (upsertError.message && upsertError.message.includes("does not exist")) {
                console.warn("[DETECTION EVENT] Table not found. Run migration: add_detection_events.sql");
                return res.status(200).json({ message: "Detection event skipped (table not found)" });
            }
            console.error("[DETECTION EVENT] Upsert error:", upsertError);
            return res.status(500).json({ error: "Failed to store detection event", details: upsertError.message });
        }

        console.log("[DETECTION EVENT] Upserted successfully:", data?.[0]?.id);
        return res.status(200).json({ message: "Detection event stored", id: data?.[0]?.id });
    } catch (error) {
        console.error("[DETECTION EVENT] Exception:", error);
        return res.status(500).json({ error: "Internal Server Error", details: error.message });
    }
});

// ✅ Get Recent Detections (only today's detections in India/Delhi timezone)
app.get("/api/detections/recent", authenticateToken, async (req, res) => {
    try {
        const limit = parseInt(req.query.limit || "20", 10);
        
        // Get current date in India/Delhi timezone (IST - UTC+5:30)
        const now = new Date();
        // IST offset: UTC+5:30 = 5 hours 30 minutes = 330 minutes
        const istOffsetMs = 5.5 * 60 * 60 * 1000;
        
        // Get current time in IST
        const istNow = new Date(now.getTime() + istOffsetMs);
        
        // Get start of today in IST (00:00:00 IST)
        const todayStartIST = new Date(Date.UTC(
            istNow.getUTCFullYear(),
            istNow.getUTCMonth(),
            istNow.getUTCDate(),
            0, 0, 0, 0
        ));
        // Convert IST start time back to UTC for database query
        const todayStartUTC = new Date(todayStartIST.getTime() - istOffsetMs);
        const todayStartISO = todayStartUTC.toISOString();
        
        // Get end of today in IST (23:59:59.999 IST)
        const todayEndIST = new Date(Date.UTC(
            istNow.getUTCFullYear(),
            istNow.getUTCMonth(),
            istNow.getUTCDate(),
            23, 59, 59, 999
        ));
        // Convert IST end time back to UTC for database query
        const todayEndUTC = new Date(todayEndIST.getTime() - istOffsetMs);
        const todayEndISO = todayEndUTC.toISOString();
        
        console.log(`[RECENT DETECTIONS] Fetching today's detections (IST)`);
        console.log(`[RECENT DETECTIONS] IST Today: ${todayStartIST.toISOString()} to ${todayEndIST.toISOString()}`);
        console.log(`[RECENT DETECTIONS] UTC Query: ${todayStartISO} to ${todayEndISO}`);
        
        const { data, error } = await supabase
            .from("detection_events")
            .select("*")
            .gte("detected_at", todayStartISO)
            .lte("detected_at", todayEndISO)
            .order("detected_at", { ascending: false })
            .limit(limit);

        if (error) {
            // If table doesn't exist, return empty array
            if (error.message && error.message.includes("does not exist")) {
                console.warn("[RECENT DETECTIONS] Table not found. Run migration: add_detection_events.sql");
                return res.status(200).json([]);
            }
            console.error("[RECENT DETECTIONS] Error:", error);
            return res.status(500).json({ error: "Failed to fetch detections", details: error.message });
        }

        console.log(`[RECENT DETECTIONS] ✅ Returning ${data?.length || 0} detections for today (IST)`);
        if (data && data.length > 0) {
            console.log(`[RECENT DETECTIONS] Sample: ${data[0].worker_name} at ${data[0].detected_at}`);
        }
        return res.status(200).json(data || []);
    } catch (error) {
        console.error("[RECENT DETECTIONS] Exception:", error);
        return res.status(500).json({ error: "Internal Server Error", details: error.message });
    }
});

// ✅ Admin: List All Employees
app.get("/api/admin/employees", authenticateToken, async (req, res) => {
    try {
        // Check if user is admin (req.user is set by authenticateToken middleware)
        // Check case-insensitively to handle "admin", "Admin", "ADMIN", etc.
        if (!req.user) {
            console.error("[ADMIN CHECK] req.user is missing");
            return res.status(403).json({ error: "Forbidden: Admin access required" });
        }
        
        // Case-insensitive admin check - handles "admin", "Admin", "ADMIN", etc.
        if (!req.user.designation) {
            console.error("[ADMIN CHECK] req.user.designation is missing. req.user:", req.user);
            return res.status(403).json({ error: "Forbidden: Admin access required" });
        }
        const userDesignation = req.user.designation.toString().toLowerCase().trim();
        if (userDesignation !== "admin") {
            console.error(`[ADMIN CHECK] User designation is "${req.user.designation}" (normalized: "${userDesignation}"), not admin`);
            return res.status(403).json({ error: "Forbidden: Admin access required" });
        }

        const { data, error } = await supabase
            .from("users")
            .select("employee_id, name, email, phone, department, designation, profilephoto, created_at")
            .order("created_at", { ascending: false });

        if (error) {
            console.error("Supabase fetch employees error:", error);
            return res.status(500).json({ error: "Failed to fetch employees", details: error.message });
        }

        // Get first image from uploads folder for each employee
        const employees = await Promise.all((data || []).map(async (emp) => {
            let profilePhoto = [];
            try {
                const userUploadFolder = path.join(__dirname, "uploads", emp.name);
                if (fs.existsSync(userUploadFolder)) {
                    const files = fs.readdirSync(userUploadFolder);
                    const imageFiles = files.filter(file => {
                        const ext = path.extname(file).toLowerCase();
                        return ['.jpg', '.jpeg', '.png', '.gif', '.webp'].includes(ext);
                    }).sort();
                    
                    if (imageFiles.length > 0) {
                        const firstImage = imageFiles[0];
                        profilePhoto = [`/uploads/${encodeURIComponent(emp.name)}/${encodeURIComponent(firstImage)}`];
                    }
                }
            } catch (err) {
                console.error(`Error getting profile photo for ${emp.name}:`, err);
            }
            
            return {
                ...emp,
                profilephoto: profilePhoto
            };
        }));

        return res.status(200).json(employees);
    } catch (error) {
        console.error("Error in list-employees:", error);
        return res.status(500).json({ error: "Internal Server Error", details: error.message });
    }
});

// User login
app.post("/login", async (req, res) => {
    try {
        const { email, password } = req.body;

        // Find user in Supabase
        const { data: users, error: findError } = await supabase
            .from("users")
            .select("id, email, password, designation") // Added designation to the select clause
            .eq("email", email)
            .limit(1);

        if (!users || users.length === 0) {
            return res.status(401).json({ error: "Invalid email or password" });
        }

        const user = users[0];

        // Compare password
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return res.status(401).json({ error: "Invalid email or password" });
        }

        // Generate token
        const token = jwt.sign({ email: user.email, userid: user.id, designation: user.designation }, process.env.JWT_SECRET); // Added designation to the token
        res.cookie("token", token, { httpOnly: true });

        res.status(200).json({ message: "Login Successful", designation: user.designation }); // Added designation to the response
    } catch (error) {
        console.error("Error in login:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

app.get("/api/user/name", authenticateToken, async (req, res) => {
    try {
        const { email } = req.user;

        const { data, error } = await supabase
            .from("users")
            .select("name")
            .eq("email", email)
            .limit(1);

        if (error) {
            console.error("Error fetching name:", error);
            return res.status(500).json({ error: "Internal Server Error" });
        }

        if (!data || data.length === 0) {
            return res.status(404).json({ error: "User not found" });
        }

        res.status(200).json({ name: data[0].name });

    } catch (error) {
        console.error("Error fetching name:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});


app.get("/api/user/profilePhoto", authenticateToken, async (req, res) => {
    try {
        const { name } = req.user;
        
        if (!name) {
            return res.status(400).json({ error: "User name not found" });
        }
        
        // Get first image from uploads folder
        const userUploadFolder = path.join(__dirname, "uploads", name);
        
        if (!fs.existsSync(userUploadFolder)) {
            return res.status(200).json({ profilePhoto: [] });
        }
        
        // Get all image files (exclude embeddings.csv)
        const files = fs.readdirSync(userUploadFolder);
        const imageFiles = files.filter(file => {
            const ext = path.extname(file).toLowerCase();
            return ['.jpg', '.jpeg', '.png', '.gif', '.webp'].includes(ext);
        }).sort(); // Sort alphabetically
        
        // Return first image if available
        if (imageFiles.length > 0) {
            const firstImage = imageFiles[0];
            const imageUrl = `/uploads/${encodeURIComponent(name)}/${encodeURIComponent(firstImage)}`;
            return res.status(200).json({ profilePhoto: [imageUrl] });
        }
        
        return res.status(200).json({ profilePhoto: [] });
    } catch (error) {
        console.error("Error fetching profile photo:", error);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

app.get("/api/user", authenticateToken, async (req, res) => {
    try {
        const { email } = req.user;
        const { data, error } = await supabase
            .from("users")
            .select("name, employee_id, email, phone, department, designation, profilephoto")
            .eq("email", email)
            .limit(1);

        if (error) {
            console.error("Error fetching user data:", error);
            res.status(500).json({ error: "Internal Server Error" });
        } else {
            // Log designation for debugging
            const designation = data[0]?.designation;
            console.log(`[API /api/user] User designation: "${designation}" (type: ${typeof designation})`);
            
            // Get first image from uploads folder
            let profilePhoto = [];
            try {
                const userUploadFolder = path.join(__dirname, "uploads", data[0].name);
                if (fs.existsSync(userUploadFolder)) {
                    const files = fs.readdirSync(userUploadFolder);
                    const imageFiles = files.filter(file => {
                        const ext = path.extname(file).toLowerCase();
                        return ['.jpg', '.jpeg', '.png', '.gif', '.webp'].includes(ext);
                    }).sort();
                    
                    if (imageFiles.length > 0) {
                        const firstImage = imageFiles[0];
                        profilePhoto = [`/uploads/${encodeURIComponent(data[0].name)}/${encodeURIComponent(firstImage)}`];
                    }
                }
            } catch (err) {
                console.error(`Error getting profile photo for ${data[0].name}:`, err);
            }
            
            const user = {
                ...data[0],
                profilePhoto: profilePhoto,
            };
            res.status(200).json(user);
        }
    } catch (error) {
        console.error("Error fetching user data:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

app.post("/api/updateProfilePhoto", authenticateToken, upload.single("profilePhoto"), async (req, res) => {
    try {
        const { email } = req.user;

        // Fetch user details to get the old profile photo
        const { data: userData, error: userError } = await supabase
            .from("users")
            .select("name, profilephoto")
            .eq("email", email)
            .single();

        if (userError || !userData) {
            console.error("Error fetching user data:", userError);
            return res.status(404).json({ error: "User not found" });
        }

        const userName = userData.name;
        const oldPhoto = userData.profilephoto;
        const uploadsDir = path.join(__dirname, "/uploads");

        // Delete old profile photo if it exists
        if (oldPhoto) {
            const oldPhotoPath = path.join(uploadsDir, oldPhoto);
            if (fs.existsSync(oldPhotoPath)) {
                fs.unlinkSync(oldPhotoPath);
            }
        }

        // Save new profile photo with the user's name
        const fileExtension = path.extname(req.file.originalname);
        const newPhotoName = `${userName}${fileExtension}`;
        const newPhotoPath = path.join(uploadsDir, newPhotoName);

        // Rename uploaded file
        fs.renameSync(req.file.path, newPhotoPath);

        // Update new photo name in Supabase
        const { error: updateError } = await supabase
            .from("users")
            .update({ profilephoto: newPhotoName })
            .eq("email", email);

        if (updateError) {
            console.error("Error updating profile photo in database:", updateError);
            return res.status(500).json({ error: "Internal Server Error" });
        }

        res.status(200).json({ message: "Profile photo updated successfully", profilePhoto: newPhotoName });
    } catch (error) {
        console.error("Unexpected error:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

app.patch("/api/users/update", authenticateToken, async (req, res) => {
    try {
        const { email } = req.user;
        const { name, phone, oldPassword, newPassword } = req.body;

        // Fetch user details
        const { data: userData, error: userError } = await supabase
            .from("users")
            .select("password")
            .eq("email", email)
            .single();

        if (userError || !userData) {
            console.error("Error fetching user data:", userError);
            return res.status(404).json({ error: "User not found" });
        }

        let updateFields = { name, phone };

        // If the user wants to change the password
        if (oldPassword && newPassword) {
            // Decrypt the stored password
            const isMatch = await bcrypt.compare(oldPassword, userData.password);
            if (!isMatch) {
                return res.status(401).json({ error: "Old password is incorrect" });
            }

            // If old password is correct, update with new password
            updateFields.password = await hashPassword(newPassword);
        }

        // Update user details
        const { error: updateError } = await supabase
            .from("users")
            .update(updateFields)
            .eq("email", email);

        if (updateError) {
            console.error("Error updating user data:", updateError);
            return res.status(500).json({ error: "Internal Server Error" });
        }

        res.status(200).json({ message: "User details updated successfully" });
    } catch (error) {
        console.error("Unexpected error:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

async function hashPassword(password) {
    const salt = await bcrypt.genSalt(10);
    return await bcrypt.hash(password, salt);
}

// ---------------- PPE SAFETY ROUTES (replace attendance system) ----------------

// Ingest PPE event from Python faceRecognition service
app.post("/api/ppe/event", async (req, res) => {
    try {
        const { name, ppe_compliant, ppe_items, ppe_confidence } = req.body;
        if (!name) {
            return res.status(400).json({ error: "name is required" });
        }

        const worker = await resolveWorkerFromName(name);
        if (!worker) {
            return res.status(404).json({ error: "Worker not found for given name" });
        }

        // Ensure worker_details row exists
        await upsertWorkerDetails({
            ...worker,
            state: req.body.state || "Uttar Pradesh",
        });

        const ppeRecord = await recordPpeEvent({
            worker,
            ppeCompliant: !!ppe_compliant,
            ppeItems: ppe_items || {},
            ppeConfidence: ppe_confidence || 0.0,
        });

        const notif = await maybeNotifyForPpe({
            worker,
            dailyViolations: ppeRecord.daily_violations,
            totalViolations: ppeRecord.total_violations,
            streak: ppeRecord.streak,
        });

        return res.status(200).json({
            ...ppeRecord,
            sms_sent: notif.sms,
            email_sent: notif.email,
        });
    } catch (error) {
        console.error("PPE /api/ppe/event error:", error);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

// List latest PPE records for dashboard
app.get("/api/ppe/workers", authenticateToken, async (req, res) => {
    try {
        const records = await listPpeWorkers();
        return res.status(200).json(records);
    } catch (error) {
        console.error("PPE /api/ppe/workers error:", error);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

// Defaulters (high streak of violations)
app.get("/api/ppe/defaulters", authenticateToken, async (req, res) => {
    try {
        const minStreak = parseInt(req.query.minStreak || "3", 10);
        const rows = await listDefaulters({ minStreak });
        return res.status(200).json(rows);
    } catch (error) {
        console.error("PPE /api/ppe/defaulters error:", error);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

// List notifications for safety dashboard
app.get("/api/ppe/notifications", authenticateToken, async (req, res) => {
    try {
        const { data, error } = await supabase
            .from("notifications")
            .select("*")
            .order("timestamp", { ascending: false })
            .limit(200);
        if (error) {
            console.error("PPE /api/ppe/notifications error:", error);
            return res.status(500).json({ error: "Internal Server Error" });
        }
        return res.status(200).json(data || []);
    } catch (error) {
        console.error("PPE /api/ppe/notifications error:", error);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

// User logout
app.post("/logout", (req, res) => {
    res.cookie("token", "", { httpOnly: true, expires: new Date(0) });
    res.redirect("/");
});

// Catch-all route to serve the frontend
app.get("*", (req, res) => {
    res.sendFile(path.join(__dirname, "dist", "index.html"));
});

// Start the server
app.listen(3000, () => console.log("Server started on port 3000"));
