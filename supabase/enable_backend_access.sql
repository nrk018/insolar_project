-- Optional: only needed if you want direct client-side access to Supabase.
-- The Node.js backend uses the service_role key and bypasses RLS.
-- Run this in Supabase SQL Editor if you prefer anon-key access instead.

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE attendance ENABLE ROW LEVEL SECURITY;

-- Allow backend/public registration reads (adjust for your security model)
CREATE POLICY "Allow public read users" ON users
    FOR SELECT USING (true);

CREATE POLICY "Allow public insert users" ON users
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow public read attendance" ON attendance
    FOR SELECT USING (true);

CREATE POLICY "Allow public insert attendance" ON attendance
    FOR INSERT WITH CHECK (true);
