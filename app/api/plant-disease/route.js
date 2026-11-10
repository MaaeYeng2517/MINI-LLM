import { spawn } from "child_process";
import path from "path";
import fs from "fs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function runPython(body) {
  return new Promise((resolve) => {
    const cwd = process.cwd();
    const script = path.join(cwd, "plant_disease_classifier", "serve.py");
    // Prefer venv Python if available, fall back to system python3
    const venvPy = path.join(cwd, ".venv", "bin", "python");
    const py = process.env.PYTHON_BIN || (fs.existsSync(venvPy) ? venvPy : "python3");
    const env = { 
      ...process.env, 
      PYTHONIOENCODING: "utf-8", 
      LANG: process.env.LANG || "en_US.UTF-8", 
      LC_ALL: process.env.LC_ALL || "en_US.UTF-8" 
    };
    const child = spawn(py, [script], { cwd, stdio: ["pipe", "pipe", "pipe"], env });
    let outBuf = Buffer.alloc(0); 
    let errBuf = Buffer.alloc(0);
    const timer = setTimeout(() => { 
      try { child.kill("SIGKILL"); } catch {} 
    }, 120000);
    child.stdout.on("data", (d) => (outBuf = Buffer.concat([outBuf, d])));
    child.stderr.on("data", (d) => (errBuf = Buffer.concat([errBuf, d])));
    child.on("close", (code) => { 
      clearTimeout(timer); 
      resolve({ code, stdout: outBuf.toString("utf8"), stderr: errBuf.toString("utf8") }); 
    });
    child.on("error", (err) => { 
      clearTimeout(timer); 
      resolve({ code: -1, stdout: "", stderr: String(err) }); 
    });
    const payload = { ...body, temperature: body?.temperature ?? 0.7, top_k: body?.top_k ?? 24, repetition_penalty: body?.repetition_penalty ?? 1.15 };
    child.stdin.write(JSON.stringify(payload || {}), "utf8");
    child.stdin.end();
  });
}

export async function POST(req) {
  try {
    const body = await req.json().catch(() => ({}));
    const { code, stdout, stderr } = await runPython(body);
    const text = (stdout || "").trim();
    let data = null; 
    try { data = text ? JSON.parse(text) : null; } catch {}
    if (!data) {
      return Response.json({ 
        ok: false, 
        error: "Python inference failed", 
        code, 
        stderr: (stderr || "").slice(-4000), 
        stdout: text.slice(-2000) 
      }, { status: 500 });
    }
    return new Response(JSON.stringify(data), { 
      status: data.ok ? 200 : 500, 
      headers: { "Content-Type": "application/json; charset=utf-8" } 
    });
  } catch (err) {
    return new Response(JSON.stringify({ ok: false, error: String(err?.message || err) }), { 
      status: 500, 
      headers: { "Content-Type": "application/json; charset=utf-8" } 
    });
  }
}