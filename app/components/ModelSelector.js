"use client";

export default function ModelSelector({ models = [], value, onChange, disabled }) {
  return (
    <label className="stack">
      Model (data/checkpoint แยกตามโมเดล)
      <select value={value || ""} onChange={(e) => onChange?.(e.target.value)} disabled={disabled}>
        {models.map((m) => (
          <option key={m.id} value={m.id}>{m.name || m.id} ({m.backend || "browser"})</option>
        ))}
      </select>
    </label>
  );
}
