import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Bar, ReferenceLine } from "recharts";

const data = [
  { step: 500, rmsd: 1.94, q: 0.992, rg: 103, rmsf: 0.61, dsm: 2.08, af: 7, ram_corr: 0.760, gap: 1.782, composite: 0.58, neg: 75, stability: "STABLE" },
  { step: 1000, rmsd: 1.11, q: 1.000, rg: 100, rmsf: 0.45, dsm: 2.05, af: 4, ram_corr: 0.756, gap: 1.660, composite: 0.62, neg: 75, stability: "STABLE" },
  { step: 1500, rmsd: 0.89, q: 1.000, rg: 100, rmsf: 0.34, dsm: 2.14, af: 3, ram_corr: 0.774, gap: 1.607, composite: 0.67, neg: 75, stability: "STABLE" },
  { step: 2000, rmsd: 1.88, q: 0.987, rg: 106, rmsf: 0.60, dsm: 6.59, af: 3, ram_corr: 0.776, gap: 1.559, composite: 0.62, neg: 88, stability: "STABLE" },
  { step: 2500, rmsd: 1.38, q: 0.995, rg: 105, rmsf: 0.49, dsm: 2.05, af: 2, ram_corr: 0.769, gap: 1.535, composite: 0.82, neg: 75, stability: "STABLE" },
  { step: 3000, rmsd: 0.88, q: 1.000, rg: 104, rmsf: 0.44, dsm: 2.02, af: 2, ram_corr: 0.789, gap: 1.523, composite: 0.65, neg: 75, stability: "STABLE" },
  { step: 3500, rmsd: 1.98, q: 0.979, rg: 106, rmsf: 0.65, dsm: 2.07, af: 4, ram_corr: 0.744, gap: 1.512, composite: 0.74, neg: 62, stability: "STABLE" },
  { step: 4000, rmsd: 1.73, q: 0.991, rg: 102, rmsf: 0.66, dsm: 2.09, af: 6, ram_corr: 0.806, gap: 1.524, composite: 0.82, neg: 75, stability: "STABLE" },
  { step: 4500, rmsd: 1.30, q: 1.000, rg: 100, rmsf: 0.43, dsm: 1.85, af: 3, ram_corr: 0.782, gap: 1.450, composite: 0.82, neg: 88, stability: "STABLE" },
  { step: 5000, rmsd: 0.79, q: 1.000, rg: 99, rmsf: 0.35, dsm: 1.87, af: 6, ram_corr: 0.754, gap: 1.468, composite: 0.77, neg: 75, stability: "STABLE" },
  { step: 5500, rmsd: 1.82, q: 0.983, rg: 104, rmsf: 0.55, dsm: 2.37, af: 4, ram_corr: 0.778, gap: 1.479, composite: 0.57, neg: 62, stability: "STABLE" },
  { step: 6000, rmsd: 1.26, q: 1.000, rg: 100, rmsf: 0.46, dsm: 2.09, af: 9, ram_corr: 0.778, gap: 1.404, composite: 0.97, neg: 62, stability: "STABLE" },
  { step: 6500, rmsd: 1.32, q: 1.000, rg: 99, rmsf: 0.54, dsm: 2.03, af: 4, ram_corr: 0.796, gap: 1.445, composite: 1.02, neg: 62, stability: "STABLE" },
  { step: 7000, rmsd: 1.25, q: 1.000, rg: 100, rmsf: 0.41, dsm: 1.95, af: 3, ram_corr: 0.759, gap: 1.445, composite: 0.97, neg: 88, stability: "STABLE" },
  { step: 7500, rmsd: 1.96, q: 1.000, rg: 96, rmsf: 0.57, dsm: 2.06, af: 4, ram_corr: 0.790, gap: 1.418, composite: 0.90, neg: 75, stability: "STABLE" },
  { step: 8000, rmsd: 2.04, q: 0.977, rg: 103, rmsf: 0.63, dsm: 1.92, af: 5, ram_corr: 0.761, gap: 1.442, composite: 0.84, neg: 50, stability: "STABLE" },
  { step: 8500, rmsd: 1.45, q: 1.000, rg: 101, rmsf: 0.54, dsm: 2.02, af: 4, ram_corr: 0.825, gap: 1.386, composite: 0.81, neg: 62, stability: "STABLE" },
  { step: 9000, rmsd: 0.99, q: 1.000, rg: 97, rmsf: 0.43, dsm: 1.96, af: 9, ram_corr: 0.791, gap: 1.430, composite: 0.84, neg: 88, stability: "STABLE" },
  { step: 9500, rmsd: 1.49, q: 1.000, rg: 95, rmsf: 0.44, dsm: 2.05, af: 5, ram_corr: 0.778, gap: 1.418, composite: 0.69, neg: 88, stability: "STABLE" },
  { step: 10000, rmsd: 1.65, q: 0.963, rg: 106, rmsf: 0.58, dsm: 1.94, af: 3, ram_corr: 0.759, gap: 1.465, composite: 1.15, neg: 75, stability: "STABLE" },
];

const lambdaData = [
  { step: 200, tp: 0.694, ram: 0.818, hba: 4.970, rep: 0.336, geom: 0.255, cont: 1.938 },
  { step: 800, tp: 0.675, ram: 0.859, hba: 4.913, rep: 0.306, geom: 0.255, cont: 1.915 },
  { step: 1600, tp: 0.657, ram: 0.919, hba: 4.863, rep: 0.284, geom: 0.256, cont: 1.879 },
  { step: 2400, tp: 0.650, ram: 0.971, hba: 4.828, rep: 0.265, geom: 0.261, cont: 1.884 },
  { step: 3200, tp: 0.654, ram: 1.055, hba: 4.830, rep: 0.256, geom: 0.268, cont: 1.871 },
  { step: 4000, tp: 0.652, ram: 1.120, hba: 4.808, rep: 0.247, geom: 0.271, cont: 1.871 },
  { step: 5000, tp: 0.633, ram: 1.184, hba: 4.762, rep: 0.241, geom: 0.273, cont: 1.858 },
  { step: 6000, tp: 0.625, ram: 1.279, hba: 4.733, rep: 0.236, geom: 0.279, cont: 1.867 },
  { step: 7000, tp: 0.621, ram: 1.338, hba: 4.713, rep: 0.233, geom: 0.283, cont: 1.855 },
  { step: 8000, tp: 0.620, ram: 1.391, hba: 4.689, rep: 0.231, geom: 0.292, cont: 1.859 },
  { step: 9000, tp: 0.626, ram: 1.450, hba: 4.679, rep: 0.230, geom: 0.296, cont: 1.855 },
  { step: 9600, tp: 0.626, ram: 1.488, hba: 4.679, rep: 0.231, geom: 0.298, cont: 1.853 },
  { step: 10000, tp: 0.623, ram: 1.510, hba: 4.668, rep: 0.230, geom: 0.302, cont: 1.858 },
];

const subgapData = [
  { step: 200, local: 0.094, rep: 0.661, secondary: 0.367, packing: 0.401, rg: 0.263 },
  { step: 800, local: 0.086, rep: 0.490, secondary: 0.535, packing: 0.496, rg: 0.266 },
  { step: 1600, local: 0.108, rep: 0.742, secondary: 0.384, packing: 0.183, rg: 0.177 },
  { step: 2400, local: 0.126, rep: 0.727, secondary: 0.436, packing: 0.263, rg: 0.282 },
  { step: 3200, local: 0.103, rep: 0.461, secondary: 0.398, packing: 0.496, rg: 0.292 },
  { step: 4000, local: 0.065, rep: 0.432, secondary: 0.315, packing: 0.752, rg: 0.461 },
  { step: 5000, local: 0.061, rep: 0.683, secondary: 0.289, packing: 0.336, rg: 0.298 },
  { step: 6000, local: 0.065, rep: 0.437, secondary: 0.315, packing: 0.213, rg: 0.160 },
  { step: 7000, local: 0.096, rep: 0.426, secondary: 0.470, packing: 0.640, rg: 0.378 },
  { step: 8000, local: 0.095, rep: 0.340, secondary: 0.335, packing: 0.455, rg: 0.155 },
  { step: 9000, local: 0.180, rep: 0.328, secondary: 0.281, packing: 0.388, rg: 0.086 },
  { step: 9600, local: 0.080, rep: 0.437, secondary: 0.249, packing: 0.607, rg: 0.457 },
  { step: 10000, local: 0.073, rep: 0.652, secondary: 0.308, packing: 0.253, rg: 0.225 },
];

const COLORS = {
  rmsd: "#ef4444",
  q: "#22c55e",
  rg: "#3b82f6",
  rmsf: "#f59e0b",
  dsm: "#8b5cf6",
  af: "#ec4899",
  ram: "#06b6d4",
  gap: "#f97316",
  composite: "#6366f1",
  local: "#3b82f6",
  rep: "#ef4444",
  secondary: "#22c55e",
  packing: "#f59e0b",
  rgGap: "#8b5cf6",
  tp: "#3b82f6",
  hba: "#22c55e",
  cont: "#f59e0b",
  geom: "#ec4899",
};

const tabs = ["Dynamics", "Training", "Lambdas", "Gaps"];

export default function Run50Dashboard() {
  const [tab, setTab] = useState("Dynamics");

  return (
    <div className="min-h-screen p-4" style={{ background: "var(--bg-primary, #0f172a)", color: "var(--text-primary, #e2e8f0)" }}>
      <div className="max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold mb-1">Run50 Training Dashboard</h1>
        <p className="text-sm mb-4 opacity-70">λ_rg=1.0 · 20K steps (50% complete) · resumed from run49 s5000</p>

        <div className="flex gap-2 mb-6">
          {tabs.map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                tab === t
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700"
              }`}
            >
              {t}
            </button>
          ))}
        </div>

        {tab === "Dynamics" && (
          <div className="space-y-6">
            <div className="bg-gray-800 bg-opacity-50 rounded-xl p-4">
              <h2 className="text-lg font-semibold mb-3">RMSD & RMSF (Å) — β=100 2K-step dynamics</h2>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={data.filter(d => d.rmsd !== null)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="step" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 2.5]} />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #475569", borderRadius: 8 }} />
                  <Legend />
                  <Line type="monotone" dataKey="rmsd" stroke={COLORS.rmsd} strokeWidth={2} name="RMSD" dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="rmsf" stroke={COLORS.rmsf} strokeWidth={2} name="RMSF" dot={{ r: 3 }} />
                  <ReferenceLine y={1.0} stroke="#475569" strokeDasharray="5 5" label={{ value: "1.0 Å", fill: "#64748b", fontSize: 11 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gray-800 bg-opacity-50 rounded-xl p-4">
              <h2 className="text-lg font-semibold mb-3">Q (native contacts) & Rg (%)</h2>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={data.filter(d => d.q !== null)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="step" stroke="#94a3b8" fontSize={12} />
                  <YAxis yAxisId="left" stroke={COLORS.q} fontSize={12} domain={[0.96, 1.005]} tickFormatter={v => v.toFixed(3)} />
                  <YAxis yAxisId="right" orientation="right" stroke={COLORS.rg} fontSize={12} domain={[94, 108]} />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #475569", borderRadius: 8 }} />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="q" stroke={COLORS.q} strokeWidth={2} name="Q" dot={{ r: 3 }} />
                  <Line yAxisId="right" type="monotone" dataKey="rg" stroke={COLORS.rg} strokeWidth={2} name="Rg %" dot={{ r: 3 }} />
                  <ReferenceLine yAxisId="left" y={1.0} stroke="#22c55e" strokeDasharray="5 5" strokeOpacity={0.3} />
                  <ReferenceLine yAxisId="right" y={100} stroke="#3b82f6" strokeDasharray="5 5" strokeOpacity={0.3} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "Best RMSD", value: "0.79 Å", sub: "s5000", color: COLORS.rmsd },
                { label: "Best Q", value: "1.000", sub: "13/20 checkpoints", color: COLORS.q },
                { label: "Best RMSF", value: "0.34 Å", sub: "s1500", color: COLORS.rmsf },
              ].map(({ label, value, sub, color }) => (
                <div key={label} className="bg-gray-800 bg-opacity-50 rounded-xl p-4 text-center">
                  <div className="text-sm opacity-60">{label}</div>
                  <div className="text-2xl font-bold mt-1" style={{ color }}>{value}</div>
                  <div className="text-xs opacity-50 mt-1">{sub}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === "Training" && (
          <div className="space-y-6">
            <div className="bg-gray-800 bg-opacity-50 rounded-xl p-4">
              <h2 className="text-lg font-semibold mb-3">DSM Loss & Anti-funnel %</h2>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="step" stroke="#94a3b8" fontSize={12} />
                  <YAxis yAxisId="left" stroke={COLORS.dsm} fontSize={12} domain={[1.5, 7]} />
                  <YAxis yAxisId="right" orientation="right" stroke={COLORS.af} fontSize={12} domain={[0, 12]} />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #475569", borderRadius: 8 }} />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="dsm" stroke={COLORS.dsm} strokeWidth={2} name="DSM loss" dot={{ r: 3 }} />
                  <Bar yAxisId="right" dataKey="af" fill={COLORS.af} fillOpacity={0.4} name="Anti-funnel %" />
                  <ReferenceLine yAxisId="left" y={2.0} stroke="#8b5cf6" strokeDasharray="5 5" strokeOpacity={0.3} label={{ value: "DSM=2.0", fill: "#64748b", fontSize: 11 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gray-800 bg-opacity-50 rounded-xl p-4">
              <h2 className="text-lg font-semibold mb-3">Native Gap & Ramachandran Correlation</h2>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="step" stroke="#94a3b8" fontSize={12} />
                  <YAxis yAxisId="left" stroke={COLORS.gap} fontSize={12} domain={[1.2, 2.0]} />
                  <YAxis yAxisId="right" orientation="right" stroke={COLORS.ram} fontSize={12} domain={[0.72, 0.84]} />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #475569", borderRadius: 8 }} />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="gap" stroke={COLORS.gap} strokeWidth={2} name="Native gap @σ=0.3" dot={{ r: 3 }} />
                  <Line yAxisId="right" type="monotone" dataKey="ram_corr" stroke={COLORS.ram} strokeWidth={2} name="Ram. corr" dot={{ r: 3 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-4 gap-3">
              {[
                { label: "Stability", value: "20/20", sub: "STABLE", color: "#22c55e" },
                { label: "DSM (EMA)", value: "~2.0", sub: "baseline", color: COLORS.dsm },
                { label: "Best af%", value: "2%", sub: "s2500", color: COLORS.af },
                { label: "Best Ram", value: "0.825", sub: "s8500", color: COLORS.ram },
              ].map(({ label, value, sub, color }) => (
                <div key={label} className="bg-gray-800 bg-opacity-50 rounded-xl p-4 text-center">
                  <div className="text-sm opacity-60">{label}</div>
                  <div className="text-xl font-bold mt-1" style={{ color }}>{value}</div>
                  <div className="text-xs opacity-50 mt-1">{sub}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === "Lambdas" && (
          <div className="space-y-6">
            <div className="bg-gray-800 bg-opacity-50 rounded-xl p-4">
              <h2 className="text-lg font-semibold mb-3">Ramachandran λ (climbing — investing in backbone)</h2>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={lambdaData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="step" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} domain={[0.6, 1.6]} />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #475569", borderRadius: 8 }} />
                  <Line type="monotone" dataKey="ram" stroke={COLORS.ram} strokeWidth={3} name="λ_ram" dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gray-800 bg-opacity-50 rounded-xl p-4">
              <h2 className="text-lg font-semibold mb-3">Other Learned Lambdas</h2>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={lambdaData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="step" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 2.2]} />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #475569", borderRadius: 8 }} />
                  <Legend />
                  <Line type="monotone" dataKey="tp" stroke={COLORS.tp} strokeWidth={2} name="λ_θφ" dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="cont" stroke={COLORS.cont} strokeWidth={2} name="λ_contact" dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="geom" stroke={COLORS.geom} strokeWidth={2} name="λ_geom" dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="rep" stroke={COLORS.rep} strokeWidth={2} name="λ_rep" dot={{ r: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gray-800 bg-opacity-50 rounded-xl p-3 text-sm opacity-80">
              <p><strong>Key trend:</strong> λ_ram 0.82 → 1.51 (+84%) — the model is investing heavily in Ramachandran backbone preferences.</p>
              <p className="mt-1">λ_θφ drifting down (0.69 → 0.62): local MLP relaxing as ram takes over backbone scoring.</p>
              <p className="mt-1">λ_contact stable (~1.86): pair specificity not inflating (unlike run49 where it compensated for E_Rg).</p>
            </div>
          </div>
        )}

        {tab === "Gaps" && (
          <div className="space-y-6">
            <div className="bg-gray-800 bg-opacity-50 rounded-xl p-4">
              <h2 className="text-lg font-semibold mb-3">Per-subterm Gaps @σ=0.3 (higher = better discrimination)</h2>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={subgapData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="step" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 0.9]} />
                  <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #475569", borderRadius: 8 }} />
                  <Legend />
                  <Line type="monotone" dataKey="local" stroke={COLORS.local} strokeWidth={2} name="Local" dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="rep" stroke={COLORS.rep} strokeWidth={2} name="Repulsion" dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="secondary" stroke={COLORS.secondary} strokeWidth={2} name="Secondary" dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="packing" stroke={COLORS.packing} strokeWidth={2} name="Packing (geom+hp)" dot={{ r: 2 }} />
                  <Line type="monotone" dataKey="rg" stroke={COLORS.rgGap} strokeWidth={2} name="Rg" dot={{ r: 2 }} strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-gray-800 bg-opacity-50 rounded-xl p-3 text-sm opacity-80">
              <p><strong>Balanced discrimination:</strong> All subterms contribute positively. No single term dominates (unlike run49 where Rg gap was 3-4× everything else).</p>
              <p className="mt-1">Rg gap at λ=1.0 is ~0.1-0.4 — comparable to learned terms, not overwhelming them.</p>
              <p className="mt-1">Secondary gap climbing (0.28 → 0.47 at s7000) as ram lambda invests more in backbone.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
