import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import axios from "axios";
import { toast } from "sonner";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, AreaChart, Area, Legend, BarChart, Bar, PieChart, Pie, Cell as RCell,
} from "recharts";
import {
  CircleNotch, Cpu, HardDrives, Broadcast, Lightning, CloudArrowUp,
  WifiHigh, WifiSlash, Package, Rocket, Warning, Timer,
  ChartLineUp, ArrowsClockwise, Terminal, Database, TreeStructure,
  Cube, MapPin, Brain, Path,
} from "@phosphor-icons/react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8001";
// Ensure no trailing slash so `${API}/...` builds correctly
const API = `${BACKEND_URL.replace(/\/+$/, "")}/api`;

const LABELS = ["person", "car", "truck", "bicycle", "forklift", "helmet", "pallet", "box", "dog"];

const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
const rand = (min, max) => Math.random() * (max - min) + min;
const nowMs = () => new Date().toISOString().slice(11, 23);

const ROUTE_COLORS = { edge: "#00E676", cloud: "#0055FF", hybrid: "#FFCC00" };

const Cell = ({ children, className = "", testId }) => (
  <div data-testid={testId} className={`bg-[#0A0A0A] border border-[#27272A] ${className}`}>
    {children}
  </div>
);

const Overline = ({ children, className = "" }) => (
  <span className={`overline ${className}`}>{children}</span>
);

const KPI = ({ label, value, unit, tone = "default", testId }) => {
  const toneMap = {
    default: "text-white", accent: "text-[#0055FF]",
    good: "text-[#00E676]", warn: "text-[#FFCC00]", bad: "text-[#FF3333]",
  };
  return (
    <Cell className="p-4 flex flex-col gap-2 min-h-[104px]" testId={testId}>
      <Overline>{label}</Overline>
      <div className="flex items-baseline gap-1">
        <span className={`font-mono text-3xl font-semibold ${toneMap[tone]}`}>{value}</span>
        {unit && <span className="font-mono text-xs text-[#71717a]">{unit}</span>}
      </div>
    </Cell>
  );
};

export default function Dashboard() {
  const [devices, setDevices] = useState([]);
  const [models, setModels] = useState([]);
  const [deployments, setDeployments] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState(null);
  const [mode, setMode] = useState("edge");
  const [running, setRunning] = useState(true);
  const [series, setSeries] = useState([]);
  const [logs, setLogs] = useState([]);
  const [summary, setSummary] = useState({ edge: null, cloud: null });
  const [selectedModelId, setSelectedModelId] = useState(null);
  const [deploying, setDeploying] = useState(false);
  const [showArch, setShowArch] = useState(false);

  // Decision engine state
  const [engineName, setEngineName] = useState("rf"); // rule | dt | rf | ql
  const [engineStatus, setEngineStatus] = useState(null);
  const [lastDecision, setLastDecision] = useState(null);
  const [decisionHistory, setDecisionHistory] = useState([]);
  const [distribution, setDistribution] = useState({ edge: 0, cloud: 0, hybrid: 0 });
  const [training, setTraining] = useState(false);
  const [autoRoute, setAutoRoute] = useState(true);
  const [realInferenceLoading, setRealInferenceLoading] = useState(false);
  const [lastRealInference, setLastRealInference] = useState(null);
  const [qtable, setQtable] = useState([]);
  const [benchmark, setBenchmark] = useState(null);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenario, setSelectedScenario] = useState("mixed");
  const [ciResult, setCiResult] = useState(null);
  const [ciRunning, setCiRunning] = useState(false);
  const [reportGenerating, setReportGenerating] = useState(false);
  const [assistantMessages, setAssistantMessages] = useState([]); // [{q, a, ts}]
  const [assistantInput, setAssistantInput] = useState("");
  const [assistantLoading, setAssistantLoading] = useState(false);
  const [assistantSessionId, setAssistantSessionId] = useState(null);
  const [completeRunning, setCompleteRunning] = useState(false);
  const [completeStatus, setCompleteStatus] = useState(null);
  const [masterSummary, setMasterSummary] = useState(null);

  const bufferRef = useRef([]);
  const logsRef = useRef(null);

  const selectedDevice = devices.find((d) => d.id === selectedDeviceId) || devices[0];
  const selectedModel = models.find((m) => m.id === (selectedDevice?.deployed_model_id));

  const pushLog = useCallback((level, msg) => {
    setLogs((prev) => [...prev, { t: nowMs(), level, msg }].slice(-200));
  }, []);

  /* ---- Bootstrap ---- */
  useEffect(() => {
    (async () => {
      try {
        const [d, m, dep, st, sc] = await Promise.all([
          axios.get(`${API}/devices`),
          axios.get(`${API}/models`),
          axios.get(`${API}/deployments?limit=10`),
          axios.get(`${API}/decisions/status`),
          axios.get(`${API}/scenarios`),
        ]);
        setDevices(d.data);
        setModels(m.data);
        setDeployments(dep.data);
        setEngineStatus(st.data);
        setScenarios(sc.data.scenarios || []);
        if (d.data.length) setSelectedDeviceId(d.data[0].id);
        pushLog("info", `Loaded ${d.data.length} devices, ${m.data.length} models, ${sc.data.scenarios?.length || 0} scenarios`);
        pushLog("ok", `Decision engines ready — DT acc=${(st.data.DecisionTree?.metrics?.accuracy ?? 0).toFixed(3)}, RF acc=${(st.data.RandomForest?.metrics?.accuracy ?? 0).toFixed(3)}`);
      } catch (e) {
        pushLog("err", `Bootstrap failed: ${e.message}`);
      }
    })();
  }, [pushLog]);

  useEffect(() => {
    if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight;
  }, [logs]);

  /* ---- Fetch decision history + distribution periodically ---- */
  useEffect(() => {
    const iv = setInterval(async () => {
      try {
        const [h, d] = await Promise.all([
          axios.get(`${API}/decisions/history?limit=15&engine=${engineFullName(engineName)}`),
          axios.get(`${API}/decisions/distribution?minutes=30&engine=${engineFullName(engineName)}`),
        ]);
        setDecisionHistory(h.data);
        setDistribution(d.data);
      } catch (e) { /* silent */ }
    }, 3000);
    return () => clearInterval(iv);
  }, [engineName]);

  /* ---- Simulation tick with real decision engine call ---- */
  useEffect(() => {
    if (!running || !selectedDevice) return;
    const iv = setInterval(async () => {
      const connected = selectedDevice.connected;

      // Build realistic context
      const cpuAvail = rand(15, 90);
      const memAvail = rand(20, 85);
      const netLatency = connected ? rand(30, 400) : 9999;
      const batchSize = pick([1, 2, 4, 8, 16]);
      const priority = pick([1, 2, 2, 3, 3, 4, 5]);
      const costBudget = rand(0.01, 5.0);
      const modelSizeMb = selectedModel?.size_mb || 6.2;

      const ctx = {
        network_latency_ms: Number(netLatency.toFixed(1)),
        connectivity: connected ? 1 : 0,
        cpu_available: Number(cpuAvail.toFixed(1)),
        memory_available: Number(memAvail.toFixed(1)),
        batch_size: batchSize,
        priority,
        cost_budget_usd: Number(costBudget.toFixed(3)),
        model_size_mb: Number(modelSizeMb.toFixed(1)),
      };

      let effectiveMode = mode;
      let decision = null;
      try {
        const res = await axios.post(`${API}/decisions/predict`, {
          engine: engineName, context: ctx, persist: true,
        });
        decision = res.data;
        setLastDecision({ ...decision, context: ctx });
        if (autoRoute) {
          effectiveMode = decision.route === "hybrid" ? "edge" : decision.route;
          if (effectiveMode !== mode) setMode(effectiveMode);
        }
        pushLog(
          decision.route === "cloud" ? "info" : decision.route === "hybrid" ? "warn" : "ok",
          `[${decision.engine}] → ${decision.route.toUpperCase()} (p=${decision.confidence}) · ${decision.reason}`
        );
      } catch (e) {
        pushLog("err", `decision-engine error: ${e.message}`);
      }

      const isEdge = effectiveMode === "edge";
      const cloudDown = !connected;
      const edgeLatency = rand(18, 42);
      const cloudLatency = cloudDown ? NaN : rand(120, 280);
      const cpu = isEdge ? rand(45, 82) : rand(8, 22);
      const memory = isEdge ? rand(38, 68) : rand(12, 25);
      const fps = isEdge ? rand(22, 30) : (cloudDown ? 0 : rand(6, 14));
      const edgeCost = 0.00002;
      const cloudCost = cloudDown ? 0 : 0.00085;
      const objects = Math.floor(rand(0, 6));
      const topLabel = pick(LABELS);
      const topConf = rand(0.62, 0.98);
      const activeLatency = isEdge ? edgeLatency : (cloudDown ? NaN : cloudLatency);

      bufferRef.current.push({
        device_id: selectedDevice.id, device_name: selectedDevice.name, mode: effectiveMode,
        latency_ms: isNaN(activeLatency) ? 0 : Math.round(activeLatency * 100) / 100,
        cpu: Math.round(cpu * 10) / 10, memory: Math.round(memory * 10) / 10,
        fps: Math.round(fps * 10) / 10, cost_usd: isEdge ? edgeCost : (cloudDown ? 0 : cloudCost),
        objects_detected: objects, top_label: topLabel, top_confidence: Math.round(topConf * 100) / 100,
        connected, timestamp: new Date().toISOString(),
      });

      setSeries((prev) => {
        const t = nowMs().slice(0, 8);
        return [...prev, {
          t, edge_latency: Math.round(edgeLatency),
          cloud_latency: cloudDown ? null : Math.round(cloudLatency),
          edge_cost: Number((edgeCost * 1000).toFixed(3)),
          cloud_cost: cloudDown ? 0 : Number((cloudCost * 1000).toFixed(3)),
        }].slice(-40);
      });
    }, 1500);
    return () => clearInterval(iv);
  }, [running, mode, selectedDevice, selectedModel, engineName, autoRoute, pushLog]);

  useEffect(() => {
    const flush = setInterval(async () => {
      if (!bufferRef.current.length) return;
      const batch = bufferRef.current.splice(0, bufferRef.current.length);
      try { await axios.post(`${API}/inferences`, { records: batch }); } catch (e) { /* silent */ }
    }, 5000);
    return () => clearInterval(flush);
  }, []);

  useEffect(() => {
    const iv = setInterval(async () => {
      try {
        const res = await axios.get(`${API}/metrics/summary`);
        setSummary(res.data);
      } catch (e) { /* silent */ }
    }, 4000);
    return () => clearInterval(iv);
  }, []);

  const toggleNetwork = async () => {
    if (!selectedDevice) return;
    const next = !selectedDevice.connected;
    try {
      const res = await axios.post(`${API}/devices/${selectedDevice.id}/network`, { connected: next });
      setDevices((prev) => prev.map((d) => (d.id === res.data.id ? res.data : d)));
      pushLog(next ? "ok" : "err",
        next ? `[${selectedDevice.name}] network RESTORED`
             : `[${selectedDevice.name}] SIMULATE DISCONNECT — MQTT severed`);
      toast(next ? "Network restored" : "Disconnected");
    } catch (e) {
      pushLog("err", `Network toggle failed: ${e.message}`);
    }
  };

  const deployModel = async () => {
    if (!selectedDevice || !selectedModelId) { toast("Select a model first"); return; }
    setDeploying(true);
    const model = models.find((m) => m.id === selectedModelId);
    pushLog("info", `Deploying ${model.name}@${model.version} → ${selectedDevice.name}`);
    try {
      await new Promise((r) => setTimeout(r, 900));
      pushLog("info", `[greengrass] downloading ${model.size_mb}MB...`);
      await new Promise((r) => setTimeout(r, 700));
      const res = await axios.post(`${API}/deployments`, {
        device_id: selectedDevice.id, model_id: selectedModelId,
      });
      setDeployments((prev) => [res.data, ...prev].slice(0, 10));
      setDevices((prev) => prev.map((d) =>
        d.id === selectedDevice.id
          ? { ...d, deployed_model_id: model.id, deployed_model_name: `${model.name}@${model.version}` }
          : d));
      pushLog("ok", `[greengrass] deployment ACTIVE`);
      toast.success("Deployment active");
    } catch (e) {
      pushLog("err", `Deployment failed: ${e.message}`);
    } finally { setDeploying(false); }
  };

  const retrainEngines = async () => {
    setTraining(true);
    pushLog("info", "Retraining Decision Tree + Random Forest + Q-Learning agent...");
    try {
      const [dtRes, qlRes] = await Promise.all([
        axios.post(`${API}/decisions/train`, { n_samples: 4000 }),
        axios.post(`${API}/decisions/qlearning/train`, { episodes: 5000 }),
      ]);
      const st = await axios.get(`${API}/decisions/status`);
      setEngineStatus(st.data);
      pushLog("ok", `DT acc=${dtRes.data.trained.DecisionTree.accuracy} · RF acc=${dtRes.data.trained.RandomForest.accuracy} · Q-reward=${qlRes.data.trained.mean_reward_last_500}`);
      toast.success("All engines retrained");
    } catch (e) {
      pushLog("err", `Training failed: ${e.message}`);
    } finally { setTraining(false); }
  };

  const runRealInference = async () => {    if (!selectedDevice) return;
    setRealInferenceLoading(true);
    const modelSize = selectedModel?.size_mb || 6.2;
    const effectiveMode = mode;
    const netLatency = selectedDevice.connected ? 100 : 9999;
    pushLog("info", `[real-inference] running REAL workload on ${effectiveMode.toUpperCase()} · model=${modelSize}MB`);
    try {
      const res = await axios.post(`${API}/inference/run`, {
        mode: effectiveMode,
        model_size_mb: modelSize,
        network_latency_ms: netLatency,
        connectivity: selectedDevice.connected ? 1 : 0,
        batch_size: 1,
      });
      setLastRealInference(res.data);
      pushLog(res.data.success ? "ok" : "err",
        `[real-inference] latency=${res.data.latency_ms}ms · cpu=${res.data.cpu_percent}% · mem=${res.data.memory_mb}MB · flops=${res.data.workload_flops.toLocaleString()}`
      );
      toast.success(`Real ${effectiveMode} inference: ${res.data.latency_ms}ms`);
    } catch (e) {
      pushLog("err", `real-inference failed: ${e.message}`);
    } finally { setRealInferenceLoading(false); }
  };

  const runBenchmark = async (nScenarios = 500) => {
    setBenchmarkRunning(true);
    pushLog("info", `[benchmark] starting · scenario=${selectedScenario} · ${nScenarios} samples · 4 policies`);
    try {
      const res = await axios.post(`${API}/benchmark/run`, {
        n_scenarios: nScenarios, seed: 2026, save_artefacts: true,
        scenario_id: selectedScenario,
      }, { timeout: 60000 });
      setBenchmark(res.data);
      const eng = Object.keys(res.data.summary);
      pushLog("ok", `[benchmark] done · ${res.data.n_records} records · ${eng.join(", ")}`);
      toast.success("Experiment complete", { description: `scenario: ${selectedScenario}` });
    } catch (e) {
      pushLog("err", `benchmark failed: ${e.message}`);
    } finally { setBenchmarkRunning(false); }
  };

  const runCI = async () => {
    setCiRunning(true);
    const seeds = [2026, 42, 7, 1234, 9999];
    pushLog("info", `[benchmark-ci] running ${seeds.length} seeds × 200 samples × 4 policies · scenario=${selectedScenario}`);
    try {
      const res = await axios.post(`${API}/benchmark/ci`, {
        scenario_id: selectedScenario, seeds, n_scenarios: 200,
      }, { timeout: 120000 });
      setCiResult(res.data);
      pushLog("ok", `[benchmark-ci] complete · aggregated across ${seeds.length} seeds · CI 95%`);
      toast.success("Statistical validation complete", { description: `${seeds.length} seeds · CI 95%` });
    } catch (e) {
      pushLog("err", `CI aggregation failed: ${e.message}`);
    } finally { setCiRunning(false); }
  };

  const generateReport = async () => {
    setReportGenerating(true);
    pushLog("info", `[report] generating PDF thesis report · scenario=${selectedScenario}`);
    try {
      const res = await axios.post(`${API}/report/generate`, {
        scenario_id: selectedScenario, seeds: [2026, 42, 7, 1234, 9999], n_scenarios: 200,
      }, { timeout: 180000 });
      pushLog("ok", `[report] PDF ready · ${(res.data.size_bytes / 1024).toFixed(1)}KB`);
      toast.success("Thesis Report ready", { description: "Downloading PDF..." });
      const link = document.createElement("a");
      link.href = `${API}/benchmark/artefact/thesis_report.pdf`;
      link.download = `thesis_report_${selectedScenario}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (e) {
      pushLog("err", `report generation failed: ${e.message}`);
    } finally { setReportGenerating(false); }
  };

  const runCompleteBenchmark = async () => {
    if (completeRunning) return;
    setCompleteRunning(true);
    setMasterSummary(null);
    pushLog("info", "[complete] starting run — 8 scenarios × 5 seeds × 4 policies");
    try {
      await axios.post(`${API}/experiments/run_complete`, {
        seeds: [2026, 42, 7, 1234, 9999], n_scenarios: 100,
      }, { timeout: 5000 });
      toast.info("Complete Benchmark started", { description: "polling status..." });
    } catch (e) {
      pushLog("err", `complete benchmark failed to start: ${e.message}`);
      setCompleteRunning(false);
      return;
    }
    const iv = setInterval(async () => {
      try {
        const st = await axios.get(`${API}/experiments/status`);
        setCompleteStatus(st.data);
        if (st.data.status === "done") {
          clearInterval(iv);
          setCompleteRunning(false);
          setMasterSummary(st.data.master_summary);
          pushLog("ok", `[complete] all ${st.data.total} scenarios done · ${st.data.elapsed_s}s`);
          toast.success("Complete Benchmark done", {
            description: `${st.data.total} scenarios · master summary ready`,
          });
        }
      } catch (e) { /* silent */ }
    }, 2000);
  };

  const askAssistant = async (question) => {
    const q = (question ?? assistantInput).trim();
    if (!q || assistantLoading) return;
    setAssistantLoading(true);
    setAssistantInput("");
    const ts = new Date().toLocaleTimeString();
    setAssistantMessages((prev) => [...prev, { q, a: null, ts }]);
    try {
      const res = await axios.post(`${API}/assistant/ask`, {
        question: q, session_id: assistantSessionId,
      }, { timeout: 60000 });
      setAssistantSessionId(res.data.session_id);
      setAssistantMessages((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = { ...copy[copy.length - 1], a: res.data.answer };
        return copy;
      });
    } catch (e) {
      setAssistantMessages((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = { ...copy[copy.length - 1], a: `Error: ${e.message}` };
        return copy;
      });
    } finally { setAssistantLoading(false); }
  };

  // Load previous benchmark on mount (if any)
  useEffect(() => {
    (async () => {
      try {
        const res = await axios.get(`${API}/benchmark/last`);
        if (res.data.summary) setBenchmark(res.data);
      } catch (e) { /* silent */ }
      // Hydrate complete-benchmark state if a master summary already exists
      try {
        const st = await axios.get(`${API}/experiments/status`);
        setCompleteStatus(st.data);
        if (st.data.status === "done" && st.data.master_summary) {
          setMasterSummary(st.data.master_summary);
        }
      } catch (e) { /* silent */ }
    })();
  }, []);

  const globalAvgLatency = useMemo(() => {
    if (!series.length) return "--";
    const latencies = series.map((s) => (mode === "edge" ? s.edge_latency : s.cloud_latency)).filter(Boolean);
    if (!latencies.length) return "ERR";
    return Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length);
  }, [series, mode]);

  const onlineCount = devices.filter((d) => d.connected).length;

  const currentEngineMeta = engineStatus?.[engineFullName(engineName)];
  const featureImportanceData = currentEngineMeta?.metrics?.feature_importance
    ? Object.entries(currentEngineMeta.metrics.feature_importance)
        .map(([f, v]) => ({ feature: f.replace("_", "\n"), value: Number((v * 100).toFixed(1)) }))
        .sort((a, b) => b.value - a.value)
    : [];

  const distributionData = ["edge", "cloud", "hybrid"].map((r) => ({
    name: r, value: distribution[r] || 0,
  }));

  return (
    <div className="min-h-screen bg-[#050505] text-white">
      {/* HEADER */}
      <header data-testid="app-header" className="border-b border-[#27272A] bg-[#050505] sticky top-0 z-40 backdrop-blur">
        <div className="max-w-[1600px] mx-auto flex items-center justify-between px-4 md:px-6 py-3">
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 border border-[#0055FF] flex items-center justify-center">
              <Broadcast size={16} className="text-[#0055FF]" weight="duotone" />
            </div>
            <div>
              <div className="font-mono text-sm tracking-tighter font-semibold">EDGE-CLOUD // ORCHESTRATOR</div>
              <div className="overline mt-0.5">AWS IoT Greengrass · Decision Engine · Rule-Based · Decision Tree · Random Forest · Q-Learning · Research Prototype</div>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-6">
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full pulse-dot ${onlineCount > 0 ? "bg-[#00E676]" : "bg-[#FF3333]"}`} />
              <span className="font-mono text-xs text-[#a1a1aa]" data-testid="header-devices-online">
                {onlineCount}/{devices.length} DEVICES ONLINE
              </span>
            </div>
            <div className="flex items-center gap-2 font-mono text-xs">
              <Timer size={14} className="text-[#a1a1aa]" />
              <span className="text-[#a1a1aa]">AVG LATENCY</span>
              <span className="text-white" data-testid="header-avg-latency">{globalAvgLatency}</span>
              <span className="text-[#71717a]">ms</span>
            </div>
            <button data-testid="show-architecture-btn" onClick={() => setShowArch(true)}
              className="font-mono uppercase text-xs tracking-wider px-3 py-1.5 border border-[#27272A] hover:border-white text-[#a1a1aa] hover:text-white transition-colors">
              Architecture
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto p-4 md:p-6">
        <div className="grid grid-cols-12 gap-4">
          {/* LEFT: FLEET + DECISION ENGINE */}
          <section className="col-span-12 lg:col-span-3 space-y-4">
            <Cell testId="fleet-panel">
              <div className="px-4 pt-4 pb-3 border-b border-[#27272A] flex items-center justify-between">
                <Overline>Edge Fleet</Overline>
                <span className="font-mono text-xs text-[#71717a]">{devices.length}</span>
              </div>
              <ul className="divide-y divide-[#27272A]">
                {devices.map((d) => {
                  const active = d.id === selectedDeviceId;
                  return (
                    <li key={d.id} data-testid={`device-${d.name}`} onClick={() => setSelectedDeviceId(d.id)}
                      className={`px-4 py-3 cursor-pointer hover:bg-[#141414] transition-colors ${
                        active ? "bg-[#0055FF]/10 border-l-2 border-[#0055FF]" : "border-l-2 border-transparent"
                      }`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Cube size={14} className={d.connected ? "text-[#00E676]" : "text-[#FF3333]"} weight="duotone" />
                          <span className="font-mono text-sm">{d.name}</span>
                        </div>
                        {d.connected ? <WifiHigh size={14} className="text-[#00E676]" /> : <WifiSlash size={14} className="text-[#FF3333]" />}
                      </div>
                      <div className="mt-1 flex items-center gap-1 text-[10px] text-[#71717a] font-mono">
                        <MapPin size={10} /> {d.location}
                      </div>
                      {d.deployed_model_name && (
                        <div className="mt-1 text-[10px] font-mono text-[#a1a1aa]">
                          <span className="text-[#71717a]">model:</span> {d.deployed_model_name}
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            </Cell>

            {/* DECISION ENGINE CONFIG */}
            <Cell testId="decision-engine-panel">
              <div className="px-4 pt-4 pb-3 border-b border-[#27272A] flex items-center justify-between">
                <Overline>Research Policy</Overline>
                <Brain size={14} className="text-[#0055FF]" weight="duotone" />
              </div>
              <div className="p-4 space-y-3">
                <div className="flex flex-col divide-y divide-[#27272A] border border-[#27272A]" data-testid="policy-radio-group">
                  {[
                    { k: "rule", label: "Rule-Based", icon: <Path size={14} className="text-[#FFCC00]" /> },
                    { k: "dt", label: "Decision Tree", icon: <TreeStructure size={14} className="text-[#00E676]" /> },
                    { k: "rf", label: "Random Forest", icon: <Brain size={14} className="text-[#0055FF]" /> },
                    { k: "ql", label: "Q-Learning", icon: <Lightning size={14} className="text-[#FF3333]" /> },
                  ].map((e) => {
                    const active = engineName === e.k;
                    return (
                      <button
                        key={e.k}
                        data-testid={`engine-${e.k}-btn`}
                        onClick={() => setEngineName(e.k)}
                        className={`flex items-center gap-3 px-3 py-2.5 text-left transition-colors ${
                          active ? "bg-[#0055FF]/10 text-white" : "text-[#a1a1aa] hover:bg-[#141414] hover:text-white"
                        }`}
                      >
                        <span
                          className={`inline-flex w-3 h-3 rounded-full border transition-colors ${
                            active ? "border-[#0055FF] bg-[#0055FF]" : "border-[#52525b] bg-transparent"
                          }`}
                          aria-hidden
                        />
                        {e.icon}
                        <span className="font-mono text-xs flex-1">{e.label}</span>
                      </button>
                    );
                  })}
                </div>

                <label className="flex items-center justify-between text-xs font-mono text-[#a1a1aa]">
                  <span className="uppercase tracking-wider text-[10px]">Auto-Route</span>
                  <input
                    data-testid="auto-route-toggle"
                    type="checkbox"
                    checked={autoRoute}
                    onChange={(e) => setAutoRoute(e.target.checked)}
                    className="accent-[#0055FF] w-4 h-4"
                  />
                </label>

                {currentEngineMeta && (
                  <div className="text-xs font-mono space-y-1 pt-2 border-t border-[#27272A]">
                    {engineName === "ql" ? (
                      <>
                        <div className="flex justify-between">
                          <span className="text-[#71717a]">episodes</span>
                          <span className="text-white">{currentEngineMeta.metrics?.episodes ?? "—"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#71717a]">mean_reward</span>
                          <span className="text-white">{currentEngineMeta.metrics?.mean_reward_last_500 ?? "—"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#71717a]">states</span>
                          <span className="text-white">{currentEngineMeta.metrics?.n_states ?? "—"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#71717a]">α · γ</span>
                          <span className="text-white">{currentEngineMeta.metrics?.alpha} · {currentEngineMeta.metrics?.gamma}</span>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="flex justify-between">
                          <span className="text-[#71717a]">accuracy</span>
                          <span className="text-white">{((currentEngineMeta.metrics?.accuracy ?? 0) * 100).toFixed(2)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#71717a]">n_train</span>
                          <span className="text-white">{currentEngineMeta.metrics?.n_train ?? "—"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#71717a]">n_test</span>
                          <span className="text-white">{currentEngineMeta.metrics?.n_test ?? "—"}</span>
                        </div>
                      </>
                    )}
                  </div>
                )}

                <button data-testid="retrain-btn" onClick={retrainEngines} disabled={training}
                  className="w-full font-mono uppercase text-xs tracking-wider px-3 py-2 border border-[#27272A] hover:border-white text-[#a1a1aa] hover:text-white disabled:opacity-40 flex items-center justify-center gap-2">
                  {training ? <><CircleNotch size={12} className="animate-spin" /> Training...</> : <><ArrowsClockwise size={12} /> Retrain Models</>}
                </button>
              </div>
            </Cell>
          </section>

          {/* CENTER: SIMULATOR + CHARTS */}
          <section className="col-span-12 lg:col-span-6 space-y-4">
            <Cell className="p-4 md:p-6 relative overflow-hidden" testId="simulator-panel">
              <div className="flex items-start justify-between flex-wrap gap-4">
                <div>
                  <Overline>Inference Execution</Overline>
                  <h2 className="mt-1 font-mono text-2xl tracking-tighter">{selectedDevice?.name || "—"}</h2>
                  <div className="mt-1 text-xs text-[#a1a1aa]">
                    {selectedDevice?.location}
                    {selectedDevice?.deployed_model_name && (
                      <> · <span className="text-[#0055FF] font-mono">{selectedDevice.deployed_model_name}</span></>
                    )}
                  </div>
                </div>
                <div className="flex flex-col items-end gap-2">
                  <div className="inline-flex border border-[#27272A]" data-testid="mode-toggle">
                    <button data-testid="mode-edge-btn" onClick={() => { setMode("edge"); setAutoRoute(false); }}
                      className={`font-mono uppercase text-xs tracking-wider px-4 py-2 flex items-center gap-2 ${
                        mode === "edge" ? "bg-[#0055FF] text-white" : "text-[#a1a1aa] hover:text-white"
                      }`}>
                      <Lightning size={12} weight="fill" /> Edge
                    </button>
                    <button data-testid="mode-cloud-btn" onClick={() => { setMode("cloud"); setAutoRoute(false); }}
                      className={`font-mono uppercase text-xs tracking-wider px-4 py-2 flex items-center gap-2 ${
                        mode === "cloud" ? "bg-[#0055FF] text-white" : "text-[#a1a1aa] hover:text-white"
                      }`}>
                      <CloudArrowUp size={12} weight="fill" /> Cloud
                    </button>
                  </div>
                  <button data-testid="toggle-network-btn" onClick={toggleNetwork} disabled={!selectedDevice}
                    className={`font-mono uppercase text-xs tracking-wider px-4 py-2 border transition-colors flex items-center gap-2 ${
                      selectedDevice?.connected
                        ? "border-[#FF3333] text-[#FF3333] hover:bg-[#FF3333] hover:text-white"
                        : "border-[#00E676] text-[#00E676] hover:bg-[#00E676] hover:text-black"
                    }`}>
                    {selectedDevice?.connected ? <WifiSlash size={12} /> : <WifiHigh size={12} />}
                    {selectedDevice?.connected ? "Simulate Disconnect" : "Restore Network"}
                  </button>
                </div>
              </div>

              <div className="mt-6 border border-[#27272A] bg-black p-4 relative min-h-[180px] scan-bg">
                <div className="absolute top-3 left-3 flex items-center gap-2">
                  <span className="w-2 h-2 bg-[#FF3333] pulse-dot rounded-full" />
                  <span className="font-mono text-[10px] tracking-widest text-[#a1a1aa]">LIVE · CAM-01</span>
                </div>
                <div className="absolute top-3 right-3 font-mono text-[10px] text-[#71717a]">
                  {new Date().toISOString().slice(0, 19).replace("T", " ")}
                </div>
                <div className="grid grid-cols-3 gap-6 mt-8">
                  <div>
                    <div className="overline">Active Route</div>
                    <div className="font-mono text-3xl mt-1" style={{ color: ROUTE_COLORS[mode] || "#fff" }} data-testid="active-route">
                      {mode.toUpperCase()}
                    </div>
                    <div className="font-mono text-xs text-[#71717a] mt-1">
                      {autoRoute ? "auto (engine)" : "manual"}
                    </div>
                  </div>
                  <div>
                    <div className="overline">FPS</div>
                    <div className="font-mono text-3xl text-white mt-1" data-testid="live-fps">
                      {bufferRef.current.length
                        ? (bufferRef.current[bufferRef.current.length - 1].fps || 0).toFixed(1)
                        : (mode === "edge" ? "25.0" : "10.0")}
                    </div>
                    <div className="font-mono text-xs text-[#71717a] mt-1">{mode.toUpperCase()} MODE</div>
                  </div>
                  <div>
                    <div className="overline">Latency</div>
                    <div className={`font-mono text-3xl mt-1 ${globalAvgLatency === "ERR" ? "text-[#FF3333]" : "text-white"}`} data-testid="live-latency">
                      {globalAvgLatency}<span className="text-xs text-[#71717a] ml-1">ms</span>
                    </div>
                    <div className="font-mono text-xs text-[#71717a] mt-1">p50 window</div>
                  </div>
                </div>
              </div>

              {/* Last decision explainer */}
              {lastDecision && (
                <div className="mt-4 border border-[#27272A] p-3" data-testid="last-decision-card">
                  <div className="flex items-center justify-between mb-2">
                    <Overline>Explain Decision · {lastDecision.engine}</Overline>
                    <span className="font-mono text-[10px] text-[#71717a]">{lastDecision.latency_us}µs</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <div>
                      <div className="text-[10px] font-mono text-[#71717a] uppercase tracking-wider">Route</div>
                      <div className="font-mono text-xl" style={{ color: ROUTE_COLORS[lastDecision.route] }}>
                        {lastDecision.route.toUpperCase()}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] font-mono text-[#71717a] uppercase tracking-wider">Confidence</div>
                      <div className="font-mono text-xl text-white">{(lastDecision.confidence * 100).toFixed(1)}%</div>
                    </div>
                    <div className="flex-1">
                      <div className="text-[10px] font-mono text-[#71717a] uppercase tracking-wider">Summary</div>
                      <div className="font-mono text-[11px] text-[#a1a1aa] truncate">{lastDecision.explanation?.summary || lastDecision.reason}</div>
                    </div>
                  </div>
                  <div className="mt-2 flex gap-2">
                    {Object.entries(lastDecision.probabilities).map(([r, p]) => (
                      <div key={r} className="flex-1">
                        <div className="text-[10px] font-mono text-[#71717a] uppercase">{r}</div>
                        <div className="h-1.5 bg-[#141414] mt-1">
                          <div className="h-full" style={{ width: `${p * 100}%`, background: ROUTE_COLORS[r] }} />
                        </div>
                      </div>
                    ))}
                  </div>
                  {/* Structured factors */}
                  {lastDecision.explanation?.factors && lastDecision.explanation.factors.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-[#27272A]" data-testid="explain-factors">
                      <div className="overline mb-2">Contributing Factors</div>
                      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                        {lastDecision.explanation.factors.slice(0, 8).map((f, i) => (
                          <div key={i} className="flex items-center gap-2 text-[10px] font-mono">
                            <span className={f.met ? "text-[#00E676]" : "text-[#52525b]"}>{f.met ? "✓" : "○"}</span>
                            <span className="text-[#a1a1aa]">{f.name}</span>
                            <span className="text-[#71717a]">=</span>
                            <span className="text-white">{typeof f.value === "number" ? f.value.toFixed(1) : String(f.value)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {lastDecision.explanation?.top_features && lastDecision.explanation.top_features.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-[#27272A]" data-testid="explain-top-features">
                      <div className="overline mb-2">Top Features (this decision)</div>
                      <div className="space-y-1">
                        {lastDecision.explanation.top_features.slice(0, 4).map((f, i) => (
                          <div key={i} className="flex items-center gap-2 text-[10px] font-mono">
                            <span className="text-[#0055FF]">{i + 1}.</span>
                            <span className="text-[#a1a1aa] flex-1">{f.name}</span>
                            <span className="text-white">{f.value !== null && f.value !== undefined ? (typeof f.value === "number" ? f.value.toFixed(2) : f.value) : "—"}</span>
                            {f.weight !== null && f.weight !== undefined && (
                              <span className="text-[#71717a] w-16 text-right">{(f.weight * 100).toFixed(1)}%</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {lastDecision.explanation?.counterfactual && (
                    <div className="mt-3 pt-3 border-t border-[#27272A] text-[10px] font-mono text-[#FFCC00]" data-testid="explain-counterfactual">
                      ⚡ Counterfactual: {lastDecision.explanation.counterfactual}
                    </div>
                  )}
                </div>
              )}

              {/* Real inference runner */}
              <div className="mt-4 border border-[#27272A] p-3" data-testid="real-inference-card">
                <div className="flex items-center justify-between mb-2">
                  <Overline>Real Workload Runner</Overline>
                  <button
                    data-testid="run-real-inference-btn"
                    onClick={runRealInference}
                    disabled={realInferenceLoading || !selectedDevice}
                    className="font-mono uppercase text-[10px] tracking-wider px-3 py-1.5 border border-[#0055FF] text-[#0055FF] hover:bg-[#0055FF] hover:text-white disabled:opacity-40 flex items-center gap-2"
                  >
                    {realInferenceLoading ? <><CircleNotch size={10} className="animate-spin" /> Running...</> : <><Lightning size={10} weight="fill" /> Execute REAL Inference</>}
                  </button>
                </div>
                {lastRealInference ? (
                  <div className="grid grid-cols-4 gap-3 text-xs font-mono">
                    <div>
                      <div className="text-[10px] text-[#71717a] uppercase">Latency</div>
                      <div className="text-white text-lg">{lastRealInference.latency_ms}<span className="text-[10px] text-[#71717a] ml-1">ms</span></div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[#71717a] uppercase">CPU (proc)</div>
                      <div className="text-white text-lg">{lastRealInference.cpu_percent}<span className="text-[10px] text-[#71717a] ml-1">%</span></div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[#71717a] uppercase">Memory</div>
                      <div className="text-white text-lg">{lastRealInference.memory_mb}<span className="text-[10px] text-[#71717a] ml-1">MB</span></div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[#71717a] uppercase">FLOPs</div>
                      <div className="text-white text-lg">{(lastRealInference.workload_flops / 1e6).toFixed(1)}<span className="text-[10px] text-[#71717a] ml-1">M</span></div>
                    </div>
                  </div>
                ) : (
                  <div className="text-[11px] font-mono text-[#52525b]">
                    Runs a REAL numpy compute-bound workload with psutil-measured CPU/memory. No random simulation — for academic validity.
                  </div>
                )}
              </div>
            </Cell>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <KPI label="Edge Latency" value={summary.edge?.avg_latency ?? "--"} unit="ms" tone="good" testId="kpi-edge-latency" />
              <KPI label="Cloud Latency" value={selectedDevice?.connected ? (summary.cloud?.avg_latency ?? "--") : "ERR"} unit="ms" tone={selectedDevice?.connected ? "accent" : "bad"} testId="kpi-cloud-latency" />
              <KPI label="Edge CPU" value={summary.edge?.avg_cpu ?? "--"} unit="%" tone="warn" testId="kpi-edge-cpu" />
              <KPI label="Cloud Cost (10m)" value={summary.cloud?.total_cost?.toFixed?.(4) ?? "0.0000"} unit="USD" testId="kpi-cloud-cost" />
            </div>

            <Cell className="p-4" testId="latency-chart-card">
              <div className="flex items-center justify-between mb-3">
                <Overline>Latency · Edge vs Cloud</Overline>
                <span className="font-mono text-[10px] text-[#71717a]">ms · lower is better</span>
              </div>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={series} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#1a1a1a" />
                    <XAxis dataKey="t" stroke="#52525b" tick={{ fontFamily: "JetBrains Mono", fontSize: 10 }} />
                    <YAxis stroke="#52525b" tick={{ fontFamily: "JetBrains Mono", fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "#000", border: "1px solid #27272A", borderRadius: 0, fontFamily: "JetBrains Mono", fontSize: 11 }} />
                    <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.1em" }} />
                    <Line type="monotone" dataKey="edge_latency" name="Edge" stroke="#00E676" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="cloud_latency" name="Cloud" stroke="#0055FF" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Cell>

            {/* Feature Importance + Route Distribution */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Cell className="p-4" testId="feature-importance-card">                <Overline>Feature Importance · {engineFullName(engineName)}</Overline>
                <div className="h-48 mt-3">
                  {featureImportanceData.length ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={featureImportanceData} layout="vertical" margin={{ left: 0 }}>
                        <XAxis type="number" stroke="#52525b" tick={{ fontFamily: "JetBrains Mono", fontSize: 9 }} />
                        <YAxis dataKey="feature" type="category" stroke="#52525b" tick={{ fontFamily: "JetBrains Mono", fontSize: 9 }} width={110} />
                        <Tooltip contentStyle={{ background: "#000", border: "1px solid #27272A", borderRadius: 0, fontFamily: "JetBrains Mono", fontSize: 11 }} />
                        <Bar dataKey="value" fill="#0055FF" />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-[#52525b] font-mono text-xs">
                      Rule-based engine has no feature importance
                    </div>
                  )}
                </div>
              </Cell>

              <Cell className="p-4" testId="route-distribution-card">
                <Overline>Route Distribution · Last 30m</Overline>
                <div className="h-48 mt-3 flex items-center justify-center">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={distributionData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={45} outerRadius={75} paddingAngle={2}>
                        {distributionData.map((d) => (
                          <RCell key={d.name} fill={ROUTE_COLORS[d.name]} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ background: "#000", border: "1px solid #27272A", borderRadius: 0, fontFamily: "JetBrains Mono", fontSize: 11 }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex justify-around text-[10px] font-mono">
                  {distributionData.map((d) => (
                    <div key={d.name} className="flex items-center gap-1">
                      <span className="w-2 h-2" style={{ background: ROUTE_COLORS[d.name] }} />
                      <span className="text-[#a1a1aa] uppercase">{d.name}</span>
                      <span className="text-white">{d.value}</span>
                    </div>
                  ))}
                </div>
              </Cell>
            </div>

            {/* BENCHMARK — Research experiment output */}
            <Cell className="p-4" testId="benchmark-card">
              <div className="flex items-center justify-between mb-3 gap-3 flex-wrap">
                <div>
                  <Overline>Comparative Experiment · Research Instrument</Overline>
                  <div className="text-[11px] font-mono text-[#71717a] mt-1">
                    Runs the same reproducible scenarios through all 4 policies. Outputs CSV + PNG + PDF for the thesis appendix.
                  </div>
                </div>
              </div>

              {/* Scenario selector */}
              <div className="mb-3 border border-[#27272A] p-2" data-testid="scenario-selector">
                <div className="flex items-center justify-between mb-2">
                  <Overline>Scenario</Overline>
                  <span className="font-mono text-[9px] text-[#71717a]">{scenarios.length} available</span>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-1">
                  {scenarios.map((s) => (
                    <button
                      key={s.id}
                      data-testid={`scenario-${s.id}-btn`}
                      onClick={() => setSelectedScenario(s.id)}
                      className={`text-left px-2 py-1.5 border transition-colors ${
                        selectedScenario === s.id
                          ? "border-[#0055FF] bg-[#0055FF]/10 text-white"
                          : "border-[#27272A] text-[#a1a1aa] hover:border-[#52525b] hover:text-white"
                      }`}
                    >
                      <div className="font-mono text-[10px] font-semibold">{s.name}</div>
                    </button>
                  ))}
                </div>
                {scenarios.find((s) => s.id === selectedScenario)?.description && (
                  <div className="text-[10px] font-mono text-[#71717a] mt-2 px-1">
                    ↳ {scenarios.find((s) => s.id === selectedScenario)?.description}
                  </div>
                )}
              </div>

              {/* Action buttons row */}
              <div className="flex flex-wrap gap-2 mb-3">
                <button
                  data-testid="run-complete-benchmark-btn"
                  onClick={runCompleteBenchmark}
                  disabled={completeRunning}
                  className="font-mono uppercase text-[10px] tracking-wider px-3 py-2 bg-[#0055FF] text-white hover:bg-[#0044CC] disabled:opacity-40 flex items-center gap-2 font-semibold"
                >
                  {completeRunning ? <><CircleNotch size={10} className="animate-spin" /> Running {completeStatus?.completed?.length ?? 0}/{completeStatus?.total ?? 8}...</> : <><Rocket size={10} weight="fill" /> Run Complete Benchmark</>}
                </button>
                <button
                  data-testid="run-benchmark-btn"
                  onClick={() => runBenchmark(500)}
                  disabled={benchmarkRunning}
                  className="font-mono uppercase text-[10px] tracking-wider px-3 py-2 border border-[#27272A] text-[#a1a1aa] hover:border-white hover:text-white disabled:opacity-40 flex items-center gap-2"
                >
                  {benchmarkRunning ? <><CircleNotch size={10} className="animate-spin" /> Running...</> : <><ChartLineUp size={10} /> Run 500 Scenarios</>}
                </button>
                <button
                  data-testid="run-ci-btn"
                  onClick={runCI}
                  disabled={ciRunning}
                  className="font-mono uppercase text-[10px] tracking-wider px-3 py-2 border border-[#27272A] text-[#a1a1aa] hover:border-white hover:text-white disabled:opacity-40 flex items-center gap-2"
                >
                  {ciRunning ? <><CircleNotch size={10} className="animate-spin" /> Running CI...</> : <><Lightning size={10} /> Run CI (5 seeds)</>}
                </button>
                <button
                  data-testid="export-report-btn"
                  onClick={generateReport}
                  disabled={reportGenerating}
                  className="font-mono uppercase text-[10px] tracking-wider px-3 py-2 border border-[#27272A] text-[#a1a1aa] hover:border-white hover:text-white disabled:opacity-40 flex items-center gap-2"
                >
                  {reportGenerating ? <><CircleNotch size={10} className="animate-spin" /> Generating...</> : <><Package size={10} /> Export Thesis Report (PDF)</>}
                </button>
                {benchmark && (
                  <a
                    data-testid="download-csv-btn"
                    href={`${API}/benchmark/artefact/benchmark_records.csv`}
                    className="font-mono uppercase text-[10px] tracking-wider px-3 py-2 border border-[#27272A] text-[#a1a1aa] hover:text-white hover:border-white flex items-center gap-2"
                  >
                    <Database size={10} /> CSV
                  </a>
                )}
              </div>

              {/* Complete-benchmark progress + completion stats */}
              {(completeRunning || completeStatus?.status === "done") && (
                <div className="mb-3 border border-[#0055FF] p-3 bg-[#0055FF]/5" data-testid="complete-benchmark-panel">
                  {completeRunning && (
                    <>
                      <div className="flex items-center justify-between mb-2">
                        <Overline>Complete Benchmark · Progress</Overline>
                        <span className="font-mono text-[10px] text-[#0055FF]">
                          {completeStatus?.elapsed_s ?? 0}s elapsed
                        </span>
                      </div>
                      <div className="font-mono text-xs text-white">
                        Current: {completeStatus?.current_scenario || "starting..."}
                      </div>
                      <div className="mt-2 h-2 bg-[#141414]">
                        <div
                          className="h-full bg-[#0055FF] transition-all"
                          style={{ width: `${(((completeStatus?.completed?.length ?? 0) / (completeStatus?.total || 8))) * 100}%` }}
                        />
                      </div>
                      <div className="mt-1 font-mono text-[10px] text-[#71717a]">
                        Completed: {completeStatus?.completed?.join(" · ") || "—"}
                      </div>
                    </>
                  )}
                  {!completeRunning && completeStatus?.status === "done" && masterSummary && (
                    <div data-testid="completion-stats">
                      <Overline>Benchmark Completed</Overline>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-3">
                        <div>
                          <div className="text-[10px] font-mono text-[#71717a] uppercase">Scenarios</div>
                          <div className="font-mono text-2xl text-white">{masterSummary.n_scenarios}</div>
                        </div>
                        <div>
                          <div className="text-[10px] font-mono text-[#71717a] uppercase">Runs</div>
                          <div className="font-mono text-2xl text-white">{masterSummary.n_scenarios * 5}</div>
                        </div>
                        <div>
                          <div className="text-[10px] font-mono text-[#71717a] uppercase">Policy Evals</div>
                          <div className="font-mono text-2xl text-white">{masterSummary.n_scenarios * 4}</div>
                        </div>
                        <div>
                          <div className="text-[10px] font-mono text-[#71717a] uppercase">Charts</div>
                          <div className="font-mono text-2xl text-white">{masterSummary.n_scenarios * 5}</div>
                        </div>
                        <div>
                          <div className="text-[10px] font-mono text-[#71717a] uppercase">Reports</div>
                          <div className="font-mono text-2xl text-white">{masterSummary.n_scenarios}</div>
                        </div>
                      </div>

                      <div className="mt-4 pt-3 border-t border-[#0055FF]/30">
                        <div className="flex items-center gap-3 flex-wrap">
                          <span className="overline">Overall Ranking (by reward):</span>
                          {masterSummary.overall_ranking_by_reward?.map((e, i) => (
                            <span key={e} className="font-mono text-xs">
                              <span className="text-[#0055FF]">{i + 1}.</span>{" "}
                              <span className="text-white">{e}</span>
                            </span>
                          ))}
                        </div>
                        <div className="mt-2 flex items-center gap-3 flex-wrap">
                          <span className="overline">Scenario Wins:</span>
                          {Object.entries(masterSummary.wins_by_engine || {}).map(([e, n]) => (
                            <span key={e} className="font-mono text-xs">
                              <span className="text-[#a1a1aa]">{e}:</span>{" "}
                              <span className="text-[#00E676]">{n}</span>
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="mt-4 flex flex-wrap gap-2">
                        <a
                          data-testid="download-master-pdf-btn"
                          href={`${API}/experiments/master_summary.pdf`}
                          className="font-mono uppercase text-[10px] tracking-wider px-3 py-2 border border-[#00E676] text-[#00E676] hover:bg-[#00E676] hover:text-black flex items-center gap-2"
                        >
                          <Package size={10} weight="fill" /> Master Summary (PDF)
                        </a>
                        <a
                          data-testid="download-master-csv-btn"
                          href={`${API}/experiments/master_summary.csv`}
                          className="font-mono uppercase text-[10px] tracking-wider px-3 py-2 border border-[#27272A] text-[#a1a1aa] hover:text-white hover:border-white flex items-center gap-2"
                        >
                          <Database size={10} /> Master (CSV)
                        </a>
                        <a
                          data-testid="download-zip-btn"
                          href={`${API}/experiments/download_zip`}
                          className="font-mono uppercase text-[10px] tracking-wider px-3 py-2 border border-[#FFCC00] text-[#FFCC00] hover:bg-[#FFCC00] hover:text-black flex items-center gap-2"
                        >
                          <Database size={10} weight="fill" /> Export All Results (ZIP)
                        </a>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {benchmark && benchmark.summary && (
                <>
                  {/* Summary table */}
                  <div className="mt-2 overflow-x-auto">
                    <table className="w-full font-mono text-xs" data-testid="benchmark-table">
                      <thead>
                        <tr className="text-[10px] text-[#71717a] uppercase tracking-wider border-b border-[#27272A]">
                          <th className="text-left py-2">Policy</th>
                          <th className="text-right">Latency p50</th>
                          <th className="text-right">Latency p95</th>
                          <th className="text-right">Cost total</th>
                          <th className="text-right">Decision µs</th>
                          <th className="text-right">Success</th>
                          <th className="text-right">Reward</th>
                          <th className="text-right">Agree w/ Rule</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(benchmark.summary).map(([engine, m]) => (
                          <tr key={engine} className="border-b border-[#141414]">
                            <td className="py-2 text-white">{engine}</td>
                            <td className="text-right text-[#a1a1aa]">{m.latency_ms_p50}<span className="text-[9px] text-[#52525b] ml-1">ms</span></td>
                            <td className="text-right text-[#a1a1aa]">{m.latency_ms_p95}<span className="text-[9px] text-[#52525b] ml-1">ms</span></td>
                            <td className="text-right text-[#a1a1aa]">${m.cost_usd_total}</td>
                            <td className="text-right text-[#a1a1aa]">{m.decision_time_us_mean}</td>
                            <td className="text-right text-[#00E676]">{(m.success_rate * 100).toFixed(1)}%</td>
                            <td className="text-right text-[#a1a1aa]">{m.reward_mean}</td>
                            <td className="text-right text-[#a1a1aa]">{(m.agreement_with_rule * 100).toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Chart previews */}
                  <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-2" data-testid="benchmark-charts">
                    {[
                      ["chart_latency_boxplot.png", "Latency"],
                      ["chart_decision_time.png", "Decision Time"],
                      ["chart_cost.png", "Cost"],
                      ["chart_route_distribution.png", "Route Distribution"],
                      ["chart_reward.png", "Reward"],
                    ].map(([file, label]) => (
                      <a
                        key={file}
                        href={`${API}/benchmark/artefact/${file}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="group border border-[#27272A] hover:border-[#0055FF] transition-colors"
                      >
                        <img
                          src={`${API}/benchmark/artefact/${file}?t=${benchmark.meta?.generated_at || Date.now()}`}
                          alt={label}
                          className="w-full h-24 object-contain bg-white"
                        />
                        <div className="text-[10px] font-mono text-[#a1a1aa] p-1.5 group-hover:text-white uppercase tracking-wider">
                          {label} ↗
                        </div>
                      </a>
                    ))}
                  </div>
                  <div className="mt-2 text-[10px] font-mono text-[#71717a]">
                    n = {benchmark.meta?.n_scenarios || "—"} scenarios · seed = {benchmark.meta?.seed || "—"} · click a chart to open full size · CSV/JSON also downloadable
                  </div>
                </>
              )}

              {!benchmark && (
                <div className="mt-4 text-[11px] font-mono text-[#52525b] py-8 text-center border border-dashed border-[#27272A]">
                  No experiment has been run yet. Click <span className="text-white">Run 500 Scenarios</span> to generate the comparative dataset for the thesis.
                </div>
              )}

              {/* CI RESULTS — statistical validation */}
              {ciResult && ciResult.aggregated && (
                <div className="mt-6 border-t border-[#27272A] pt-4" data-testid="ci-results">
                  <div className="flex items-center justify-between mb-3">
                    <Overline>Statistical Validation · mean ± 95% CI</Overline>
                    <span className="font-mono text-[10px] text-[#71717a]">
                      {ciResult.meta?.n_runs} seeds × {ciResult.meta?.n_scenarios_per_seed} samples · t-crit={ciResult.meta?.t_critical_95}
                    </span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full font-mono text-xs" data-testid="ci-table">
                      <thead>
                        <tr className="text-[10px] text-[#71717a] uppercase tracking-wider border-b border-[#27272A]">
                          <th className="text-left py-2">Policy</th>
                          <th className="text-right">Latency p50 (ms)</th>
                          <th className="text-right">Latency p95 (ms)</th>
                          <th className="text-right">Cost total ($)</th>
                          <th className="text-right">Decision (µs)</th>
                          <th className="text-right">Reward</th>
                          <th className="text-right">Success rate</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(ciResult.aggregated).map(([engine, m]) => (
                          <tr key={engine} className="border-b border-[#141414]">
                            <td className="py-2 text-white">{engine}</td>
                            <td className="text-right text-[#a1a1aa]">{m.latency_ms_p50 ? `${m.latency_ms_p50.mean.toFixed(2)} ± ${m.latency_ms_p50.ci95.toFixed(2)}` : "—"}</td>
                            <td className="text-right text-[#a1a1aa]">{m.latency_ms_p95 ? `${m.latency_ms_p95.mean.toFixed(1)} ± ${m.latency_ms_p95.ci95.toFixed(1)}` : "—"}</td>
                            <td className="text-right text-[#a1a1aa]">{m.cost_usd_total ? `${m.cost_usd_total.mean.toFixed(3)} ± ${m.cost_usd_total.ci95.toFixed(3)}` : "—"}</td>
                            <td className="text-right text-[#a1a1aa]">{m.decision_time_us_mean ? `${m.decision_time_us_mean.mean.toFixed(1)} ± ${m.decision_time_us_mean.ci95.toFixed(1)}` : "—"}</td>
                            <td className="text-right text-[#a1a1aa]">{m.reward_mean ? `${m.reward_mean.mean.toFixed(2)} ± ${m.reward_mean.ci95.toFixed(2)}` : "—"}</td>
                            <td className="text-right text-[#00E676]">{m.success_rate ? `${(m.success_rate.mean * 100).toFixed(1)}% ± ${(m.success_rate.ci95 * 100).toFixed(1)}%` : "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="mt-2 text-[10px] font-mono text-[#71717a]">
                    seeds: {ciResult.meta?.seeds?.join(", ")} · aggregated with Student t 95% CI
                  </div>
                </div>
              )}
            </Cell>

            {/* EXPERIMENT ASSISTANT — post-TCC preview */}
            <Cell className="p-4" testId="assistant-card">
              <div className="flex items-center justify-between mb-3 gap-2">
                <div>
                  <Overline>Experiment Assistant</Overline>
                  <div className="text-[11px] font-mono text-[#71717a] mt-1 italic">
                    Future Work (Post-TCC Preview) — ask questions in natural language about the experiments. Grounded in the SQLite decision log and the latest benchmark/CI results.
                  </div>
                </div>
                <span className="font-mono text-[9px] text-[#71717a] px-2 py-1 border border-[#27272A]">
                  Claude Sonnet 4.5
                </span>
              </div>

              {/* Suggested prompts */}
              <div className="flex flex-wrap gap-1 mb-3" data-testid="assistant-suggestions">
                {[
                  "Which policy achieved the lowest latency in the last experiment?",
                  "Compare Rule-Based and Q-Learning.",
                  "Which policy achieved the highest success rate?",
                  "Why did Random Forest choose Edge?",
                  "Which features contributed most to this decision?",
                  "Summarize the latest experiment results.",
                  "Generate a thesis-ready summary for this experiment.",
                  "Compare all four policies for the Network Failure scenario.",
                  "Show the confidence intervals for Random Forest.",
                  "Which scenario produced the highest cloud utilization?",
                ].map((prompt) => (
                  <button
                    key={prompt}
                    data-testid="assistant-suggestion-btn"
                    disabled={assistantLoading}
                    onClick={() => askAssistant(prompt)}
                    className="font-mono text-[10px] px-2 py-1 border border-[#27272A] text-[#a1a1aa] hover:text-white hover:border-[#0055FF] transition-colors disabled:opacity-40 text-left"
                  >
                    {prompt}
                  </button>
                ))}
              </div>

              {/* Conversation */}
              <div className="border border-[#27272A] bg-black min-h-[120px] max-h-[400px] overflow-y-auto thin-scroll" data-testid="assistant-messages">
                {assistantMessages.length === 0 ? (
                  <div className="p-4 text-center font-mono text-[11px] text-[#52525b]">
                    Ask something about your experiments — click a suggestion or type below.
                  </div>
                ) : (
                  <div className="divide-y divide-[#141414]">
                    {assistantMessages.map((m, i) => (
                      <div key={i} className="p-3">
                        <div className="flex items-start gap-2">
                          <span className="font-mono text-[10px] text-[#0055FF] mt-0.5">You</span>
                          <span className="text-[#71717a] font-mono text-[9px] mt-0.5">{m.ts}</span>
                        </div>
                        <div className="font-mono text-[12px] text-white mt-1 pl-8">{m.q}</div>
                        {m.a && (
                          <>
                            <div className="mt-3 font-mono text-[10px] text-[#00E676]">Assistant</div>
                            <div className="font-mono text-[11px] text-[#e4e4e7] mt-1 pl-8 whitespace-pre-wrap leading-relaxed">{m.a}</div>
                          </>
                        )}
                        {!m.a && (
                          <div className="mt-3 flex items-center gap-2 pl-8">
                            <CircleNotch size={11} className="animate-spin text-[#0055FF]" />
                            <span className="font-mono text-[10px] text-[#71717a]">Thinking...</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Input */}
              <form
                onSubmit={(e) => { e.preventDefault(); askAssistant(); }}
                className="flex gap-2 mt-3"
                data-testid="assistant-input-form"
              >
                <input
                  data-testid="assistant-input"
                  type="text"
                  value={assistantInput}
                  onChange={(e) => setAssistantInput(e.target.value)}
                  placeholder="Ask about latency, cost, agreement, seeds..."
                  disabled={assistantLoading}
                  className="flex-1 bg-black border border-[#27272A] px-3 py-2 font-mono text-xs text-white placeholder:text-[#52525b] focus:border-[#0055FF] outline-none"
                />
                <button
                  data-testid="assistant-send-btn"
                  type="submit"
                  disabled={assistantLoading || !assistantInput.trim()}
                  className="font-mono uppercase text-[10px] tracking-wider px-4 py-2 bg-[#0055FF] text-white hover:bg-[#0044CC] disabled:opacity-40"
                >
                  {assistantLoading ? "..." : "Ask"}
                </button>
              </form>
              <div className="mt-2 text-[9px] font-mono text-[#52525b] italic">
                Future Work — not part of the core thesis evaluation
              </div>
            </Cell>
          </section>


          {/* RIGHT */}
          <section className="col-span-12 lg:col-span-3 space-y-4">
            <Cell testId="model-repo-panel">
              <div className="px-4 pt-4 pb-3 border-b border-[#27272A] flex items-center justify-between">
                <Overline>Model Repository</Overline>
                <Package size={14} className="text-[#a1a1aa]" />
              </div>
              <ul className="divide-y divide-[#27272A] max-h-[240px] overflow-y-auto thin-scroll">
                {models.map((m) => {
                  const active = m.id === selectedModelId;
                  return (
                    <li key={m.id} data-testid={`model-${m.name}`} onClick={() => setSelectedModelId(m.id)}
                      className={`px-4 py-3 cursor-pointer transition-colors ${
                        active ? "bg-[#0055FF]/10 border-l-2 border-[#0055FF]" : "hover:bg-[#141414] border-l-2 border-transparent"
                      }`}>
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-sm">{m.name}</span>
                        <span className="font-mono text-[10px] text-[#71717a]">v{m.version}</span>
                      </div>
                      <div className="mt-1 flex gap-3 text-[10px] font-mono text-[#71717a]">
                        <span>{m.size_mb} MB</span>
                        <span>acc {(m.accuracy * 100).toFixed(0)}%</span>
                        <span>{m.task}</span>
                      </div>
                    </li>
                  );
                })}
              </ul>
              <div className="p-3 border-t border-[#27272A]">
                <button data-testid="deploy-model-btn" onClick={deployModel} disabled={!selectedModelId || deploying}
                  className="w-full font-mono uppercase text-xs tracking-wider px-4 py-2.5 bg-[#0055FF] text-white hover:bg-[#0044CC] disabled:bg-[#141414] disabled:text-[#52525b] transition-colors flex items-center justify-center gap-2">
                  {deploying ? <><CircleNotch size={12} className="animate-spin" /> Deploying...</> : <><Rocket size={12} weight="fill" /> Deploy to {selectedDevice?.name || "device"}</>}
                </button>
              </div>
            </Cell>

            <Cell testId="decisions-history-panel">
              <div className="px-4 pt-4 pb-3 border-b border-[#27272A] flex items-center justify-between">
                <Overline>Recent Decisions</Overline>
                <Database size={14} className="text-[#a1a1aa]" />
              </div>
              <ul className="divide-y divide-[#27272A] max-h-[240px] overflow-y-auto thin-scroll">
                {decisionHistory.length === 0 && (
                  <li className="px-4 py-6 text-center font-mono text-xs text-[#52525b]">No decisions yet</li>
                )}
                {decisionHistory.map((d) => (
                  <li key={d.id} className="px-4 py-2">
                    <div className="flex items-center justify-between font-mono text-xs">
                      <span style={{ color: ROUTE_COLORS[d.route] }}>{d.route.toUpperCase()}</span>
                      <span className="text-[#71717a] text-[10px]">{(d.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="text-[10px] font-mono text-[#71717a] truncate">{d.reason}</div>
                  </li>
                ))}
              </ul>
            </Cell>

            <Cell className="bg-black" testId="logs-panel">
              <div className="px-4 pt-4 pb-3 border-b border-[#27272A] flex items-center justify-between">
                <Overline>Runtime Logs</Overline>
                <Terminal size={14} className="text-[#a1a1aa]" />
              </div>
              <div ref={logsRef} className="term-log px-4 py-3 max-h-[260px] overflow-y-auto thin-scroll">
                {logs.map((l, i) => (
                  <div key={i}>
                    <span className="ts">[{l.t}]</span>{" "}
                    <span className={`lvl-${l.level === "err" ? "err" : l.level === "warn" ? "warn" : l.level === "ok" ? "ok" : "info"}`}>{l.level.toUpperCase()}</span>{" "}
                    <span>{l.msg}</span>
                  </div>
                ))}
                <div className="caret" />
              </div>
            </Cell>
          </section>
        </div>

        <footer className="mt-8 border-t border-[#27272A] pt-4 pb-8">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 font-mono text-[10px] text-[#52525b] uppercase tracking-wider">
            <span>artefact: edge-cloud-orchestrator</span>
            <span>·</span>
            <span>decision engine: rule-based · decision tree · random forest · q-learning</span>
            <span>·</span>
            <span>runtime: greengrass v2</span>
            <span>·</span>
            <span>region: us-east-1</span>
          </div>
        </footer>
      </main>

      {showArch && <ArchitectureModal onClose={() => setShowArch(false)} status={engineStatus} />}
    </div>
  );
}

function engineFullName(k) {
  return { rule: "RuleBased", dt: "DecisionTree", rf: "RandomForest", ql: "QLearning" }[k] || "RuleBased";
}

function ArchitectureModal({ onClose, status }) {
  return (
    <div data-testid="architecture-modal" className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-4" onClick={onClose}>
      <div onClick={(e) => e.stopPropagation()} className="bg-[#0A0A0A] border border-[#27272A] max-w-5xl w-full max-h-[90vh] overflow-y-auto thin-scroll"
        style={{ backgroundImage: "linear-gradient(rgba(5,5,5,0.9), rgba(5,5,5,0.95)), url('https://images.pexels.com/photos/37730212/pexels-photo-37730212.jpeg')", backgroundSize: "cover" }}>
        <div className="p-6 md:p-10">
          <div className="flex justify-between items-start mb-8">
            <div>
              <Overline>System Architecture</Overline>
              <h2 className="mt-2 font-mono text-2xl md:text-3xl tracking-tighter">AI-Driven Infrastructure Orchestration for Edge–Cloud Environments</h2>
              <p className="mt-2 text-sm text-[#a1a1aa] max-w-2xl">
                Research platform for AI-driven infrastructure orchestration across Edge–Cloud environments using NVIDIA Jetson and AWS.
                <br /><br />
                The platform evaluates four intelligent decision policies (Rule-Based, Decision Tree, Random Forest and Q-Learning) for dynamically routing AI inference workloads between edge, cloud and hybrid execution according to real-time infrastructure conditions.
                <br /><br />
                The objective of this research is to compare orchestration strategies for optimizing latency, cost, resource utilisation and operational resilience.
              </p>
            </div>
            <button data-testid="close-architecture-btn" onClick={onClose}
              className="font-mono text-xs uppercase tracking-wider px-3 py-1.5 border border-[#27272A] hover:border-white text-[#a1a1aa] hover:text-white">
              Close
            </button>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="border border-[#27272A] p-5">
              <Overline>AWS Cloud (EC2 · Greengrass v2)</Overline>
              <ul className="mt-3 space-y-2 font-mono text-sm">
                <li className="flex items-center gap-2"><CloudArrowUp size={14} className="text-[#0055FF]" /> AWS IoT Greengrass v2 core</li>
                <li className="flex items-center gap-2"><HardDrives size={14} className="text-[#0055FF]" /> EC2 · FastAPI + Decision Engine</li>
                <li className="flex items-center gap-2"><Database size={14} className="text-[#0055FF]" /> SQLite (decisions) + MongoDB (telemetry)</li>
                <li className="flex items-center gap-2"><Package size={14} className="text-[#0055FF]" /> S3 model repository</li>
                <li className="flex items-center gap-2"><ChartLineUp size={14} className="text-[#0055FF]" /> CloudWatch metrics & alarms</li>
              </ul>
            </div>
            <div className="border border-[#27272A] p-5">
              <Overline>NVIDIA Jetson Edge</Overline>
              <ul className="mt-3 space-y-2 font-mono text-sm">
                <li className="flex items-center gap-2"><Cube size={14} className="text-[#00E676]" /> Docker container runtime</li>
                <li className="flex items-center gap-2"><Cpu size={14} className="text-[#00E676]" /> Greengrass Core nucleus</li>
                <li className="flex items-center gap-2"><Lightning size={14} className="text-[#00E676]" /> YOLO/MobileNet inference engine</li>
                <li className="flex items-center gap-2"><Broadcast size={14} className="text-[#00E676]" /> MQTT client · IoT Core</li>
                <li className="flex items-center gap-2"><Warning size={14} className="text-[#00E676]" /> Offline buffering & retry</li>
              </ul>
            </div>
          </div>

          <div className="mt-6 border border-[#27272A] p-5">
            <Overline>Research Instrumentation</Overline>
            <div className="mt-3 grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <div className="flex items-center gap-2 font-mono text-white">
                  <MapPin size={14} className="text-[#0055FF]" /> Scenario Manager
                </div>
                <p className="mt-1 text-xs text-[#a1a1aa]">
                  8 named reproducible scenarios: Mixed Traffic, Factory Normal, Network Failure, High CPU Load, Large Model, Cloud Congestion, Low Bandwidth, High Priority.
                </p>
              </div>
              <div>
                <div className="flex items-center gap-2 font-mono text-white">
                  <ChartLineUp size={14} className="text-[#00E676]" /> Multi-Seed CI (95%)
                </div>
                <p className="mt-1 text-xs text-[#a1a1aa]">
                  Statistical validation via Student-t 95% confidence intervals across 5 seeds — eliminates the &quot;was it luck?&quot; question.
                </p>
              </div>
              <div>
                <div className="flex items-center gap-2 font-mono text-white">
                  <Brain size={14} className="text-[#FFCC00]" /> Explain Decision (XAI)
                </div>
                <p className="mt-1 text-xs text-[#a1a1aa]">
                  Every prediction ships with contributing factors (met ✓/✗), ranked feature importance, and a counterfactual sentence when applicable.
                </p>
              </div>
              <div>
                <div className="flex items-center gap-2 font-mono text-white">
                  <Package size={14} className="text-[#FF3333]" /> Thesis Report (PDF)
                </div>
                <p className="mt-1 text-xs text-[#a1a1aa]">
                  One-click generation of a 12-page academic report: cover, executive summary, experimental configuration, comparative results with ±95% CI, five figures (IEEE-style captions), sample explained decisions, conclusions and references. Formal University of Bedfordshire branding and embedded PDF metadata (author, supervisor, module code).
                </p>
              </div>
              <div className="md:col-span-2 border-t border-[#27272A] pt-3 mt-1">
                <div className="flex items-center gap-2 font-mono text-white">
                  <Terminal size={14} className="text-[#0055FF]" /> Experiment Assistant
                </div>
                <p className="mt-1 text-xs text-[#a1a1aa]">
                  Natural-language Q&amp;A over experimental data — powered by Claude Sonnet 4.5. Grounded in the SQLite decision log and the latest benchmark/CI results; every answer cites its source, never invents numbers.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-6 border border-[#27272A] p-5">
            <Overline>Decision Engine · Four Policies (comparative study)</Overline>
            <div className="mt-3 grid md:grid-cols-4 gap-4 text-sm">
              <div>
                <div className="flex items-center gap-2 font-mono"><Path size={14} className="text-[#FFCC00]" /> Rule-Based</div>
                <p className="mt-1 text-xs text-[#a1a1aa]">Interpretable heuristics. Zero-training baseline.</p>
                <div className="mt-2 text-[10px] font-mono text-[#71717a]">acc: 100% (label source)</div>
              </div>
              <div>
                <div className="flex items-center gap-2 font-mono"><TreeStructure size={14} className="text-[#00E676]" /> Decision Tree</div>
                <p className="mt-1 text-xs text-[#a1a1aa]">Learns policy surface offline. Highly interpretable.</p>
                <div className="mt-2 text-[10px] font-mono text-[#71717a]">
                  acc: {((status?.DecisionTree?.metrics?.accuracy ?? 0) * 100).toFixed(2)}%
                </div>
              </div>
              <div>
                <div className="flex items-center gap-2 font-mono"><Brain size={14} className="text-[#0055FF]" /> Random Forest</div>
                <p className="mt-1 text-xs text-[#a1a1aa]">Ensemble of 80 trees. Robust to noise.</p>
                <div className="mt-2 text-[10px] font-mono text-[#71717a]">
                  acc: {((status?.RandomForest?.metrics?.accuracy ?? 0) * 100).toFixed(2)}%
                </div>
              </div>
              <div>
                <div className="flex items-center gap-2 font-mono"><Lightning size={14} className="text-[#FF3333]" /> Q-Learning</div>
                <p className="mt-1 text-xs text-[#a1a1aa]">Reinforcement-learning agent — learns from reward feedback online.</p>
                <div className="mt-2 text-[10px] font-mono text-[#71717a]">
                  reward: {status?.QLearning?.metrics?.mean_reward_last_500 ?? "—"}
                </div>
              </div>
            </div>
            <div className="mt-4 text-xs font-mono text-[#71717a]">
              Progression: <span className="text-[#FFCC00]">fixed rules</span> → <span className="text-[#00E676]">supervised tree</span> → <span className="text-[#0055FF]">ensemble</span> → <span className="text-[#FF3333]">online RL agent</span>
            </div>
            <div className="mt-1 text-xs font-mono text-[#71717a]">
              Classes: <span className="text-[#00E676]">edge</span> · <span className="text-[#0055FF]">cloud</span> · <span className="text-[#FFCC00]">hybrid</span>
            </div>
          </div>

          <div className="mt-6 grid md:grid-cols-2 gap-4">
            <div className="border border-[#27272A] p-5">
              <Overline>Research Artefact</Overline>
              <p className="mt-3 text-sm text-[#a1a1aa]">
                An AI-driven infrastructure orchestration platform for Edge–Cloud environments that integrates NVIDIA Jetson, AWS IoT Greengrass and pre-trained AI models.
                <br /><br />
                The platform dynamically orchestrates AI inference workloads through four decision policies, enabling comparative evaluation of latency, cost, resource utilisation, reliability and adaptive behaviour under multiple operational scenarios.
              </p>
            </div>
            <div className="border border-[#27272A] p-5">
              <Overline>Research Contribution</Overline>
              <p className="mt-3 text-sm text-[#a1a1aa]">
                Design, implementation and experimental evaluation of an AI-driven infrastructure orchestration platform for Edge–Cloud environments.
                <br /><br />
                The study compares four intelligent orchestration policies deployed on NVIDIA Jetson and AWS IoT Greengrass, demonstrating how different decision strategies affect AI workload placement, latency, infrastructure utilisation, operational cost and system resilience.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
