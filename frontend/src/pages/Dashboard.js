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
  Cube, MapPin, Brain, Path, CheckCircle, Cloud, PaperPlaneRight, FilePdf,
} from "@phosphor-icons/react";

const BACKEND_URL =
  process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8001";

// Ensure no trailing slash so `${API}/...` builds correctly
const API = `${BACKEND_URL.replace(/\/+$/, "")}/api`;

const LABELS = [
  "person",
  "car",
  "truck",
  "bicycle",
  "forklift",
  "helmet",
  "pallet",
  "box",
  "dog",
];

const pick = (arr) =>
  arr[Math.floor(Math.random() * arr.length)];

const rand = (min, max) =>
  Math.random() * (max - min) + min;

const nowMs = () =>
  new Date().toISOString().slice(11, 23);

const ROUTE_COLORS = {
  edge: "#00E676",
  cloud: "#0055FF",
  hybrid: "#FFCC00",
};

const Cell = ({
  children,
  className = "",
  testId,
}) => (
  <div
    data-testid={testId}
    className={`bg-[#0A0A0A] border border-[#27272A] ${className}`}
  >
    {children}
  </div>
);

const Overline = ({
  children,
  className = "",
}) => (
  <span className={`overline ${className}`}>
    {children}
  </span>
);

const KPI = ({
  label,
  value,
  unit,
  tone = "default",
  testId,
}) => {
  const toneMap = {
    default: "text-white",
    accent: "text-[#0055FF]",
    good: "text-[#00E676]",
    warn: "text-[#FFCC00]",
    bad: "text-[#FF3333]",
  };

  return (
    <Cell
      className="p-4 flex flex-col gap-2 min-h-[104px]"
      testId={testId}
    >
      <Overline>{label}</Overline>

      <div className="flex items-baseline gap-1">
        <span
          className={`font-mono text-3xl font-semibold ${toneMap[tone]}`}
        >
          {value}
        </span>

        {unit && (
          <span className="font-mono text-xs text-[#71717a]">
            {unit}
          </span>
        )}
      </div>
    </Cell>
  );
};

export default function Dashboard() {
  const [devices, setDevices] = useState([]);
  const [models, setModels] = useState([]);
  const [deployments, setDeployments] = useState([]);

  const [selectedDeviceId, setSelectedDeviceId] =
    useState(null);

  const [mode, setMode] = useState("edge");

  const [running, setRunning] =
    useState(true);

  const [series, setSeries] =
    useState([]);

  const [logs, setLogs] =
    useState([]);

  const [summary, setSummary] =
    useState({
      edge: null,
      cloud: null,
    });

  const [selectedModelId, setSelectedModelId] =
    useState(null);

  const [deploying, setDeploying] =
    useState(false);

  const [showArch, setShowArch] =
    useState(false);

  // ---------------------------------------------------------
  // Decision Engine
  // ---------------------------------------------------------

  const [engineName, setEngineName] =
    useState("ql");

  const [engineStatus, setEngineStatus] =
    useState(null);

  const [orchestrationSeries, setOrchestrationSeries] = useState({
    latency: [],
    throughput: [],
    qUpdates: [],
    success: [],
  });

  const [lastDecision, setLastDecision] =
    useState(null);

  const [training, setTraining] =
    useState(false);

  const [autoRoute, setAutoRoute] =
    useState(true);

  const autoRouteRef = useRef(true);

  const [realInferenceLoading, setRealInferenceLoading] =
    useState(false);

  const [lastRealInference, setLastRealInference] =
    useState(null);

  const [liveDecision, setLiveDecision] =
    useState(null);

  const [liveExecution, setLiveExecution] =
    useState(null);
const [liveLearning, setLiveLearning] =
  useState(null);
  const [failoverApplied, setFailoverApplied] =
    useState(false);

  const [failoverReason, setFailoverReason] =
    useState(null);

  const [
    liveInfrastructure,
    setLiveInfrastructure,
  ] = useState({
    edge: null,
    cloud: null,
  });

  const [decisionHistory, setDecisionHistory] =
    useState([]);

  const [distribution, setDistribution] =
    useState({
      edge: 0,
      cloud: 0,
      hybrid: 0,
    });

  // ---------------------------------------------------------
  // AI Workload State
  // ---------------------------------------------------------

  const [workloadType, setWorkloadType] =
    useState("yolo");

  const [imagePath, setImagePath] =
    useState("");

  // PC camera support for real inference demonstration
  const videoRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraUploading, setCameraUploading] = useState(false);
  const [cameraPreview, setCameraPreview] = useState("");

  const [confidence, setConfidence] =
    useState(0.25);

  // ---------------------------------------------------------
  // Research / Benchmark State
  // ---------------------------------------------------------

  const [qtable, setQtable] =
    useState([]);

  const [benchmark, setBenchmark] =
    useState(null);

  const [benchmarkRunning, setBenchmarkRunning] =
    useState(false);

  const [scenarios, setScenarios] =
    useState([]);

  const [
    selectedScenario,
    setSelectedScenario,
  ] = useState("mixed");

  const [ciResult, setCiResult] =
    useState(null);

  const [ciRunning, setCiRunning] =
    useState(false);

  const [
    reportGenerating,
    setReportGenerating,
  ] = useState(false);

  // ---------------------------------------------------------
  // Experiment Assistant
  // ---------------------------------------------------------

  const [
    assistantMessages,
    setAssistantMessages,
  ] = useState([]);

  const [
    assistantInput,
    setAssistantInput,
  ] = useState("");

  const [
    assistantLoading,
    setAssistantLoading,
  ] = useState(false);

  const [
    assistantSessionId,
    setAssistantSessionId,
  ] = useState(null);

  const [
    completeRunning,
    setCompleteRunning,
  ] = useState(false);

  const [
    completeStatus,
    setCompleteStatus,
  ] = useState(null);

  const [
    masterSummary,
    setMasterSummary,
  ] = useState(null);

  const bufferRef =
    useRef([]);

  const logsRef =
    useRef(null);

  const selectedDevice =
    devices.find(
      (d) => d.id === selectedDeviceId
    ) || devices[0];

  const selectedModel =
    models.find(
      (m) =>
        m.id ===
        selectedDevice?.deployed_model_id
    );

  const pushLog = useCallback(
    (level, msg) => {
      setLogs((prev) =>
        [
          ...prev,
          {
            t: nowMs(),
            level,
            msg,
          },
        ].slice(-200)
      );
    },
    []
  );

  // ---------------------------------------------------------
  // Bootstrap
  // ---------------------------------------------------------

  useEffect(() => {
    (async () => {
      try {
        const [
          d,
          m,
          dep,
          st,
          sc,
        ] = await Promise.all([
          axios.get(
            `${API}/devices`
          ),
          axios.get(
            `${API}/models`
          ),
          axios.get(
            `${API}/deployments?limit=10`
          ),
          axios.get(
            `${API}/decisions/status`
          ),
          axios.get(
            `${API}/scenarios`
          ),
        ]);

        setDevices(d.data);
        setModels(m.data);
        setDeployments(dep.data);
        setEngineStatus(st.data);

        setScenarios(
          sc.data.scenarios || []
        );

        if (d.data.length) {
          setSelectedDeviceId(
            d.data[0].id
          );
        }

        pushLog(
          "info",
          `Platform resources loaded | ${d.data.length} devices | ${m.data.length} models`
        );

        pushLog(
          "ok",
          `Routing policy services ready`
        );
      } catch (e) {
        pushLog(
          "err",
          `Bootstrap failed: ${e.message}`
        );
      }
    })();
  }, [pushLog]);

  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop =
        logsRef.current.scrollHeight;
    }
  }, [logs]);

  // ---------------------------------------------------------
  // Decision History + Distribution
  // ---------------------------------------------------------

  useEffect(() => {
    const iv = setInterval(
      async () => {
        try {
          const [h, d, st] =
            await Promise.all([
              axios.get(
                `${API}/decisions/history?limit=15&engine=${engineFullName(
                  engineName
                )}`
              ),

              axios.get(
                `${API}/decisions/distribution?minutes=30&engine=${engineFullName(
                  engineName
                )}`
              ),

              axios.get(
                `${API}/decisions/status`
              ),
            ]);

          const historyRows = Array.isArray(h.data)
            ? h.data
            : Array.isArray(h.data?.value)
              ? h.data.value
              : [];

          setDecisionHistory(historyRows);
          setDistribution(d.data);
          setEngineStatus(st.data);

          const latencyValues = historyRows
            .slice(0, 12)
            .reverse()
            .map((item) => Number(item?.latency_us || 0))
            .filter((value) => Number.isFinite(value));

          let currentThroughput = 0;

          if (historyRows.length > 1) {
            const timestamps = historyRows
              .map((item) => Number(item?.created_at || 0))
              .filter((value) => value > 0);

            if (timestamps.length > 1) {
              const windowSeconds =
                Math.max(...timestamps) - Math.min(...timestamps);

              currentThroughput =
                windowSeconds > 0
                  ? (timestamps.length - 1) / windowSeconds
                  : 0;
            }
          }

          const qMetrics =
            st.data?.QLearning?.metrics || {};

          setOrchestrationSeries((previous) => ({
            latency:
              latencyValues.length > 0
                ? latencyValues
                : previous.latency,

            throughput: [
              ...previous.throughput,
              currentThroughput,
            ].slice(-12),

            qUpdates: [
              ...previous.qUpdates,
              Number(qMetrics.real_updates || 0),
            ].slice(-12),

            success: [
              ...previous.success,
              qMetrics.last_real_success === true
                ? 1
                : qMetrics.last_real_success === false
                  ? 0
                  : null,
            ]
              .filter((value) => value !== null)
              .slice(-12),
          }));
        } catch (e) {
          // Silent refresh failure.
        }
      },
      3000
    );

    return () =>
      clearInterval(iv);
  }, [engineName]);

  // ---------------------------------------------------------
  // Live Orchestration Tick
  // ---------------------------------------------------------

  useEffect(() => {
    if (
      !running ||
      !selectedDevice
    ) {
      return;
    }

    const iv = setInterval(
      async () => {
        try {
          const telemetryRes =
            await axios.get(
              `${API}/telemetry/live`
            );

          const telemetry =
            telemetryRes.data;

          const connected =
            telemetry.connected;

          const cpuAvail =
            telemetry.cpu_available;

          const memAvail =
            telemetry.memory_available;

          const netLatency =
            telemetry.network_latency_ms ??
            9999;

          const batchSize = 1;
          const priority = 3;
          const costBudget = 1.0;

          const modelSizeMb =
            selectedModel?.size_mb ||
            6.2;
                    const ctx = {
            network_latency_ms: Number(
              Number(netLatency).toFixed(1)
            ),
            connectivity:
              connected ? 1 : 0,
            cpu_available: Number(
              Number(cpuAvail ?? 0).toFixed(1)
            ),
            memory_available: Number(
              Number(memAvail ?? 0).toFixed(1)
            ),
            batch_size: batchSize,
            priority,
            cost_budget_usd: Number(
              costBudget.toFixed(3)
            ),
            model_size_mb: Number(
              Number(modelSizeMb).toFixed(1)
            ),
          };

          const res =
            await axios.post(
              `${API}/decisions/predict-live`,
              null,
              {
                params: {
                  engine: engineName,
                  batch_size:
                    batchSize,
                  priority,
                  cost_budget_usd:
                    costBudget,
                  model_size_mb:
                    modelSizeMb,
                },
                timeout: 90000,
              }
            );

          const data =
            res.data || {};

          const decision =
            data.decision || {};
const execution =
  data.execution || null;

const learning =
  data.learning || null;

const infrastructure =
  data.infrastructure || {
    edge: null,
    cloud: null,
  };
         setLiveInfrastructure(
  infrastructure
);

setLiveDecision(
  decision
);

setLiveExecution(
  execution
);

setLiveLearning(
  learning
);

setLastDecision({
  ...decision,
  context: ctx,
});

const didFailover =
  decision.failover_applied ??
  false;

const reason =
  decision.failover_reason ??
  null;

setFailoverApplied(
  didFailover
);

setFailoverReason(
  reason
);
          const selectedRoute =
            decision.route ||
            mode;

          if (
            autoRouteRef.current &&
            selectedRoute &&
            selectedRoute !== mode
          ) {
            setMode(
              selectedRoute
            );
          }

          pushLog(
            didFailover
              ? "warn"
              : "info",
            `[${
              decision.engine ||
              engineFullName(
                engineName
              )
            }] policy=${
              decision.policy_route ||
              decision.route ||
              "--"
            } | execution=${String(
              selectedRoute
            ).toUpperCase()} | failover=${
              didFailover
                ? "YES"
                : "NO"
            }`
          );
        } catch (e) {
          pushLog(
            "err",
            `Live orchestration error: ${
              e.response?.data
                ?.detail ||
              e.message
            }`
          );
        }
      },
      3000
    );

    return () =>
      clearInterval(iv);
  }, [
    running,
    mode,
    selectedDevice,
    selectedModel,
    engineName,
    autoRoute,
    pushLog,
  ]);

  // ---------------------------------------------------------
  // Persist inference records
  // ---------------------------------------------------------

  useEffect(() => {
    const flush =
      setInterval(
        async () => {
          if (
            !bufferRef.current
              .length
          ) {
            return;
          }

          const batch =
            bufferRef.current.splice(
              0,
              bufferRef.current
                .length
            );

          try {
            await axios.post(
              `${API}/inferences`,
              {
                records: batch,
              }
            );
          } catch (e) {
            // Keep dashboard
            // responsive even if
            // persistence fails.
          }
        },
        5000
      );

    return () =>
      clearInterval(flush);
  }, []);

  // ---------------------------------------------------------
  // Metrics Summary
  // ---------------------------------------------------------

  useEffect(() => {
    const iv =
      setInterval(
        async () => {
          try {
            const res =
              await axios.get(
                `${API}/metrics/summary`
              );

            setSummary(
              res.data
            );
          } catch (e) {
            // Silent refresh
            // failure.
          }
        },
        4000
      );

    return () =>
      clearInterval(iv);
  }, []);

  // ---------------------------------------------------------
  // Toggle Edge Network
  // ---------------------------------------------------------

  const toggleNetwork =
    async () => {
      if (
        !selectedDevice
      ) {
        return;
      }

      const next =
        !selectedDevice.connected;

      try {
        const res =
          await axios.post(
            `${API}/devices/${selectedDevice.id}/network`,
            {
              connected:
                next,
            }
          );

        setDevices(
          (prev) =>
            prev.map(
              (d) =>
                d.id ===
                res.data.id
                  ? res.data
                  : d
            )
        );

        pushLog(
          next
            ? "ok"
            : "err",
          next
            ? `[${
                selectedDevice.name
              }] network RESTORED`
            : `[${
                selectedDevice.name
              }] network DISCONNECTED`
        );

        if (next) {
          toast.success(
            "Network restored"
          );
        } else {
          toast.warning(
            "Edge network disconnected"
          );
        }
      } catch (e) {
        pushLog(
          "err",
          `Network toggle failed: ${
            e.response?.data
              ?.detail ||
            e.message
          }`
        );

        toast.error(
          "Network change failed"
        );
      }
    };

  // ---------------------------------------------------------
  // Deploy AI Model
  // ---------------------------------------------------------

  const deployModel =
    async () => {
      if (
        !selectedDevice ||
        !selectedModelId
      ) {
        toast(
          "Select a model first"
        );
        return;
      }

      const model =
        models.find(
          (m) =>
            m.id ===
            selectedModelId
        );

      if (!model) {
        toast.error(
          "Model not found"
        );
        return;
      }

      setDeploying(true);

      pushLog(
        "info",
        `Deploying ${model.name}@${model.version} -> ${selectedDevice.name}`
      );

      try {
        const res =
          await axios.post(
            `${API}/deployments`,
            {
              device_id:
                selectedDevice.id,
              model_id:
                selectedModelId,
            }
          );

        setDeployments(
          (prev) => [
            res.data,
            ...prev,
          ].slice(0, 10)
        );

        setDevices(
          (prev) =>
            prev.map(
              (d) =>
                d.id ===
                selectedDevice.id
                  ? {
                      ...d,
                      deployed_model_id:
                        model.id,
                      deployed_model_name:
                        `${model.name}@${model.version}`,
                    }
                  : d
            )
        );

        pushLog(
          "ok",
          `[deployment] ${model.name}@${model.version} ACTIVE on ${selectedDevice.name}`
        );

        toast.success(
          "Deployment active"
        );
      } catch (e) {
        const detail =
          e.response?.data
            ?.detail ||
          e.message;

        pushLog(
          "err",
          `Deployment failed: ${detail}`
        );

        toast.error(
          "Deployment failed"
        );
      } finally {
        setDeploying(
          false
        );
      }
    };

// ---------------------------------------------------------
// Retrain Decision Policies
// ---------------------------------------------------------

const retrainEngines =
  async () => {
    setTraining(true);

    pushLog(
      "info",
      "Retraining Decision Tree + Random Forest..."
    );

    try {
      const supervisedRes =
        await axios.post(
          `${API}/decisions/train`,
          {
            n_samples:
              4000,
          },
          {
            timeout:
              120000,
          }
        );

      const statusRes =
        await axios.get(
          `${API}/decisions/status`
        );

      setEngineStatus(
        statusRes.data
      );

      const dtAcc =
        supervisedRes.data
          ?.trained
          ?.DecisionTree
          ?.accuracy;

      const rfAcc =
        supervisedRes.data
          ?.trained
          ?.RandomForest
          ?.accuracy;

      pushLog(
        "ok",
        `Supervised training complete | DT=${
          dtAcc ?? "--"
        } | RF=${
          rfAcc ?? "--"
        } | Q-Learning remains in real-execution learning mode`
      );

      toast.success(
        "Decision Tree and Random Forest retrained"
      );
    } catch (e) {
      const detail =
        e.response?.data
          ?.detail ||
        e.message;

      pushLog(
        "err",
        `Training failed: ${detail}`
      );

      toast.error(
        "Training failed"
      );
    } finally {
      setTraining(
        false
      );
    }
  };

  // ---------------------------------------------------------
  // Real AI Workload Execution
  // ---------------------------------------------------------
  // ---------------------------------------------------------
  // PC Camera - Real Inference Input
  // ---------------------------------------------------------

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      cameraStreamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      setCameraActive(true);
      pushLog("ok", "[camera] PC camera started");
      toast.success("Camera started");
    } catch (error) {
      const detail = error?.message || "Unable to access camera";
      pushLog("err", `[camera] ${detail}`);
      toast.error("Unable to access camera");
    }
  };

  const stopCamera = () => {
    if (cameraStreamRef.current) {
      cameraStreamRef.current
        .getTracks()
        .forEach((track) => track.stop());

      cameraStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
    pushLog("info", "[camera] PC camera stopped");
  };

  const powerOffCamera = () => {
    stopCamera();
    setCameraPreview("");
    setLastRealInference(null);
    pushLog("info", "[camera] PC camera powered off");
    toast.success("Camera powered off");
  };
  const captureCameraFrame = async () => {
    if (!videoRef.current || !cameraActive) {
      toast.error("Start the camera first");
      return;
    }

    if (!["yolo", "florence"].includes(workloadType)) {
      toast.error("Select YOLO or Florence-2");
      return;
    }

    const video = videoRef.current;

    if (!video.videoWidth || !video.videoHeight) {
      toast.error("Camera is not ready yet");
      return;
    }

    setCameraUploading(true);

    try {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const context = canvas.getContext("2d");
      context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
      );

      const preview = canvas.toDataURL(
        "image/jpeg",
        0.9
      );

      setCameraPreview(preview);

      const blob = await new Promise((resolve) =>
        canvas.toBlob(
          resolve,
          "image/jpeg",
          0.9
        )
      );

      if (!blob) {
        throw new Error(
          "Unable to capture camera frame"
        );
      }

      const formData = new FormData();
      formData.append(
        "file",
        blob,
        "camera-frame.jpg"
      );

      const params = new URLSearchParams({
        model: workloadType,
        confidence: String(
          Number(confidence)
        ),
        mode: autoRoute ? "auto" : mode,
        engine: engineName,
      });

      pushLog(
        "info",
        `[camera] Captured frame - running ${workloadType.toUpperCase()} analysis`
      );

      const response = await fetch(
        `${API}/inference/vision?${params.toString()}`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData =
          await response
            .json()
            .catch(() => ({}));

        throw new Error(
          errorData.detail ||
          `Vision inference failed with HTTP ${response.status}`
        );
      }

      const result =
        await response.json();

      const finalResult = {
        ...result,
        workload_type:
          workloadType,
      };

      setLastRealInference(
        finalResult
      );

      setImagePath("");

      pushLog(
        "ok",
        `[camera] ${workloadType.toUpperCase()} analysis completed`
      );

      toast.success(
        workloadType === "yolo"
          ? "YOLO analysis completed"
          : "Florence-2 analysis completed"
      );
    } catch (error) {
      const detail =
        error?.message ||
        "Vision analysis failed";

      pushLog(
        "err",
        `[camera] Analysis failed: ${detail}`
      );

      toast.error(detail);
    } finally {
      setCameraUploading(false);
    }
  };


  const runRealInference =
    async () => {
      const requiresImage =
        [
          "yolo",
          "florence",
        ].includes(
          workloadType
        );

      if (
        requiresImage &&
        !imagePath.trim()
      ) {
        toast.error(
          "Please enter an image path."
        );
        return;
      }

      setRealInferenceLoading(
        true
      );

      const modelSize =
        selectedModel?.size_mb ||
        6.2;

      const effectiveMode =
        mode;

      let liveTelemetry =
        null;

      try {
        try {
          const telemetryRes =
            await axios.get(
              `${API}/telemetry/live`,
              {
                timeout:
                  5000,
              }
            );

          liveTelemetry =
            telemetryRes.data;
        } catch (telemetryError) {
          pushLog(
            "info",
            "[ai-workload] Live telemetry unavailable - continuing with fallback values"
          );
        }

        const netLatency =
          liveTelemetry
            ?.network_latency_ms ??
          liveInfrastructure
            ?.cloud?.rtt_ms ??
          9999;

        const workloadLabel =
          workloadType ===
          "yolo"
            ? "YOLO"
            : workloadType ===
                "florence"
              ? "Florence-2"
              : "Compute Benchmark";

        pushLog(
          "info",
          `[ai-workload] Starting ${workloadLabel} | route=${String(
            effectiveMode
          ).toUpperCase()}`
        );

        let res;

        if (
          workloadType ===
          "yolo"
        ) {
          const yoloResponse = await fetch(
            `${API}/inference/yolo`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                mode: effectiveMode,
                image_path: imagePath.trim(),
                confidence: Number(confidence),
                model_name: "yolov8n.pt",
              }),
            }
          );

          if (!yoloResponse.ok) {
            const errorData = await yoloResponse.json().catch(() => ({}));
            throw new Error(
              errorData.detail ||
              `YOLO request failed with HTTP ${yoloResponse.status}`
            );
          }

          res = {
            data: await yoloResponse.json(),
          };} else if (
          workloadType ===
          "florence"
        ) {
          res =
            await axios.post(
              `${API}/inference/florence`,
              {
                image_path:
                  imagePath.trim(),
                task:
                  "<MORE_DETAILED_CAPTION>",
              },
              {
                timeout:
                  180000,
              }
            );
        } else {
          res =
            await axios.post(
              `${API}/inference/run`,
              {
                mode:
                  effectiveMode,
                model_size_mb:
                  modelSize,
                network_latency_ms:
                  netLatency,
                connectivity:
                  liveTelemetry
                    ?.connected
                    ? 1
                    : 0,
                batch_size:
                  1,
              },
              {
                timeout:
                  120000,
              }
            );
        }

        const result = {
          ...res.data,
          workload_type:
            workloadType,
        };

        setLastRealInference(
          result
        );

        if (
          result.latency_ms !=
            null &&
          Number.isFinite(
            Number(
              result.latency_ms
            )
          )
        ) {
          const inferenceMode =
            result.mode ||
            effectiveMode;

          const latency =
            Number(
              result.latency_ms
            );

          const cpu =
            Number(
              result.cpu_percent ??
                result.cpu ??
                0
            );

          const memory =
            Number(
              result.memory_mb ??
                0
            );

          const fps =
            Number(
              result.fps_estimate ??
                0
            );

          const objects =
            result.detections_count ??
            result.objects_detected ??
            result.detections
              ?.length ??
            0;

          const topLabel =
            result.detections?.[0]
              ?.class_name ??
            result.detections?.[0]
              ?.label ??
            "none";

          const topConf =
            Number(
              result.detections?.[0]
                ?.confidence ??
                0
            );

          bufferRef.current.push(
            {
              device_id:
                selectedDevice?.id ||
                "edge-node-01",

              device_name:
                selectedDevice
                  ?.name ||
                "edge-node-01",

              mode:
                inferenceMode,

              latency_ms:
                Math.round(
                  latency * 100
                ) / 100,

              cpu:
                Math.round(
                  cpu * 10
                ) / 10,

              memory:
                Math.round(
                  memory * 10
                ) / 10,

              fps:
                Math.round(
                  fps * 10
                ) / 10,

              cost_usd:
  Number(
    result.cost_usd ??
    0
  ),

              objects_detected:
                objects,

              top_label:
                topLabel,

              top_confidence:
                Math.round(
                  topConf * 100
                ) / 100,

              connected:
                liveTelemetry
                  ?.connected ??
                true,

              timestamp:
                new Date()
                  .toISOString(),
            }
          );

          setSeries(
            (prev) => [
              ...prev,
              {
                t: nowMs().slice(
                  0,
                  8
                ),

                edge_latency:
                  inferenceMode ===
                    "edge" ||
                  inferenceMode ===
                    "hybrid"
                    ? Math.round(
                        latency
                      )
                    : null,

                cloud_latency:
                  inferenceMode ===
                  "cloud"
                    ? Math.round(
                        latency
                      )
                    : result.cloud_latency_ms !=
                        null
                      ? Math.round(
                          Number(
                            result.cloud_latency_ms
                          )
                        )
                      : null,
              },
            ].slice(-40)
          );
        }

        pushLog(
          result.success ===
            false
            ? "err"
            : "ok",
          `[ai-workload] ${workloadLabel} completed | latency=${
            result.latency_ms ??
            "--"
          }ms`
        );

        if (
          result.success ===
          false
        ) {
          toast.error(
            `${workloadLabel} execution failed`
          );
        } else {
          toast.success(
            `${workloadLabel} workload completed`
          );
        }
      } catch (e) {
        const detail =
          e.response?.data
            ?.detail;

        const errorMessage =
          typeof detail ===
          "string"
            ? detail
            : e.response?.data
                ?.message ||
              e.message ||
              "AI workload failed";

        pushLog(
          "err",
          `[ai-workload] failed: ${errorMessage}`
        );

        toast.error(
          errorMessage
        );
      } finally {
        setRealInferenceLoading(
          false
        );
      }
    };  
      // ---------------------------------------------------------
  // Comparative Benchmark
  // ---------------------------------------------------------

  const runBenchmark =
    async (
      nScenarios = 500
    ) => {
      setBenchmarkRunning(
        true
      );

      pushLog(
        "info",
        `[benchmark] starting | scenario=${selectedScenario} | ${nScenarios} samples`
      );

      try {
        const res =
          await axios.post(
            `${API}/benchmark/run`,
            {
              n_scenarios:
                nScenarios,
              seed: 2026,
              save_artefacts:
                true,
              scenario_id:
                selectedScenario,
            },
            {
              timeout:
                180000,
            }
          );

        setBenchmark(
          res.data
        );

        pushLog(
          "ok",
          `[benchmark] completed | scenario=${selectedScenario} | ${
            res.data?.meta
              ?.n_scenarios ??
            nScenarios
          } samples`
        );

        toast.success(
          "Benchmark completed"
        );
      } catch (e) {
        const detail =
          e.response?.data
            ?.detail ||
          e.message ||
          "Benchmark failed";

        pushLog(
          "err",
          `[benchmark] failed: ${
            typeof detail ===
            "string"
              ? detail
              : "unknown error"
          }`
        );

        toast.error(
          typeof detail ===
          "string"
            ? detail
            : "Benchmark failed"
        );
      } finally {
        setBenchmarkRunning(
          false
        );
      }
    };

  // ---------------------------------------------------------
  // Multi-seed CI
  // ---------------------------------------------------------

  const runCI =
    async () => {
      setCiRunning(
        true
      );

      pushLog(
        "info",
        `[ci] starting multi-seed validation | scenario=${selectedScenario}`
      );

      try {
        const res =
          await axios.post(
            `${API}/benchmark/ci`,
            {
              scenario_id:
                selectedScenario,
              seeds: [
                2026,
                42,
                7,
                1234,
                9999,
              ],
              n_scenarios:
                200,
            },
            {
              timeout:
                240000,
            }
          );

        setCiResult(
          res.data
        );

        pushLog(
          "ok",
          `[ci] completed | scenario=${selectedScenario} | 5 seeds`
        );

        toast.success(
          "95% CI validation completed"
        );
      } catch (e) {
        const detail =
          e.response?.data
            ?.detail ||
          e.message ||
          "CI validation failed";

        pushLog(
          "err",
          `[ci] failed: ${
            typeof detail ===
            "string"
              ? detail
              : "unknown error"
          }`
        );

        toast.error(
          typeof detail ===
          "string"
            ? detail
            : "CI validation failed"
        );
      } finally {
        setCiRunning(
          false
        );
      }
    };

  // ---------------------------------------------------------
  // Thesis Report
  // ---------------------------------------------------------

  const generateReport =
    async () => {
      setReportGenerating(
        true
      );

      pushLog(
        "info",
        `[report] generating thesis report | scenario=${selectedScenario}`
      );

      try {
        const res =
          await axios.post(
            `${API}/benchmark/report`,
            {
              scenario_id:
                selectedScenario,
              seeds: [
                2026,
                42,
                7,
                1234,
                9999,
              ],
              n_scenarios:
                200,
            },
            {
              timeout:
                240000,
              responseType:
                "blob",
            }
          );

        const blob =
          new Blob(
            [res.data],
            {
              type:
                "application/pdf",
            }
          );

        const url =
          window.URL
            .createObjectURL(
              blob
            );

        const link =
          document
            .createElement(
              "a"
            );

        link.href =
          url;

        link.download =
          `thesis_report_${selectedScenario}.pdf`;

        document.body
          .appendChild(
            link
          );

        link.click();

        link.remove();

        window.URL
          .revokeObjectURL(
            url
          );

        pushLog(
          "ok",
          `[report] thesis PDF generated | scenario=${selectedScenario}`
        );

        toast.success(
          "Thesis report generated"
        );
      } catch (e) {
        const detail =
          e.response?.data
            ?.detail ||
          e.message ||
          "Report generation failed";

        pushLog(
          "err",
          `[report] failed: ${
            typeof detail ===
            "string"
              ? detail
              : "unknown error"
          }`
        );

        toast.error(
          "Report generation failed"
        );
      } finally {
        setReportGenerating(
          false
        );
      }
    };

  // ---------------------------------------------------------
  // Complete Benchmark
  // ---------------------------------------------------------

  const runCompleteBenchmark =
    async () => {
      setCompleteRunning(
        true
      );

      setCompleteStatus({
        status:
          "running",
        current_scenario:
          null,
        completed: [],
        total: 8,
        elapsed_s: 0,
      });

      pushLog(
        "info",
        "[complete-benchmark] starting all scenarios | 5 seeds"
      );

      try {
        const startedAt =
          Date.now();

        const res =
          await axios.post(
            `${API}/benchmark/complete`,
            {
              seeds: [
                2026,
                42,
                7,
                1234,
                9999,
              ],
              n_scenarios:
                100,
              include:
                null,
            },
            {
              timeout:
                600000,
            }
          );

        const elapsedS =
          Math.round(
            (
              Date.now() -
              startedAt
            ) / 1000
          );

        const data =
          res.data || {};

        setMasterSummary(
          data.master_summary ||
          data.summary ||
          data
        );

        setCompleteStatus({
          status:
            "done",
          current_scenario:
            null,
          completed:
            data.completed ||
            [],
          total:
            data.total ||
            8,
          elapsed_s:
            data.elapsed_s ||
            elapsedS,
        });

        pushLog(
          "ok",
          "[complete-benchmark] all scenarios completed"
        );

        toast.success(
          "Complete benchmark finished"
        );
      } catch (e) {
        const detail =
          e.response?.data
            ?.detail ||
          e.message ||
          "Complete benchmark failed";

        setCompleteStatus({
          status:
            "error",
          message:
            typeof detail ===
            "string"
              ? detail
              : "Complete benchmark failed",
        });

        pushLog(
          "err",
          `[complete-benchmark] failed: ${
            typeof detail ===
            "string"
              ? detail
              : "unknown error"
          }`
        );

        toast.error(
          "Complete benchmark failed"
        );
      } finally {
        setCompleteRunning(
          false
        );
      }
    };

  // ---------------------------------------------------------
  // Q-Learning Q-table
  // ---------------------------------------------------------

  const refreshQTable =
    async () => {
      try {
        const res =
          await axios.get(
            `${API}/decisions/qlearning/qtable`
          );

        const rows =
          Array.isArray(
            res.data
          )
            ? res.data
            : res.data?.rows || res.data?.states ||
              res.data
                ?.qtable ||
              [];

        setQtable(
          rows
        );

        pushLog(
          "ok",
          "[q-learning] Q-table refreshed"
        );
      } catch (e) {
        pushLog(
          "err",
          `[q-learning] Q-table failed: ${
            e.response?.data
              ?.detail ||
            e.message
          }`
        );
      }
    };

  // ---------------------------------------------------------
  // Experiment Assistant
  // ---------------------------------------------------------

  const askAssistant =
    async (
      promptOverride = null
    ) => {
      const question =
        (
          promptOverride ??
          assistantInput
        ).trim();

      if (!question) {
        return;
      }

      const userMessage = {
        q: question,
        a: null,
        ts: nowMs(),
      };

      setAssistantMessages(
        (prev) => [
          ...prev,
          userMessage,
        ]
      );

      setAssistantInput(
        ""
      );

      setAssistantLoading(
        true
      );

      try {
        const res =
          await axios.post(
            `${API}/assistant/ask`,
            {
              question,
              session_id:
                assistantSessionId,
            },
            {
              timeout:
                180000,
            }
          );

        const answer =
          res.data?.answer ||
          res.data?.response ||
          res.data?.text ||
          "No answer returned.";

        if (
          res.data?.session_id
        ) {
          setAssistantSessionId(
            res.data.session_id
          );
        }

        setAssistantMessages(
          (prev) => {
            const updated =
              [...prev];

            for (
              let i =
                updated.length -
                1;
              i >= 0;
              i--
            ) {
              if (
                updated[i].q ===
                  question &&
                updated[i].a ===
                  null
              ) {
                updated[i] = {
                  ...updated[i],
                  a: answer,
                };

                break;
              }
            }

            return updated;
          }
        );

        pushLog(
          "ok",
          "[assistant] answered experiment question"
        );
      } catch (e) {
        const detail =
          e.response?.data
            ?.detail ||
          e.message ||
          "Assistant request failed";

        setAssistantMessages(
          (prev) => {
            const updated =
              [...prev];

            for (
              let i =
                updated.length -
                1;
              i >= 0;
              i--
            ) {
              if (
                updated[i].q ===
                  question &&
                updated[i].a ===
                  null
              ) {
                updated[i] = {
                  ...updated[i],
                  a: `Error: ${
                    typeof detail ===
                    "string"
                      ? detail
                      : "Assistant request failed"
                  }`,
                };

                break;
              }
            }

            return updated;
          }
        );

        pushLog(
          "err",
          `[assistant] failed: ${
            typeof detail ===
            "string"
              ? detail
              : "unknown error"
          }`
        );
      } finally {
        setAssistantLoading(
          false
        );
      }
    };

  // ---------------------------------------------------------
  // Derived Data
  // ---------------------------------------------------------

  const engineFullName =
    (name) => {
      const names = {
        rule:
          "RuleBased",
        dt:
          "DecisionTree",
        rf:
          "RandomForest",
        ql:
          "QLearning",
      };

      return (
        names[name] ||
        "QLearning"
      );
    };

  const engineLabel =
    (name) => {
      const names = {
        rule:
          "RULE-BASED",
        dt:
          "DECISION TREE",
        rf:
          "RANDOM FOREST",
        ql:
          "Q-LEARNING",
      };

      return (
        names[name] ||
        "Q-LEARNING"
      );
    };

  const routeTone =
    (route) => {
      if (
        route === "edge"
      ) {
        return "text-[#00E676]";
      }

      if (
        route === "cloud"
      ) {
        return "text-[#0055FF]";
      }

      if (
        route === "hybrid"
      ) {
        return "text-[#FFCC00]";
      }

      return "text-white";
    };
  const infrastructureProfile = (state) => {
    const s = Number(state);

    if (!Number.isInteger(s) || s < 0 || s >= 108) {
      return "Unknown Condition";
    }

    const connectivity = s % 2;
    let remaining = Math.floor(s / 2);

    const priority = remaining % 3;
    remaining = Math.floor(remaining / 3);

    const cost = remaining % 2;
    remaining = Math.floor(remaining / 2);

    const cpu = remaining % 3;
    remaining = Math.floor(remaining / 3);

    const network = remaining % 3;

    const networkLabel =
      connectivity === 0
        ? "Cloud Unavailable"
        : network === 0
        ? "Good Network"
        : network === 1
        ? "Moderate Network"
        : "High Latency";

    const cpuLabel =
      cpu === 0
        ? "Low CPU Available"
        : cpu === 1
        ? "Medium CPU Available"
        : "High CPU Available";

    return `${networkLabel} / ${cpuLabel}`;
  };

  const formatMetric =
    (
      value,
      digits = 1,
      fallback = "--"
    ) => {
      const number =
        Number(value);

      if (
        !Number.isFinite(
          number
        )
      ) {
        return fallback;
      }

      return number
        .toFixed(
          digits
        );
    };

  const edgeInfra =
    liveInfrastructure
      ?.edge ||
    null;

  const cloudInfra =
    liveInfrastructure
      ?.cloud ||
    null;

  const edgeConnected =
    edgeInfra
      ?.connected ??
    selectedDevice
      ?.connected ??
    false;

  const cloudConnected =
    cloudInfra
      ?.connected ??
    false;

  const edgeCpu =
    edgeInfra
      ?.cpu_percent ??
    (
      edgeInfra
        ?.cpu_available !=
      null
        ? 100 -
          Number(
            edgeInfra
              .cpu_available
          )
        : null
    );

  const cloudCpu =
    cloudInfra
      ?.cpu_percent ??
    null;

  const edgeMemory =
    edgeInfra
      ?.memory_percent ??
    (
      edgeInfra
        ?.memory_available !=
      null
        ? 100 -
          Number(
            edgeInfra
              .memory_available
          )
        : null
    );

  const cloudMemory =
    cloudInfra
      ?.memory_percent ??
    null;

  const edgeDisk =
    edgeInfra
      ?.disk_percent ??
    null;

  const cloudDisk =
    cloudInfra
      ?.disk_percent ??
    null;

  const edgeRtt =
    edgeInfra
      ?.network_latency_ms ??
    null;

  const cloudRtt =
    cloudInfra
      ?.rtt_ms ??
    null;

  const currentPolicyRoute =
    liveDecision
      ?.policy_route ||
    liveDecision
      ?.route ||
    lastDecision
      ?.policy_route ||
    lastDecision
      ?.route ||
    "--";

  const currentExecutedRoute =
    liveDecision
      ?.route ||
    lastDecision
      ?.route ||
    mode ||
    "--";

  const executionSuccess =
    liveExecution
      ?.success;

  const executionStatus =
    executionSuccess ===
    true
      ? "SUCCESS"
      : executionSuccess ===
          false
        ? "FAILED"
        : "WAITING";

  const totalDistribution =
    Number(
      distribution
        ?.edge ||
      0
    ) +
    Number(
      distribution
        ?.cloud ||
      0
    ) +
    Number(
      distribution
        ?.hybrid ||
      0
    );

  const distributionData = [
    {
      name:
        "edge",
      value:
        Number(
          distribution
            ?.edge ||
          0
        ),
    },
    {
      name:
        "cloud",
      value:
        Number(
          distribution
            ?.cloud ||
          0
        ),
    },
    {
      name:
        "hybrid",
      value:
        Number(
          distribution
            ?.hybrid ||
          0
        ),
    },
  ];

  const utilisationData = [
    {
      name:
        "Edge CPU",
      value:
        Number(
          edgeCpu ||
          0
        ),
    },
    {
      name:
        "Cloud CPU",
      value:
        Number(
          cloudCpu ||
          0
        ),
    },
    {
      name:
        "Edge RAM",
      value:
        Number(
          edgeMemory ||
          0
        ),
    },
    {
      name:
        "Cloud RAM",
      value:
        Number(
          cloudMemory ||
          0
        ),
    },
  ];

  const latestExecutionLatency =
    liveExecution
      ?.latency_ms !=
    null
      ? Number(
          liveExecution
            .latency_ms
        )
      : null;

  const latestCloudExecutionLatency =
    liveExecution
      ?.cloud_latency_ms !=
    null
      ? Number(
          liveExecution
            .cloud_latency_ms
        )
      : null;

  const dashboardLatencyData =
    useMemo(() => {
      const existing =
        Array.isArray(
          series
        )
          ? series.slice(
              -20
            )
          : [];

      if (
        !existing.length &&
        latestExecutionLatency !=
          null
      ) {
        return [
          {
            t: nowMs()
              .slice(
                0,
                8
              ),

            edge_latency:
              currentExecutedRoute ===
                "edge" ||
              currentExecutedRoute ===
                "hybrid"
                ? latestExecutionLatency
                : null,

            cloud_latency:
              currentExecutedRoute ===
              "cloud"
                ? latestExecutionLatency
                : latestCloudExecutionLatency,
          },
        ];
      }

      return existing;
    }, [
      series,
      latestExecutionLatency,
      latestCloudExecutionLatency,
      currentExecutedRoute,
    ]);

  const featureImportanceData =
    useMemo(() => {
      const importance =
        engineStatus?.[
          engineFullName(
            engineName
          )
        ]?.metrics
          ?.feature_importance;

      if (
        !importance ||
        typeof importance !==
          "object"
      ) {
        return [];
      }

      return Object.entries(
        importance
      )
        .map(
          (
            [
              feature,
              value,
            ]
          ) => ({
            feature,
            value:
              Number(
                value
              ),
          })
        )
        .sort(
          (a, b) =>
            b.value -
            a.value
        );
    }, [
      engineStatus,
      engineName,
    ]);

  const policyMetrics =
    engineStatus?.[
      engineFullName(
        engineName
      )
    ]?.metrics ||
    null;

  const qLearningMetrics =
    engineStatus
      ?.QLearning
      ?.metrics ||
    null;

  const activePolicyAvailable =
    engineStatus?.[
      engineFullName(
        engineName
      )
    ]?.available !==
    false;

  const activePolicyTrained =
    engineStatus?.[
      engineFullName(
        engineName
      )
    ]?.trained !==
    false;

  const activeScenario =
    scenarios.find(
      (scenario) =>
        scenario.id ===
          selectedScenario ||
        scenario.scenario_id ===
          selectedScenario
    ) ||
    null;

  const sparkHeight = (value, series, minHeight = 3, maxHeight = 18) => {
    const values = Array.isArray(series)
      ? series.map(Number).filter(Number.isFinite)
      : [];

    if (!values.length) return minHeight;

    const min = Math.min(...values);
    const max = Math.max(...values);

    if (max === min) {
      return Math.round((minHeight + maxHeight) / 2);
    }

    const ratio = (Number(value) - min) / (max - min);

    return Math.max(
      minHeight,
      Math.min(
        maxHeight,
        Math.round(minHeight + ratio * (maxHeight - minHeight))
      )
    );
  };

  const xaiDecision =
    lastDecision ||
    liveDecision ||
    (
      Array.isArray(decisionHistory) && decisionHistory.length > 0
        ? decisionHistory[0]
        : null
    );

  const xaiContext =
    xaiDecision?.context || {};

  const xaiNetworkLatency =
    Number(xaiContext.network_latency_ms || 0);

  const xaiEdgeCpu =
    Math.max(
      0,
      100 - Number(xaiContext.cpu_available || 100)
    );

  const xaiMemoryAvailable =
    Number(xaiContext.memory_available || 0);

  const xaiConnectivity =
    Number(xaiContext.connectivity || 0);

  const xaiPriority =
    Number(xaiContext.priority || 0);

  const xaiCostBudget =
    Number(xaiContext.cost_budget_usd || 0);

  const xaiConfidence =
    Number(xaiDecision?.confidence || 0) * 100;

  const xaiRoute =
    String(
      currentPolicyRoute ||
      xaiDecision?.route ||
      "edge"
    ).toUpperCase();

  const xaiWhyText =
    xaiRoute === "HYBRID"
      ? "Q-Learning selected HYBRID because the current network and resource conditions favoured distributing execution across Edge and Cloud."
      : xaiRoute === "CLOUD"
        ? "Q-Learning selected CLOUD because the current context favoured remote execution under the observed network, resource and policy conditions."
        : "Q-Learning selected EDGE because the current context favoured local execution with lower dependency on remote Cloud resources.";

  const recentDecisionRows =
    Array.isArray(
      decisionHistory
    )
      ? decisionHistory.slice(
          0,
          8
        )
      : [];
        // ---------------------------------------------------------
  // Policy Metadata
  // ---------------------------------------------------------

  const engineOptions = [
    {
      value: "rule",
      label: "Rule-Based",
    },
    {
      value: "ql",
      label: "Q-Learning",
    },
  ];

  const policyDescriptions = {
    rule:
      "Deterministic baseline using explicit infrastructure and workload thresholds.",

    dt:
      "Supervised Decision Tree policy trained to select Edge, Cloud or Hybrid execution.",

    rf:
      "Random Forest ensemble policy combining multiple decision trees for robust routing.",

    ql:
      "Reinforcement-learning policy that selects routing actions according to learned Q-values.",
  };

  const policyObjectives = {
    rule:
      "Provides the interpretable baseline used for comparative evaluation.",

    dt:
      "Learns transparent routing boundaries from infrastructure and workload features.",

    rf:
      "Improves routing robustness through ensemble learning and feature aggregation.",

    ql:
      "Optimises routing behaviour through reward-driven interaction with the environment.",
  };

  const selectedEngineDescription =
    policyDescriptions[
      engineName
    ];

  const selectedEngineObjective =
    policyObjectives[
      engineName
    ];

  const policyAccuracy =
    policyMetrics
      ?.accuracy !=
    null
      ? `${(
          Number(
            policyMetrics
              .accuracy
          ) * 100
        ).toFixed(2)}%`
      : "--";

  const qReward =
    qLearningMetrics
      ?.mean_reward_last_500 ??
    qLearningMetrics
      ?.mean_reward ??
    "--";

  // ---------------------------------------------------------
  // Navigation
  // ---------------------------------------------------------

  const scrollToSection =
    (id) => {
      document
        .getElementById(id)
        ?.scrollIntoView({
          behavior:
            "smooth",
          block:
            "start",
        });
    };

  const navItems = [
    {
      label: "Dashboard",
      icon: ChartLineUp,
      target: "dashboard-section",
    },
    {
      label: "Orchestration Engine",
      icon: Lightning,
      target: "orchestration-engine-section",
    },
    {
      label: "Nodes",
      icon: HardDrives,
      target: "nodes-section",
    },
    {
      label: "Decision Engine",
      icon: Brain,
      target: "decision-section",
    },
    {
      label: "AI Workload",
      icon: Database,
      target: "settings-section",
    },
    {
      label: "Routing Results",
      icon: Path,
      target: "routing-section",
    },
    {
      label: "Telemetry",
      icon: Broadcast,
      target: "telemetry-section",
    },
    {
      label: "Scenarios",
      icon: TreeStructure,
      target: "scenarios-section",
    },
    {
      label: "AI-Assisted",
      icon: Lightning,
      target: "ai-assisted-section",
    },
    {
      label: "Evaluation",
      icon: TreeStructure,
      target: "evidence-section",
    },
    {
      label: "Experimental Evidence (PDF)",
      icon: FilePdf,
      target: "experimental-evidence-section",
    },
    {
      label: "Logs",
      icon: Terminal,
      target: "logs-section",
    },
  ];

  // =========================================================
  // RENDER
  // =========================================================

  return (
    <div
      className="min-h-screen bg-black text-white"
      data-testid="edge-cloud-dashboard"
    >
      <div className="flex min-h-screen">

        {/* ===================================================
            SIDEBAR
        ==================================================== */}

                <aside className="flex fixed inset-y-0 left-0 z-30 w-[270px] bg-[#050A12] border-r border-[#172033] flex-col">

          {/* Brand */}
          <div className="h-[74px] px-4 flex items-center border-b border-[#172033]">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 border border-[#2563EB] bg-[#081326] flex items-center justify-center">
                <Brain
                  size={20}
                  className="text-[#4F7CFF]"
                  weight="duotone"
                />
              </div>

              <div>
                <div className="font-mono text-[10px] font-semibold tracking-[0.025em] text-white uppercase whitespace-nowrap">
                  EDGE-CLOUD // ORCHESTRATOR
                </div>

                <div className="text-[8px] text-[#8B95A7] mt-0.5">
                  AI-Driven Infrastructure Orchestration
                </div>
              </div>
            </div>
          </div>

          {/* Admin */}
          <div className="px-4 py-4 border-b border-[#172033]">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-sm bg-[#5865F2] flex items-center justify-center font-mono text-[10px] font-semibold text-white">
                SA
              </div>

              <div>
                <div className="font-mono text-[10px] text-white">
                  Sophia
                </div>

                <div className="font-mono text-[8px] text-[#8B95A7] mt-0.5">
                  Admin
                </div>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-3 py-4 overflow-y-auto thin-scroll">
            <div className="font-mono text-[8px] uppercase tracking-[0.14em] text-[#596579] px-2 mb-2">
              System
            </div>

            <div className="space-y-1">
              {navItems.map(
                ({
                  label,
                  icon: Icon,
                  target,
                }) => {
                  const isDashboard =
                    label === "Dashboard";

                  return (
                    <button
                      key={label}
                      type="button"
                      onClick={() =>
                        scrollToSection(target)
                      }
                      className={`w-full flex items-center gap-3 px-3 py-2 rounded-sm border transition-colors text-left ${
                        isDashboard
                          ? "bg-[#111D35] border-[#1E3155] text-white"
                          : "border-transparent text-[#A5ADBA] hover:text-white hover:bg-[#0C1422] hover:border-[#172033]"
                      }`}
                    >
                      <Icon
                        size={14}
                        className={
                          isDashboard
                            ? "text-[#8CA9FF]"
                            : "text-[#8B95A7]"
                        }
                      />

                      <span className="text-[10px]">
                        {label}
                      </span>
                    </button>
                  );
                }
              )}
            </div>
          </nav>

          {/* Academic information */}
          <div className="px-4 pb-4">
            <div className="font-mono text-[7px] uppercase tracking-[0.08em] text-[#8B95A7] leading-4">
              <div className="text-[#B8C0CC]">
                MSc Artificial Intelligence
              </div>
              <div>University of Bedfordshire</div>
              <div>Edge-Cloud Research Project</div>
              <div className="mt-1">v1.0.0</div>
            </div>
          </div>

          {/* Sidebar status */}
          <div className="px-4 py-3 border-t border-[#172033] bg-[#060C15]">
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  edgeConnected && cloudConnected
                    ? "bg-[#72E06A]"
                    : "bg-[#F4C542]"
                }`}
              />

              <span className="font-mono text-[8px] text-[#AAB3C2]">
                {edgeConnected && cloudConnected
                  ? "System Online"
                  : "Infrastructure Partial"}
              </span>
            </div>
          </div>
        </aside>

        {/* ===================================================
            MAIN
        ==================================================== */}

        <main className="w-[calc(100%-270px)] ml-[270px] min-w-0">

          {/* Top bar */}
                    <header className="sticky top-0 z-20 h-[64px] bg-[#050A12]/95 backdrop-blur-md border-b border-[#172033] px-4 flex items-center justify-between gap-3">

            {/* Header title */}
            <div className="flex items-center gap-3">

              <div>
                <div className="font-mono text-[9px] text-[#B8C0CC] uppercase tracking-[0.06em] whitespace-nowrap">
                  Orchestration Engine
                  <span className="text-[#9B7CFF] ml-1">
                    (FastAPI)
                  </span>
                </div>

                <div className="hidden lg:block text-[7px] xl:text-[8px] text-[#657086] mt-0.5 whitespace-nowrap">
                  Central orchestrator and intelligent inference control plane
                </div>
              </div>
            </div>

            {/* Header status */}
            <div className="flex items-center gap-2 shrink-0">

              <button
                type="button"
                onClick={() =>
                  setRunning(
                    (value) => !value
                  )
                }
                className={`flex items-center gap-2 px-2.5 py-1 border rounded-sm font-mono text-[8px] ${
                  running
                    ? "border-[#285C35] bg-[#0C1C12] text-[#7EE787]"
                    : "border-[#6B2D2D] bg-[#211010] text-[#FF6B6B]"
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    running
                      ? "bg-[#72E06A]"
                      : "bg-[#FF5252]"
                  }`}
                />

                {running ? "LIVE" : "PAUSED"}
              </button>

              <div className="hidden md:flex items-center gap-2 font-mono text-[8px] text-[#A5ADBA] uppercase tracking-[0.08em]">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    edgeConnected && cloudConnected
                      ? "bg-[#72E06A]"
                      : "bg-[#F4C542]"
                  }`}
                />

                Real Infrastructure
              </div>

              <div className="h-6 w-px bg-[#172033]" />

              <button
                type="button"
                onClick={() =>
                  setShowArch(true)
                }
                className="hidden lg:block font-mono text-[8px] text-[#8B95A7] hover:text-white transition"
              >
                ARCHITECTURE
              </button>

              <div className="h-6 w-px bg-[#172033]" />

              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-full bg-[#5865F2] flex items-center justify-center font-mono text-[8px] text-white">
                  SA
                </div>

                <div className="hidden md:block">
                  <div className="font-mono text-[9px] text-white">
                    Sophia
                  </div>

                  <div className="font-mono text-[7px] text-[#778196]">
                    Admin
                  </div>
                </div>
              </div>

              <div className="hidden xl:block text-right">
                <div className="font-mono text-[7px] text-[#778196]">
                  {new Date().toLocaleDateString(
                    "en-GB",
                    {
                      weekday: "short",
                      day: "2-digit",
                      month: "short",
                      year: "numeric",
                    }
                  )}
                </div>

                <div className="font-mono text-[7px] text-[#A5ADBA] mt-0.5">
                  {new Date().toLocaleTimeString(
                    "en-GB",
                    {
                      hour: "2-digit",
                      minute: "2-digit",
                      second: "2-digit",
                    }
                  )}
                </div>
              </div>
            </div>
          </header>

          <div className="p-4 md:p-5 space-y-4 w-full">

            {/* =================================================
                DASHBOARD HERO
            ================================================== */}

            <section
              id="dashboard-section"
              className="scroll-mt-24"
            >
              {/* ORCHESTRATION ENGINE TOP PANEL */}
              <div id="orchestration-engine-section" className="mb-3 border border-[#27272A] bg-[#08090B]">

                {/* Panel header */}
                <div className="px-4 py-3 border-b border-[#27272A] flex items-start gap-3">
                  <div className="font-mono text-[9px] px-2 py-1 border border-[#27272A] text-[#A1A1AA]">
                    API
                  </div>

                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.08em] text-[#A1A1AA]">
                      ORCHESTRATION ENGINE <span className="text-[#8B5CF6]">(FastAPI)</span>
                    </div>

                    <div className="font-mono text-[8px] text-[#52525B] mt-1">
                      Central orchestrator and control plane - manages the intelligent inference workflow.
                    </div>
                  </div>
                </div>

                <div className="px-3 pt-3">
                  <h2 className="font-mono text-lg flex items-center gap-2"><Cube size={18} className="text-[#8BAEFF]" />Dashboard Overview</h2>
                  <p className="font-mono text-[9px] text-[#71717A] mt-1">Overview of the implemented Edge-Cloud Orchestration Architecture, Integrating Infrastructure Monitoring, Intelligent Routing Decisions and Execution Performance Evaluation.</p>
                </div>

                {/* Five-column orchestration overview */}
                <div className="grid grid-cols-[0.9fr_1.55fr_1.1fr_1fr_1.25fr] gap-2 p-3 min-w-0">

                  {/* COMMANDS */}
                  <div className="border border-[#1F1F22] p-3">
                    <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#71717A] mb-3">
                      Orchestration Commands
                    </div>

                    <div className="space-y-1.5 font-mono text-[8px] text-[#A1A1AA]">
                      <div>→ Telemetry Collection</div>
                      <div>→ Context Aggregation</div>
                      <div>→ Policy Evaluation</div>
                      <div>→ Route Decision</div>
                      <div>→ Execution Dispatch</div>
                      <div>→ Failover Management</div>
                      <div>&rarr; Result Collection</div>
                      <div>&rarr; Learning & Optimisation</div>
                      <div>&rarr; Logging & Monitoring</div>
                    </div>
                  </div>

                  {/* SYSTEM FLOW */}
                  <div className="border border-[#1F1F22] p-3 overflow-hidden min-w-0">
                    <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#71717A] mb-3">
                      System Flow
                    </div>

                    <div className="relative h-[215px]">

                      {/* TOP NODES */}
                      <div className="relative z-20 grid grid-cols-[1fr_28px_1fr_28px_1fr] items-center">

                        {/* EDGE NODE */}
                        <div className="h-[94px] border border-[#536638] bg-[#0D120C] flex flex-col items-center justify-center">
                          <div className="font-mono text-[9px] font-semibold text-[#B2D86B]">
                            EDGE NODE
                          </div>

                          <Cpu
                            size={32}
                            weight="duotone"
                            className="text-[#A8C85E] mt-2"
                          />
                        </div>

                        <div className="text-center font-mono text-[19px] text-[#B4C764]">
                          &larr;</div>

                        {/* AI DECISION ENGINE */}
                        <div className="h-[94px] border border-[#665B2B] bg-[#151309] flex flex-col items-center justify-center">
                          <div className="font-mono text-[9px] font-semibold text-[#D6C768] whitespace-nowrap">
                            AI DECISION ENGINE
                          </div>

                          <Brain
                            size={36}
                            weight="duotone"
                            className="text-[#CBB85C] mt-2"
                          />
                        </div>

                        <div className="text-center font-mono text-[19px] text-[#C7B95C]">
                          &rarr;</div>

                        {/* AWS CLOUD */}
                        <div className="h-[94px] border border-[#4A5278] bg-[#0D0F18] flex flex-col items-center justify-center">
                          <div className="font-mono text-[9px] font-semibold text-[#A7AED7]">
                            AWS CLOUD
                          </div>

                          <div className="mt-1 flex flex-col items-center leading-none">
                            <div className="font-sans text-[25px] font-normal tracking-[-0.08em] text-[#E1E1E4]">
                              aws
                            </div>

                            <div className="relative w-[43px] h-[9px] mt-[-2px]">
                              <div className="absolute left-[3px] top-0 w-[35px] h-[7px] border-b-2 border-[#C7A44D] rounded-[50%]" />
                              <div className="absolute right-[1px] bottom-0 w-[6px] h-[6px] border-r-2 border-b-2 border-[#C7A44D] rotate-[-20deg]" />
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* ================================================= */}
                      {/* CONNECTION NETWORK ONLY                           */}
                      {/* ================================================= */}

                      {/* EDGE NODE DOWN */}
                      <div className="absolute z-0 left-[16.7%] top-[94px] h-[35px]
                                      border-l-2 border-dashed border-[#81759D]" />

                      {/* AI ENGINE DOWN */}
                      <div className="absolute z-0 left-1/2 top-[94px] h-[23px]
                                      border-l-2 border-[#625879]" />

                      {/* AWS CLOUD DOWN */}
                      <div className="absolute z-0 right-[16.7%] top-[94px] h-[35px]
                                      border-l-2 border-dashed border-[#81759D]" />

                      {/* OUTER FEEDBACK LOOP */}
                      <div className="absolute z-0 left-[7%] right-[7%] top-[128px] h-[76px]
                                      border-l-2 border-r-2 border-b-2 border-dashed
                                      border-[#81759D] rounded-b-[9px]" />

                      {/* TOP DASHED LINE - LEFT SIDE */}
                      <div className="absolute z-0 left-[7%] top-[128px] w-[43%]
                                      border-t-2 border-dashed border-[#81759D]" />

                      {/* TOP DASHED LINE - RIGHT SIDE */}
                      <div className="absolute z-0 right-[7%] top-[128px] w-[43%]
                                      border-t-2 border-dashed border-[#81759D]" />

                      {/* CONTEXT DATA */}
                      <div className="absolute z-20 left-1/2 -translate-x-1/2 top-[116px]
                                      w-[52%] h-[26px]
                                      border border-[#514B68] bg-[#111117]
                                      flex items-center justify-center
                                      font-mono text-[8px] font-semibold text-[#AAA2C5]">
                        CONTEXT DATA
                      </div>

                      {/* INNER ROUTE LOOP */}
                      <div className="absolute z-0 left-[22%] right-[22%] top-[163px] h-[41px]
                                      border-l-2 border-r-2 border-b-2
                                      border-[#625879] rounded-b-[5px]" />

                      {/* LEFT CONNECTION INTO ROUTE */}
                      <div className="absolute z-10 left-[22%] top-[164px] w-[28%]
                                      border-t-2 border-[#625879]" />

                      {/* RIGHT CONNECTION INTO ROUTE */}
                      <div className="absolute z-10 right-[22%] top-[164px] w-[28%]
                                      border-t-2 border-[#625879]" />

                      {/* LEFT ARROW */}
                      <div className="absolute z-10 left-[21.5%] top-[158px]
                                      font-mono text-[11px] text-[#756A98]">
                        ?
                      </div>

                      {/* RIGHT ARROW */}
                      <div className="absolute z-10 right-[21.5%] top-[158px]
                                      font-mono text-[11px] text-[#756A98] rotate-180">
                        ?
                      </div>

                      {/* ROUTE EXECUTION */}
                      <div className="absolute z-20 left-1/2 -translate-x-1/2 top-[151px]
                                      w-[52%] h-[27px]
                                      border border-[#665A26] bg-[#15130C]
                                      flex items-center justify-center
                                      font-mono text-[8px] font-semibold text-[#D1BE58]">
                        ROUTE EXECUTION
                      </div>

                      {/* ROUTE -> RESULT */}
                      <div className="absolute z-10 left-1/2 top-[178px] h-[7px]
                                      border-l-2 border-[#625879]" />

                      {/* RESULT & LEARNING */}
                      <div className="absolute z-20 left-1/2 -translate-x-1/2 top-[184px]
                                      w-[52%] h-[27px]
                                      border border-[#514763] bg-[#111017]
                                      flex items-center justify-center
                                      font-mono text-[8px] font-semibold text-[#A99BC8]">
                        RESULT & LEARNING
                      </div>

                    </div>
                  </div>

                  {/* STATUS */}
                  <div className="border border-[#1F1F22] p-3">
                    <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#71717A] mb-3">
                      Orchestration Status
                    </div>

                    <div className="space-y-2 font-mono text-[8px]">
                      <div className="flex justify-between">
                        <span className="text-[#71717A]">Engine Status</span>
                        <span className="text-[#A3E635]">? ONLINE</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#71717A]">Version</span>
                        <span>1.0.0</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#71717A]">Primary Policies</span>
                        <span>2</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#71717A]">Active Routes</span>
                        <span>3</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-[#71717A]">Health</span>
                        <span className="text-[#A3E635]">? HEALTHY</span>
                      </div>
                    </div>
                  </div>

                  {/* ENDPOINTS */}
                  <div className="border border-[#1F1F22] p-3">
                    <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#71717A] mb-3">
                      Control Endpoints
                    </div>

                    <div className="space-y-2 font-mono text-[8px]">
                      <div className="flex justify-between"><span>GET /health</span><span className="text-[#A3E635]">✓ OK</span></div>
                      <div className="flex justify-between"><span>POST /predict-live</span><span className="text-[#A3E635]">✓ OK</span></div>
                      <div className="flex justify-between"><span>GET /telemetry</span><span className="text-[#A3E635]">✓ OK</span></div>
                      <div className="flex justify-between"><span>GET /decisions</span><span className="text-[#A3E635]">✓ OK</span></div>
                      <div className="flex justify-between"><span>GET /models</span><span className="text-[#A3E635]">✓ OK</span></div>
                      <div className="flex justify-between"><span>GET /policies</span><span className="text-[#A3E635]">✓ OK</span></div>
                    </div>
                  </div>

                  {/* METRICS */}
                  <div className="border border-[#1F1F22] p-3">
                    <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#71717A] mb-3">
                      Orchestration Metrics
                    </div>

                    <div className="space-y-3 font-mono">

                      {/* Avg Orchestration Time */}
                      <div className="flex items-end justify-between gap-3">
                        <div>
                          <div className="text-[7px] text-[#71717A]">
                            Avg Orchestration Time
                          </div>
                          <div className="text-[11px] text-white mt-1">
                            {Number(liveDecision?.decision_latency_us || 0).toFixed(1)} us
                          </div>
                        </div>

                        <div className="flex items-end gap-[2px] h-[18px]">
                          {orchestrationSeries.latency.map((value, i) => { const h = sparkHeight(value, orchestrationSeries.latency); return (
                            <div
                              key={`lat-${i}`}
                              className="w-[2px] bg-[#8B5CF6]"
                              style={{ height: `${h}px` }}
                            />
                          ); })}
                        </div>
                      </div>

                      {/* Decision Throughput */}
                      <div className="flex items-end justify-between gap-3">
                        <div>
                          <div className="text-[7px] text-[#71717A]">
                            Decision Throughput
                          </div>
                          <div className="text-[11px] text-white mt-1">
                            {decisionHistory?.length > 1
                              ? (
                                  decisionHistory.length /
                                  Math.max(
                                    1,
                                    (
                                      Number(decisionHistory[0]?.created_at || 0) -
                                      Number(decisionHistory[decisionHistory.length - 1]?.created_at || 0)
                                    )
                                  )
                                ).toFixed(2)
                              : "--"} req/s
                          </div>
                        </div>

                        <div className="flex items-end gap-[2px] h-[18px]">
                          {orchestrationSeries.throughput.map((value, i) => { const h = sparkHeight(value, orchestrationSeries.throughput); return (
                            <div
                              key={`thr-${i}`}
                              className="w-[2px] bg-[#A3E635]"
                              style={{ height: `${h}px` }}
                            />
                          ); })}
                        </div>
                      </div>

                      {/* Q-Learning Updates */}
                      <div className="flex items-end justify-between gap-3">
                        <div>
                          <div className="text-[7px] text-[#71717A]">
                            Real Q-Learning Updates
                          </div>
                          <div className="text-[11px] text-white mt-1">
                            {Number(engineStatus?.QLearning?.metrics?.real_updates || 0).toLocaleString()}
                          </div>
                        </div>

                        <div className="flex items-end gap-[2px] h-[18px]">
                          {orchestrationSeries.qUpdates.map((value, i) => { const h = sparkHeight(value, orchestrationSeries.qUpdates); return (
                            <div
                              key={`ql-${i}`}
                              className="w-[2px] bg-[#FACC15]"
                              style={{ height: `${h}px` }}
                            />
                          ); })}
                        </div>
                      </div>

                      {/* Last Real Execution */}
                      <div className="flex items-end justify-between gap-3">
                        <div>
                          <div className="text-[7px] text-[#71717A]">
                            Last Real Execution
                          </div>
                          <div
                            className={`text-[11px] mt-1 ${
                              engineStatus?.QLearning?.metrics?.last_real_success === true
                                ? "text-[#A3E635]"
                                : engineStatus?.QLearning?.metrics?.last_real_success === false
                                  ? "text-[#FF6B6B]"
                                  : "text-[#A1A1AA]"
                            }`}
                          >
                            {engineStatus?.QLearning?.metrics?.last_real_success === true
                              ? "SUCCESS"
                              : engineStatus?.QLearning?.metrics?.last_real_success === false
                                ? "FAILED"
                                : "--"}
                          </div>
                        </div>

                        <div className="flex items-end gap-[2px] h-[18px]">
                          {orchestrationSeries.success.map((value, i) => { const h = value === 1 ? 18 : 4; return (
                            <div
                              key={`exec-${i}`}
                              className="w-[2px] bg-[#5B8CFF]"
                              style={{ height: `${h}px` }}
                            />
                          ); })}
                        </div>
                      </div>

                    </div>
                  </div>

                </div>
              </div>

              {/* =================================================
                NODES
            ================================================== */}

            <section
              id="nodes-section"
              className="scroll-mt-24"
            >
              <div className="flex items-end justify-between mb-3">
                <div>
                  <Overline>
                    COMPUTE NODES
                  </Overline>

                  <h2 className="font-mono text-lg mt-1">
                    Edge + AWS Cloud
                  </h2>
                </div>

                <span className="font-mono text-[9px] text-[#71717A]">
                  REAL INFRASTRUCTURE
                </span>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">

                {/* Edge Node Details */}
                <Cell className="p-5">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 border border-[#00E676]/40 flex items-center justify-center">
                        <Cpu
                          size={20}
                          className="text-[#00E676]"
                        />
                      </div>

                      <div>
                        <div className="font-mono text-sm">
                          {edgeInfra
                            ?.node ||
                            "edge-node-01"}
                        </div>

                        <div className="font-mono text-[9px] text-[#71717A] mt-1">
                          Edge Compute Node
                        </div>
                      </div>
                    </div>

                    <span
                      className={`font-mono text-[9px] ${
                        edgeConnected
                          ? "text-[#00E676]"
                          : "text-[#FF3333]"
                      }`}
                    >
                      {edgeConnected
                        ? "ONLINE"
                        : "OFFLINE"}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-5">
                    <div className="border border-[#27272A] p-2">
                      <div className="font-mono text-[8px] text-[#52525B]">
                        CPU
                      </div>

                      <div className="font-mono text-sm mt-1">
                        {formatMetric(
                          edgeCpu
                        )}
                        %
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-2">
                      <div className="font-mono text-[8px] text-[#52525B]">
                        RAM
                      </div>

                      <div className="font-mono text-sm mt-1">
                        {formatMetric(
                          edgeMemory
                        )}
                        %
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-2">
                      <div className="font-mono text-[8px] text-[#52525B]">
                        DISK
                      </div>

                      <div className="font-mono text-sm mt-1">
                        {formatMetric(
                          edgeDisk
                        )}
                        %
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-2">
                      <div className="font-mono text-[8px] text-[#52525B]">
                        RTT
                      </div>

                      <div className="font-mono text-sm mt-1">
                        {formatMetric(
                          edgeRtt
                        )}
                        ms
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 border-t border-[#27272A] pt-3 flex items-center justify-between">
                    <div className="font-mono text-[9px] text-[#71717A]">
                      VirtualBox VM → Linux → AWS IoT Greengrass → Edge Runtime
                    </div>

                    <span className="font-mono text-[9px] uppercase text-[#3478FF]">
                      GREENGRASS
                    </span>
                  </div>
                </Cell>

                {/* Cloud Node Details */}
                <Cell className="p-5">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 border border-[#0055FF]/40 flex items-center justify-center">
                        <CloudArrowUp
                          size={20}
                          className="text-[#0055FF]"
                        />
                      </div>

                      <div>
                        <div className="font-mono text-sm">
                          {cloudInfra
                            ?.node ||
                            "cloud-node-01"}
                        </div>

                        <div className="font-mono text-[9px] text-[#71717A] mt-1">
                          AWS EC2 | eu-west-2
                        </div>
                      </div>
                    </div>

                    <span
                      className={`font-mono text-[9px] ${
                        cloudConnected
                          ? "text-[#00E676]"
                          : "text-[#FF3333]"
                      }`}
                    >
                      {cloudConnected
                        ? "ONLINE"
                        : "OFFLINE"}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-5">
                    <div className="border border-[#27272A] p-2">
                      <div className="font-mono text-[8px] text-[#52525B]">
                        CPU
                      </div>

                      <div className="font-mono text-sm mt-1">
                        {formatMetric(
                          cloudCpu
                        )}
                        %
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-2">
                      <div className="font-mono text-[8px] text-[#52525B]">
                        RAM
                      </div>

                      <div className="font-mono text-sm mt-1">
                        {formatMetric(
                          cloudMemory
                        )}
                        %
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-2">
                      <div className="font-mono text-[8px] text-[#52525B]">
                        DISK
                      </div>

                      <div className="font-mono text-sm mt-1">
                        {formatMetric(
                          cloudDisk
                        )}
                        %
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-2">
                      <div className="font-mono text-[8px] text-[#52525B]">
                        RTT
                      </div>

                      <div className="font-mono text-sm mt-1">
                        {formatMetric(
                          cloudRtt
                        )}
                        ms
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 border-t border-[#27272A] pt-3 flex items-center justify-between">
                    <div className="font-mono text-[9px] text-[#71717A]">
                      AWS EC2 → Linux → systemd / FastAPI → Cloud Runtime
                    </div>

                    <span className="font-mono text-[9px] text-[#0055FF]">
                      AWS
                    </span>
                  </div>
                </Cell>
              </div>
            </section>

            
            <section
              id="decision-section"
              className="scroll-mt-24"
            >
              <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-3 mb-3">
                <div>
                  <Overline>
                    DECISION ENGINE
                  </Overline>

                  <h2 className="font-mono text-lg mt-1">
                    Intelligent Routing Policies
                  </h2>
                </div>
              </div>

              {/* Policy selector */}
              <div className="grid grid-cols-2 gap-2">
                {engineOptions.map(
                  (
                    policy
                  ) => {
                    const isActive =
                      engineName ===
                      policy.value;

                    const fullName =
                      engineFullName(
                        policy.value
                      );

                    const status =
                      engineStatus?.[
                        fullName
                      ];

                    return (
                      <button
                        key={
                          policy.value
                        }
                        type="button"
                        onClick={() =>
                          setEngineName(
                            policy.value
                          )
                        }
                        className={`text-left border p-4 transition-colors ${
                          isActive
                            ? "border-[#0055FF] bg-[#0055FF]/10"
                            : "border-[#27272A] bg-[#0A0A0A] hover:border-[#52525B]"
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <span className="font-mono text-[10px] uppercase text-white">
                            {
                              policy.label
                            }
                          </span>

                          <span
                            className={`w-1.5 h-1.5 rounded-full ${
                              status
                                ?.available ===
                              false
                                ? "bg-[#FF3333]"
                                : "bg-[#00E676]"
                            }`}
                          />
                        </div>

                        <div className="font-mono text-[8px] text-[#71717A] mt-3">
                          {policy.value ===
                          "rule"
                            ? "FIXED RULES / BASELINE"
                            : policy.value ===
                                "dt"
                              ? `ACC ${(
                                  Number(
                                    status
                                      ?.metrics
                                      ?.accuracy ||
                                      0
                                  ) *
                                  100
                                ).toFixed(
                                  1
                                )}%`
                              : policy.value ===
                                  "rf"
                                ? `ACC ${(
                                    Number(
                                      status
                                        ?.metrics
                                        ?.accuracy ||
                                        0
                                    ) *
                                    100
                                  ).toFixed(
                                    1
                                  )}%`
                                : "ADAPTIVE LEARNED POLICY"}
                        </div>
                      </button>
                    );
                  }
                )}
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-[1.1fr_0.9fr] gap-3 mt-3">

                {/* Active policy */}
                <Cell className="p-5">
                  <div className="flex items-start justify-between">
                    <div>
                      <Overline>
                        ACTIVE POLICY
                      </Overline>

                      <div className="font-mono text-2xl text-[#0055FF] mt-2">
                        {engineLabel(
                          engineName
                        )}
                      </div>
                    </div>

                    <span className="font-mono text-[9px] border border-[#00E676]/30 text-[#00E676] px-2 py-1">
                      {activePolicyTrained
                        ? "TRAINED"
                        : "READY"}
                    </span>
                  </div>

                  <p className="font-mono text-[10px] leading-5 text-[#A1A1AA] mt-5">
                    {
                      selectedEngineDescription
                    }
                  </p>

                  <div className="border-t border-[#27272A] mt-4 pt-4">
                    <Overline>
                      OBJECTIVE
                    </Overline>

                    <p className="font-mono text-[10px] leading-5 text-[#71717A] mt-2">
                      {
                        selectedEngineObjective
                      }
                    </p>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-5">
                    <div className="border border-[#27272A] p-3">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        Route
                      </div>

                      <div
                        className={`font-mono text-sm uppercase mt-1 ${routeTone(
                          currentExecutedRoute
                        )}`}
                      >
                        {
                          currentExecutedRoute
                        }
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-3">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        Confidence
                      </div>

                      <div className="font-mono text-sm mt-1">
                        {liveDecision
                          ?.confidence !=
                        null
                          ? `${(
                              Number(
                                liveDecision
                                  .confidence
                              ) *
                              100
                            ).toFixed(
                              1
                            )}%`
                          : "--"}
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-3">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        Decision
                      </div>

                      <div className="font-mono text-sm mt-1">
                        {formatMetric(
                          liveDecision
                            ?.decision_latency_us,
                          1
                        )}
                        <span className="text-[8px] text-[#71717A] ml-1">
                          us
                        </span>
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-3">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        {engineName ===
                        "ql"
                          ? "Reward"
                          : "Accuracy"}
                      </div>

                      <div className="font-mono text-sm mt-1">
                        {engineName ===
                        "ql"
                          ? qReward
                          : policyAccuracy}
                      </div>
                    </div>
                  </div>
                </Cell>

                {/* Q-learning status */}
                <Cell className="p-5">
                  <div className="flex items-center justify-between">
                    <div>
                      <Overline>
                        Q-LEARNING STATUS
                      </Overline>

                      <div className="font-mono text-[9px] text-[#71717A] mt-1">
                        Reinforcement-learning agent
                      </div>
                    </div>

                    <Lightning
                      size={18}
                      className="text-[#FFCC00]"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-2 mt-5">
                    <div className="border border-[#27272A] p-3">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        Episodes
                      </div>

                      <div className="font-mono text-lg mt-1">
                        {qLearningMetrics
                          ?.episodes ??
                          "--"}
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-3">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        States
                      </div>

                      <div className="font-mono text-lg mt-1">
                        {qLearningMetrics
                          ?.n_states ??
                          "--"}
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-3">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        Actions
                      </div>

                      <div className="font-mono text-lg mt-1">
                        {qLearningMetrics
                          ?.n_actions ??
                          "--"}
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-3">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        Reward
                      </div>

                      <div className="font-mono text-lg text-[#FFCC00] mt-1">
                        {
                          qReward
                        }
                      </div>
                    </div>
                  </div>

                  <div className="mt-4">
                    <button
                      type="button"
                      onClick={
                        refreshQTable
                      }
                      className="w-full font-mono text-[9px] uppercase py-2 border border-[#27272A] text-[#A1A1AA] hover:text-white hover:border-[#FFCC00]"
                    >
                      Refresh Q-Table
                    </button>
                  </div>

                  {engineName ===
                    "ql" && (
                    <div className="mt-4 border border-[#FFCC00]/20 bg-[#FFCC00]/5 p-3">
                      <div className="font-mono text-[8px] uppercase tracking-[0.15em] text-[#FFCC00]">
                        CURRENT Q-DECISION
                      </div>

                      <div className="font-mono text-[10px] text-[#A1A1AA] leading-5 mt-2 break-words">
                        {liveDecision
                          ?.reason ||
                          "Waiting for a Q-Learning decision..."}
                      </div>
                    </div>
                  )}
                </Cell>
              </div>
            </section>
            {/* =================================================
                HOTEL APPLICATION SCENARIO
            ================================================== */}

            
            <section
              id="scenarios-section"
              className="scroll-mt-24"
            >
              <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-3 mb-3">
                <div>
                  <Overline>
                    HOTEL APPLICATION SCENARIO
                  </Overline>

                                    <div className="font-mono text-[10px] text-[#71717A] mt-2">
                    Hotel Operations Monitoring
                  </div>

                  <div className="mt-4 mb-4 border border-[#27272A] bg-[#0A0A0A] max-w-[560px] px-6 py-5">
                    <div className="flex items-center gap-10">

                      <div className="w-[92px] h-[92px] flex items-center justify-center text-[#E4E4E7]">
                        <svg
                          viewBox="0 0 64 64"
                          className="w-[72px] h-[72px]"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          aria-label="Hotel"
                        >
                          <path d="M20 55V15h24v40" />
                          <path d="M14 55V25h6" />
                          <path d="M44 25h6v30" />
                          <path d="M10 55h44" />

                          <path d="M26 21h4" />
                          <path d="M34 21h4" />
                          <path d="M26 28h4" />
                          <path d="M34 28h4" />
                          <path d="M26 35h4" />
                          <path d="M34 35h4" />

                          <path d="M17 31h3" />
                          <path d="M17 38h3" />
                          <path d="M17 45h3" />

                          <path d="M44 31h3" />
                          <path d="M44 38h3" />
                          <path d="M44 45h3" />

                          <path d="M27 55V47c0-3 2-5 5-5s5 2 5 5v8" />
                        </svg>
                      </div>

                      <div className="h-[72px] w-px bg-[#27272A]" />

                      <div>
                        <div className="font-mono text-[25px] tracking-[0.16em] text-[#D8CF76] whitespace-nowrap">
                          ★★★★★
                        </div>

                        <div className="font-mono text-[8px] uppercase tracking-[0.18em] text-[#52525B] mt-2">
                          Hotel Operations Scenario
                        </div>
                      </div>

                    </div>
                  </div>

                  <p className="font-mono text-[9px] text-[#71717A] mt-1 max-w-3xl">
                    Conceptual application of the Edge-Cloud orchestration architecture
                    to hotel operational data and business decision support.
                  </p>
                </div>

                <div className="border border-[#27272A] px-3 py-2 min-w-[150px]">
                  <div className="font-mono text-[8px] uppercase text-[#52525B]">
                    Application
                  </div>
                  <div className="font-mono text-[10px] text-white mt-1">
                    Hotel Operations
                  </div>
                </div>
              </div>

              <Cell className="p-4">
                <div className="grid grid-cols-1 xl:grid-cols-[1fr_28px_1fr_28px_1fr_28px_1fr_28px_1fr] gap-2 items-stretch">

                  {/* HOTEL DATA */}
                  <div className="border border-[#2F3642] bg-[#080A0D] p-4 min-w-0">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-full border border-[#3F4652] bg-[#111318] flex items-center justify-center">
                        <Database
                          size={18}
                          weight="duotone"
                          className="text-[#D4D4D8]"
                        />
                      </div>

                      <div className="font-mono text-[8px] uppercase tracking-[0.14em] text-[#A1A1AA]">
                        Hotel Data Sources
                      </div>
                    </div>

                    <div className="mt-4 space-y-2 font-mono text-[9px] text-[#D4D4D8]">
                      <div>Room Status</div>
                      <div>Occupancy Level</div>
                      <div>Parking Occupancy</div>
                      <div>Maintenance Events</div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-[#27272A] font-mono text-[7px] text-[#596579]">
                      PMS · IoT · SENSORS · CAMERAS
                    </div>
                  </div>

                  <div className="flex items-center justify-center font-mono text-[#39FF88] text-lg">
                    →
                  </div>

                  {/* EDGE */}
                  <div className="border border-[#2D6B42] bg-[#08100A] p-4 min-w-0">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-full border border-[#2D6B42] bg-[#0A1A10] flex items-center justify-center">
                        <Cpu
                          size={18}
                          weight="duotone"
                          className="text-[#39FF88]"
                        />
                      </div>

                      <div className="font-mono text-[8px] uppercase tracking-[0.14em] text-[#39FF88]">
                        Edge Processing
                      </div>
                    </div>

                    <div className="font-mono text-[8px] text-[#A1A1AA] mt-4 leading-5">
                      Local data processing
                      <br />
                      Computer vision inference
                      <br />
                      Low-latency response
                    </div>

                    <div className="mt-4 pt-3 border-t border-[#27272A] font-mono text-[7px] text-[#39FF88]">
                      LOCAL · FAST · PRIVATE
                    </div>
                  </div>

                  <div className="flex items-center justify-center font-mono text-[#E7D64D] text-lg">
                    →
                  </div>

                  {/* ORCHESTRATION */}
                  <div className="border border-[#665B2B] bg-[#100E06] p-4 min-w-0">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-full border border-[#665B2B] bg-[#191604] flex items-center justify-center">
                        <Brain
                          size={18}
                          weight="duotone"
                          className="text-[#E7D64D]"
                        />
                      </div>

                      <div className="font-mono text-[8px] uppercase tracking-[0.14em] text-[#E7D64D]">
                        AI Orchestration
                      </div>
                    </div>

                    <div className="font-mono text-[8px] text-[#A1A1AA] mt-4 leading-5">
                      Context evaluation
                      <br />
                      Policy-based decision
                      <br />
                      Edge / Cloud / Hybrid routing
                    </div>

                    <div className="mt-4 pt-3 border-t border-[#27272A] font-mono text-[7px] text-[#E7D64D]">
                      DECIDE · ROUTE · ADAPT
                    </div>
                  </div>

                  <div className="flex items-center justify-center font-mono text-[#4D78FF] text-lg">
                    →
                  </div>

                  {/* CLOUD */}
                  <div className="border border-[#334A8A] bg-[#070A12] p-4 min-w-0">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-full border border-[#334A8A] bg-[#0A1020] flex items-center justify-center">
                        <CloudArrowUp
                          size={18}
                          weight="duotone"
                          className="text-[#4D78FF]"
                        />
                      </div>

                      <div className="font-mono text-[8px] uppercase tracking-[0.14em] text-[#4D78FF]">
                        AWS Cloud
                      </div>
                    </div>

                    <div className="font-mono text-[8px] text-[#A1A1AA] mt-4 leading-5">
                      Cloud processing
                      <br />
                      Remote AI inference
                      <br />
                      Compute resources
                    </div>

                    <div className="mt-4 pt-3 border-t border-[#27272A] font-mono text-[7px] text-[#4D78FF]">
                      COMPUTE · INFERENCE · SCALABILITY
                    </div>
                  </div>

                  <div className="flex items-center justify-center font-mono text-[#C084FC] text-lg">
                    →
                  </div>

                  {/* BUSINESS */}
                  <div className="border border-[#69408A] bg-[#0D0812] p-4 min-w-0">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-full border border-[#69408A] bg-[#160A20] flex items-center justify-center">
                        <ChartLineUp
                          size={18}
                          weight="duotone"
                          className="text-[#C084FC]"
                        />
                      </div>

                      <div className="font-mono text-[8px] uppercase tracking-[0.14em] text-[#C084FC]">
                        Business Value
                      </div>
                    </div>

                    <div className="mt-4 space-y-2 font-mono text-[9px] text-[#D4D4D8]">
                      <div>Operational Efficiency</div>
                      <div>Revenue Insights</div>
                      <div>Maintenance Support</div>
                      <div>Decision Support</div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-[#27272A] font-mono text-[7px] text-[#C084FC]">
                      INSIGHTS · REPORTS · ACTIONS
                    </div>
                  </div>

                </div>

                <div className="mt-4 border border-dashed border-[#27272A] px-4 py-3">
                  <div className="font-mono text-[8px] text-[#71717A] leading-5">
                    APPLICATION SCENARIO — Illustrates how the proposed Edge-Cloud
                    orchestration architecture could support hotel operations.
                  </div>
                </div>

              </Cell>
            </section>




            {/* =================================================
                EXPERIMENTS
            ================================================== */}

                        {/* =================================================
                AI WORKLOAD RUNNER
            ================================================== */}

            <section
              id="settings-section"
              className="scroll-mt-24"
            >
              <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-3 mb-3">
                <div>
                  <Overline>
                    AI WORKLOAD RUNNER
                  </Overline>

                  <h2 className="font-mono text-lg mt-1">
                    Real Inference Execution
                  </h2>

                  <p className="font-mono text-[9px] text-[#71717A] mt-1">
                    Execute real YOLO or Florence-2 AI workloads using automatic or manual Edge-Cloud routing.
                  </p>
                </div>

                <div className="font-mono text-[9px] text-[#71717A] border border-[#27272A] px-3 py-2">
                  Route:{" "}
                  <span
                    className={`uppercase ${routeTone(
                      mode
                    )}`}
                  >
                    {mode}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-[0.8fr_1.2fr] gap-3">

                {/* Workload Controls */}
                <Cell className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Overline>
                        WORKLOAD CONFIGURATION
                      </Overline>

                      <div className="font-mono text-[9px] text-[#71717A] mt-1">
                        Select the inference workload to execute.
                      </div>
                    </div>

                    <Rocket
                      size={16}
                      className="text-[#0055FF]"
                    />
                  </div>

                  {/* Image input */}
                  {[
                    "yolo",
                    "florence",
                  ].includes(
                    workloadType
                  ) && (
                    <div className="mt-4">
                      <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#52525B] mb-2">
                        Camera Analysis
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mt-3">
                        <button
                          type="button"
                          onClick={startCamera}
                          disabled={cameraActive}
                          className="border border-[#0055FF] px-3 py-2 font-mono text-[8px] uppercase text-[#6B8FFF] hover:bg-[#0055FF]/10 disabled:opacity-40"
                        >
                          Start Camera
                        </button>

                        <button
                          type="button"
                          onClick={captureCameraFrame}
                          disabled={!cameraActive || cameraUploading}
                          className="border border-[#39FF88] px-3 py-2 font-mono text-[8px] uppercase text-[#39FF88] hover:bg-[#39FF88]/10 disabled:opacity-40"
                        >
                          {cameraUploading
                            ? "Capturing..."
                            : "Capture & Analyse"}
                        </button>

                        <button
                          type="button"
                          onClick={stopCamera}
                          disabled={!cameraActive}
                          className="border border-[#FF5C5C] px-3 py-2 font-mono text-[8px] uppercase text-[#FF5C5C] hover:bg-[#FF5C5C]/10 disabled:opacity-40"
                        >
                          Stop Camera
                        </button>

                        <button
                          type="button"
                          onClick={powerOffCamera}
                          disabled={!cameraActive && !cameraPreview && !lastRealInference}
                          className="border border-[#FFCC00] px-3 py-2 font-mono text-[8px] uppercase text-[#FFCC00] hover:bg-[#FFCC00]/10 disabled:opacity-40"
                        >
                          Power Off
                        </button>
                      </div>

                      <div className="mt-3 border border-[#27272A] bg-black overflow-hidden">
                        <div className="flex items-center justify-between border-b border-[#27272A] px-3 py-2">
                          <span className="font-mono text-[8px] uppercase text-[#71717A]">
                            PC Camera
                          </span>

                          <span
                            className={`font-mono text-[8px] uppercase ${
                              cameraActive
                                ? "text-[#39FF88]"
                                : "text-[#52525B]"
                            }`}
                          >
                            {cameraActive ? "LIVE" : "OFF"}
                          </span>
                        </div>

                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          muted
                          className={`w-full aspect-video object-cover ${
                            cameraActive ? "block" : "hidden"
                          }`}
                        />

                        {!cameraActive && cameraPreview && (
                          <img
                            src={cameraPreview}
                            alt="Captured camera frame"
                            className="w-full aspect-video object-cover"
                          />
                        )}

                        {!cameraActive && !cameraPreview && (
                          <div className="aspect-video flex items-center justify-center font-mono text-[9px] text-[#3F3F46]">
                            CAMERA PREVIEW
                          </div>
                        )}
                      </div>

                      {cameraPreview && (
                        <div className="mt-2 font-mono text-[8px] text-[#39FF88]">
                          CAPTURED FRAME READY
                        </div>
                      )}
                    </div>
                  )}


                  {/* Workload type */}
                  <div className="mt-5">
                    <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#52525B] mb-2">
                      AI Workload
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      {[
                        {
                          value: "yolo",
                          label: "YOLOv8",
                          role: "Object Detection",
                          examples: "Parking • Maintenance • Operations",
                        },
                        {
                          value: "florence",
                          label: "Florence-2",
                          role: "Scene Understanding",
                          examples: "Rooms • Parking • Environment",
                        },
                      ].map(
                        (item) => (
                          <button
                            key={item.value}
                            type="button"
                            onClick={() => setWorkloadType(item.value)}
                            className={`border px-3 py-3 font-mono ${
                              workloadType === item.value
                                ? "border-[#0055FF] bg-[#0055FF]/10 text-white"
                                : "border-[#27272A] text-[#71717A] hover:text-white"
                            }`}
                          >
                            <div className="flex flex-col items-center gap-1 normal-case">
                              <span className="uppercase text-[9px]">
                                {item.label}
                              </span>
                              <span className="uppercase text-[7px] tracking-[0.12em] text-[#A1A1AA]">
                                {item.role}
                              </span>
                              <span className="text-[7px] text-[#52525B]">
                                {item.examples}
                              </span>
                            </div>
                          </button>
                        )
                      )}
                    </div>
                  </div>
                  {/* YOLO confidence */}
                  {workloadType ===
                    "yolo" && (
                    <div className="mt-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#52525B]">
                          Confidence Threshold
                        </div>

                        <span className="font-mono text-[9px] text-white">
                          {Number(
                            confidence
                          ).toFixed(
                            2
                          )}
                        </span>
                      </div>

                      <input
                        type="range"
                        min="0.05"
                        max="0.95"
                        step="0.05"
                        value={
                          confidence
                        }
                        onChange={(
                          e
                        ) =>
                          setConfidence(
                            Number(
                              e.target
                                .value
                            )
                          )
                        }
                        className="w-full accent-[#0055FF]"
                      />
                    </div>
                  )}

                  {/* Active Decision Engine */}
                  <div className="mt-5 border border-[#27272A] p-3">
                    <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#52525B]">
                      Decision Engine
                    </div>

                    <div className="flex items-center justify-between mt-2">
                      <div className="font-mono text-[10px] text-[#8B5CF6]">
                        {engineLabel(engineName)}
                      </div>

                      <div className="font-mono text-[8px] uppercase text-[#3478FF]">
                        {autoRoute ? "AUTO POLICY" : "AVAILABLE"}
                      </div>
                    </div>

                    <div className="font-mono text-[8px] text-[#71717A] mt-2">
                      {autoRoute
                        ? "Active policy determines the Edge, Cloud or Hybrid route."
                        : "Manual routing is active; infrastructure route is user-selected."}
                    </div>
                  </div>
                                    {/* Routing mode */}
                  <div className="mt-5">
                    <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#52525B] mb-2">
                      Routing Mode
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          autoRouteRef.current = true;
                          setAutoRoute(true);
                        }}
                        className={`border px-3 py-2 font-mono text-[9px] uppercase ${
                          autoRoute
                            ? "border-[#00E676] bg-[#00E676]/10 text-[#00E676]"
                            : "border-[#27272A] text-[#71717A] hover:text-white"
                        }`}
                      >
                        Auto
                      </button>

                      <button
                        type="button"
                            
                            onClick={() => {
                              autoRouteRef.current = false;
                          setAutoRoute(false);
                        }}
                        className={`border px-3 py-2 font-mono text-[9px] uppercase ${
                          !autoRoute
                            ? "border-[#FFCC00] bg-[#FFCC00]/10 text-[#FFCC00]"
                            : "border-[#27272A] text-[#71717A] hover:text-white"
                        }`}
                      >
                        Manual
                      </button>
                    </div>
                  </div>

{/* Route selection */}
                  <div className="mt-5">
                    <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#52525B] mb-2">
                      {autoRoute ? "AI-Selected Route" : "Execution Route"}
                    </div>

                    {autoRoute ? (
                      <div className="border border-[#27272A] px-3 py-3">
                        {(() => {
                          const selectedRoute =
                            lastRealInference?.requested_mode === "auto"
                              ? (
                                  lastRealInference?.orchestrated_route ||
                                  lastRealInference?.decision?.route ||
                                  null
                                )
                              : null;

                          return (
                            <div className="flex items-center justify-between">
                              <span className="font-mono text-[8px] text-[#71717A] uppercase">
                                Q-Learning Decision
                              </span>

                              <span
                                className={`font-mono text-[10px] uppercase ${
                                  selectedRoute
                                    ? routeTone(selectedRoute)
                                    : "text-[#52525B]"
                                }`}
                              >
                                {selectedRoute || "Waiting for decision"}
                              </span>
                            </div>
                          );
                        })()}
                      </div>
                    ) : (
                      <div className="grid grid-cols-3 gap-2">
                        {[
                          "edge",
                          "cloud",
                          "hybrid",
                        ].map((route) => (
                          <button
                            key={route}
                            type="button"
                            onClick={() => {
                              autoRouteRef.current = false;
                              setAutoRoute(false);
                              setMode(route);
                            }}
                            className={`border px-3 py-2 font-mono text-[9px] uppercase ${
                              mode === route
                                ? route === "edge"
                                  ? "border-[#00E676] bg-[#00E676]/10 text-[#00E676]"
                                  : route === "cloud"
                                    ? "border-[#0055FF] bg-[#0055FF]/10 text-[#0055FF]"
                                    : "border-[#FFCC00] bg-[#FFCC00]/10 text-[#FFCC00]"
                                : "border-[#27272A] text-[#71717A] hover:text-white"
                            }`}
                          >
                            {route}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                  {/* Current model */}
                  
<div className="mt-5 border border-[#27272A] p-3">
                    <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#52525B]">
                      Current AI Stack
                    </div>

                    <div className="font-mono text-[10px] text-white mt-2">
                      {String(
                        lastRealInference?.orchestrated_route ||
                          mode ||
                          ""
                      ).toLowerCase() === "hybrid"
                        ? "Hybrid Vision Stack"
                        : workloadType === "florence"
                          ? "Florence-2 Base"
                          : "YOLOv8n"}
                    </div>

                    <div className="font-mono text-[8px] text-[#71717A] mt-1">
                      {selectedModel
                        ?.size_mb !=
                      null
                        ? `${selectedModel.size_mb} MB`
                        : "--"}
                    </div>
                  </div>

                  <div className="mt-4 border border-[#27272A] p-4">
                    <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#52525B]">
                      Execution Plan
                    </div>

                    <div className="font-mono text-[9px] text-[#71717A] mt-1">
                      Planned infrastructure path for the current AI workload.
                    </div>

                    {(() => {
                      const activeRoute = String(
                        autoRoute
                          ? (
                              lastRealInference?.requested_mode === "auto"
                                ? (
                                    lastRealInference?.orchestrated_route ||
                                    lastRealInference?.decision?.route ||
                                    ""
                                  )
                                : ""
                            )
                          : mode
                      ).toLowerCase();

                      if (!activeRoute) {
                        return (
                          <div className="mt-4 border border-[#18181B] p-4">
                            <div className="font-mono text-[9px] text-[#71717A]">
                              AUTO routing active.
                            </div>

                            <div className="font-mono text-[10px] text-white mt-2">
                              Waiting for Capture & Analyse
                            </div>

                            <div className="font-mono text-[8px] text-[#52525B] mt-2">
                              The orchestration engine will determine EDGE, CLOUD or HYBRID.
                            </div>
                          </div>
                        );
                      }

                      if (activeRoute === "edge") {
                        return (
                          <div className="mt-4 border border-[#00E676]/30 p-4">
                            <div className="font-mono text-[9px] uppercase text-[#3478FF]">
                              Edge Execution Plan
                            </div>

                            <div className="grid grid-cols-2 gap-2 mt-3 font-mono text-[9px]">
                              <div className="text-[#71717A]">Node</div>
                              <div className="text-white">edge-node-01</div>

                              <div className="text-[#71717A]">Environment</div>
                              <div className="text-white">VirtualBox Edge VM</div>

                              <div className="text-[#71717A]">Platform</div>
                              <div className="text-white">Linux</div>

                              <div className="text-[#71717A]">Workload</div>
                              <div className="text-white">
                                {workloadType === "florence"
                                  ? "Florence-2"
                                  : "YOLOv8n"}
                              </div>
                            </div>

                            <div className="font-mono text-[9px] text-[#00E676] mt-4">
                              EDGE ? LOCAL INFERENCE
                            </div>
                          </div>
                        );
                      }

                      if (activeRoute === "cloud") {
                        return (
                          <div className="mt-4 border border-[#0055FF]/30 p-4">
                            <div className="font-mono text-[9px] uppercase text-[#5B8CFF]">
                              Cloud Execution Plan
                            </div>

                            <div className="grid grid-cols-2 gap-2 mt-3 font-mono text-[9px]">
                              <div className="text-[#71717A]">Node</div>
                              <div className="text-white">cloud-node-01</div>

                              <div className="text-[#71717A]">Environment</div>
                              <div className="text-white">AWS EC2</div>

                              <div className="text-[#71717A]">Platform</div>
                              <div className="text-white">Linux</div>

                              <div className="text-[#71717A]">Workload</div>
                              <div className="text-white">
                                {workloadType === "florence"
                                  ? "Florence-2"
                                  : "YOLOv8n"}
                              </div>
                            </div>

                            <div className="font-mono text-[9px] text-[#5B8CFF] mt-4">
                              CLOUD ? REMOTE INFERENCE
                            </div>
                          </div>
                        );
                      }

                      return (
                        <div className="mt-4 border border-[#FFCC00]/30 p-4">
                          <div className="font-mono text-[9px] uppercase text-[#FFCC00]">
                            Hybrid Execution Plan
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
                            <div className="border border-[#00E676]/20 p-3">
                              <div className="font-mono text-[8px] uppercase text-[#3478FF]">
                                Edge Node
                              </div>

                              <div className="font-mono text-[10px] text-white mt-2">
                                edge-node-01
                              </div>

                              <div className="font-mono text-[9px] text-[#71717A] mt-1">
                                VirtualBox Edge VM / Linux
                              </div>

                              <div className="font-mono text-[9px] text-white mt-3">
                                {workloadType === "florence" ? "Florence-2" : "YOLOv8n"}
                              </div>

                              <div className="font-mono text-[8px] text-[#00E676] mt-1">
                                {workloadType === "florence"
                                  ? "Semantic Scene Understanding"
                                  : "Fast Object Detection"}
                              </div>
                            </div>

                            <div className="border border-[#0055FF]/20 p-3">
                              <div className="font-mono text-[8px] uppercase text-[#5B8CFF]">
                                Cloud Node
                              </div>

                              <div className="font-mono text-[10px] text-white mt-2">
                                cloud-node-01
                              </div>

                              <div className="font-mono text-[9px] text-[#71717A] mt-1">
                                AWS EC2 / Linux
                              </div>

                              <div className="font-mono text-[9px] text-white mt-3">
                                {workloadType === "florence" ? "Florence-2" : "YOLOv8n"}
                              </div>

                              <div className="font-mono text-[8px] text-[#5B8CFF] mt-1">
                                {workloadType === "florence"
                                  ? "Semantic Scene Understanding"
                                  : "Fast Object Detection"}
                              </div>
                            </div>
                          </div>

                          <div className="mt-4 border border-[#27272A] p-3">
                            <div className="font-mono text-[8px] uppercase text-[#71717A]">
                              Cooperative Pipeline
                            </div>

                            <div className="font-mono text-[9px] mt-3">
                              <span className="text-[#00E676]">EDGE</span>
                              <span className="text-[#71717A]">
                                {" -- "}{workloadType === "florence" ? "Florence-2" : "YOLOv8n"}{" -------+"}
                              </span>
                            </div>

                            <div className="font-mono text-[9px] text-[#FFCC00] pl-28 mt-1">
                              +--? COMBINED RESULT
                            </div>

                            <div className="font-mono text-[9px] mt-1">
                              <span className="text-[#5B8CFF]">CLOUD</span>
                              <span className="text-[#71717A]">
                                {" -- "}{workloadType === "florence" ? "Florence-2" : "YOLOv8n"}{" -----+"}
                              </span>
                            </div>
                          </div>

                          <div className="mt-4 border border-[#27272A] p-4">
                            <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#FFCC00]">
                              Result Fusion Logic
                            </div>

                            <div className="grid grid-cols-1 gap-3 mt-4">
                              <div className="border border-[#00E676]/20 p-3">
                                <div className="font-mono text-[8px] uppercase text-[#3478FF]">
                                  1. Edge Execution
                                </div>

                                <div className="font-mono text-[9px] text-[#D4D4D8] leading-5 mt-2">
                                  {workloadType === "florence"
                                    ? "Florence-2 executes the captured frame on the Edge node and returns the Edge inference output."
                                    : "YOLOv8n executes the captured frame on the Edge node and returns object detection results."}
                                </div>
                              </div>

                              <div className="border border-[#0055FF]/20 p-3">
                                <div className="font-mono text-[8px] uppercase text-[#5B8CFF]">
                                  2. Cloud Execution
                                </div>

                                <div className="font-mono text-[9px] text-[#D4D4D8] leading-5 mt-2">
                                  {workloadType === "florence"
                                    ? "Florence-2 executes the same captured frame on the AWS Cloud node and returns the Cloud inference output."
                                    : "YOLOv8n executes the same captured frame on the AWS Cloud node and returns object detection results."}
                                </div>
                              </div>

                              <div className="border border-[#FFCC00]/20 p-3">
                                <div className="font-mono text-[8px] uppercase text-[#FFCC00]">
                                  3. Hybrid Aggregation
                                </div>

                                <div className="font-mono text-[9px] text-[#D4D4D8] leading-5 mt-2">
                                  The orchestration layer collects the real execution outputs from both infrastructures and aggregates them into the hybrid result.
                                </div>
                              </div>
                            </div>

                            <div className="mt-4 border border-[#18181B] p-3">
                              <div className="font-mono text-[8px] uppercase text-[#71717A]">
                                Hybrid Execution Path
                              </div>

                              <div className="font-mono text-[9px] mt-3">
                                <span className="text-[#00E676]">EDGE</span>
                                <span className="text-[#71717A]">
                                  {" ? "}{workloadType === "florence" ? "Florence-2" : "YOLOv8n"}
                                </span>
                              </div>

                              <div className="font-mono text-[9px] mt-2">
                                <span className="text-[#5B8CFF]">CLOUD</span>
                                <span className="text-[#71717A]">
                                  {" ? "}{workloadType === "florence" ? "Florence-2" : "YOLOv8n"}
                                </span>
                              </div>

                              <div className="font-mono text-[9px] text-[#FFCC00] mt-3">
                                EDGE + CLOUD ? HYBRID RESULT
                              </div>
                            </div>

                            <div className="mt-4 border border-[#18181B] p-3">
                              <div className="font-mono text-[8px] uppercase text-[#71717A]">
                                Final Output
                              </div>

                              <div className="font-mono text-[10px] text-white mt-2">
                                {workloadType === "florence"
                                  ? "Aggregated Florence-2 Edge + Cloud inference result"
                                  : "Aggregated YOLOv8n Edge + Cloud inference result"}
                              </div>
                            </div>
                            <div className="mt-4 border border-[#FFCC00]/20 p-3">
                              <div className="font-mono text-[8px] uppercase text-[#FFCC00]">
                                Why This Is Hybrid
                              </div>

                              <div className="font-mono text-[9px] text-[#D4D4D8] leading-5 mt-2">
                                Hybrid routing executes the selected AI workload across both the Edge and AWS Cloud nodes. The orchestration layer collects the execution outputs from both infrastructures and aggregates them into the hybrid result.
                              </div>
                            </div>

                            {lastRealInference?.orchestrated_route === "hybrid" && (
                              <div className="mt-4 border border-[#27272A] p-3">
                                <div className="font-mono text-[8px] uppercase text-[#71717A]">
                                  Current Hybrid Execution
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-3">
                                  <div>
                                    <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                      Edge Output
                                    </div>
                                    <div className="font-mono text-[9px] text-[#00E676] mt-1">
                                      {lastRealInference?.edge_result?.detections_count ??
                                        lastRealInference?.detections_count ??
                                        "--"}{" "}
                                      objects
                                    </div>
                                  </div>

                                  <div>
                                    <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                      Cloud Output
                                    </div>
                                    <div className="font-mono text-[9px] text-[#5B8CFF] mt-1">
                                      Semantic interpretation
                                    </div>
                                  </div>

                                  <div>
                                    <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                      Overall Latency
                                    </div>
                                    <div className="font-mono text-[9px] text-[#FFCC00] mt-1">
                                      {lastRealInference?.latency_ms != null
                                        ? `${Number(lastRealInference.latency_ms).toFixed(1)} ms`
                                        : "--"}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })()}
                  </div>

                  {workloadType === "benchmark" && (
                    <button
                      type="button"
                      onClick={runRealInference}
                      disabled={realInferenceLoading}
                      className="w-full mt-5 bg-[#0055FF] hover:bg-[#0044CC] disabled:opacity-40 px-4 py-3 font-mono text-[10px] uppercase tracking-[0.12em] text-white flex items-center justify-center gap-2"
                    >
                      {realInferenceLoading ? (
                        <>
                          <CircleNotch
                            size={13}
                            className="animate-spin"
                          />
                          Running Benchmark
                        </>
                      ) : (
                        <>
                          <Lightning
                            size={13}
                            weight="fill"
                          />
                          Run Compute Benchmark
                        </>
                      )}
                    </button>
                  )}
                </Cell>

                {/* Latest Inference */}
                <Cell className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Overline>
                        INFERENCE REQUEST | LATEST
                      </Overline>

                      <div className="font-mono text-[9px] text-[#71717A] mt-1">
                        Most recent workload execution result.
                      </div>
                    </div>

                    <span
                      className={`font-mono text-[9px] px-2 py-1 border ${
                        lastRealInference
                          ?.success ===
                        true
                          ? "border-[#00E676]/30 text-[#00E676]"
                          : lastRealInference
                                ?.success ===
                              false
                            ? "border-[#FF3333]/30 text-[#FF3333]"
                            : "border-[#27272A] text-[#71717A]"
                      }`}
                    >
                      {lastRealInference
                        ?.success ===
                      true
                        ? "SUCCESS"
                        : lastRealInference
                              ?.success ===
                            false
                          ? "FAILED"
                          : "WAITING"}
                    </span>
                  </div>

                  {lastRealInference ? (
                    <div className="mt-5">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">

                        <div className="border border-[#27272A] p-3">
                          <div className="font-mono text-[8px] uppercase text-[#52525B]">
                            Workload
                          </div>

                          <div className="font-mono text-sm mt-1 uppercase">
                            {lastRealInference
                              .workload_type ||
                              "--"}
                          </div>
                        </div>

                        <div className="border border-[#27272A] p-3">
                          <div className="font-mono text-[8px] uppercase text-[#52525B]">
                            Route
                          </div>

                          <div
                            className={`font-mono text-sm mt-1 uppercase ${routeTone(
                              lastRealInference
                                .mode ||
                                mode
                            )}`}
                          >
                            {lastRealInference
                              .mode ||
                              mode}
                          </div>
                        </div>

                        <div className="border border-[#27272A] p-3">
                          <div className="font-mono text-[8px] uppercase text-[#52525B]">
                            Latency
                          </div>

                          <div className="font-mono text-sm mt-1">
                            {lastRealInference
                              .latency_ms !=
                            null
                              ? Number(
                                  lastRealInference
                                    .latency_ms
                                ).toFixed(
                                  2
                                )
                              : "--"}
                            <span className="font-mono text-[8px] text-[#71717A] ml-1">
                              ms
                            </span>
                          </div>
                        </div>

                        <div className="border border-[#27272A] p-3">
                          <div className="font-mono text-[8px] uppercase text-[#52525B]">
                            CPU
                          </div>

                          <div className="font-mono text-sm mt-1">
                            {lastRealInference
                              .cpu_percent ??
                              lastRealInference
                                .cpu ??
                              "--"}
                            <span className="font-mono text-[8px] text-[#71717A] ml-1">
                              %
                            </span>
                          </div>
                        </div>
                      </div>

                     {/* Benchmark Compute */}
                      {lastRealInference
                        .workload_type ===
                        "benchmark" && (
                        <div className="mt-3 border border-[#27272A] p-3">
                          <Overline>
REAL COMPUTE BENCHMARK
                          </Overline>

                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
                            <div>
                              <div className="font-mono text-[8px] text-[#52525B] uppercase">
                                FPS
                              </div>

                              <div className="font-mono text-sm mt-1">
                                {lastRealInference
                                  .fps_estimate ??
                                  "--"}
                              </div>
                            </div>

                            <div>
                              <div className="font-mono text-[8px] text-[#52525B] uppercase">
                                Memory
                              </div>

                              <div className="font-mono text-sm mt-1">
                                {lastRealInference
                                  .memory_mb ??
                                  "--"}
                                <span className="text-[8px] text-[#71717A] ml-1">
                                  MB
                                </span>
                              </div>
                            </div>

                            <div>
                              <div className="font-mono text-[8px] text-[#52525B] uppercase">
                                FLOPs
                              </div>

                              <div className="font-mono text-sm mt-1">
                                {lastRealInference
                                  .workload_flops !=
                                null
                                  ? (
                                      Number(
                                        lastRealInference
                                          .workload_flops
                                      ) /
                                      1e6
                                    ).toFixed(
                                      1
                                    )
                                  : "--"}
                                <span className="text-[8px] text-[#71717A] ml-1">
                                  M
                                </span>
                              </div>
                            </div>

                            <div>
                              <div className="font-mono text-[8px] text-[#52525B] uppercase">
                                Payload
                              </div>

                              <div className="font-mono text-sm mt-1">
                                {lastRealInference
                                  .payload_kb ??
                                  "--"}
                                <span className="text-[8px] text-[#71717A] ml-1">
                                  KB
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                      {["yolo", "florence"].includes(
                        lastRealInference.workload_type
                      ) && (() => {
                        const detections = Array.isArray(
                          lastRealInference.detections
                        )
                          ? lastRealInference.detections
                          : [];

                        const vehicleClasses = [
                          "car",
                          "truck",
                          "bus",
                          "motorcycle",
                          "bicycle",
                        ];

                        const peopleCount =
                          detections.filter(
                            (item) =>
                              String(
                                item.class_name ||
                                item.label ||
                                ""
                              ).toLowerCase() === "person"
                          ).length;

                        const vehicleCount =
                          detections.filter(
                            (item) =>
                              vehicleClasses.includes(
                                String(
                                  item.class_name ||
                                  item.label ||
                                  ""
                                ).toLowerCase()
                              )
                          ).length;

                        const otherCount =
                          Math.max(
                            detections.length -
                              peopleCount -
                              vehicleCount,
                            0
                          );

                        const analysisDate = new Date(
                          lastRealInference.analysis_timestamp ||
                          lastRealInference.timestamp ||
                          Date.now()
                        );

                        const dateText =
                          analysisDate
                            .toLocaleDateString(
                              "en-GB",
                              {
                                day: "2-digit",
                                month: "short",
                                year: "numeric",
                              }
                            )
                            .toUpperCase();

                        const timeText =
                          analysisDate
                            .toLocaleTimeString(
                              "en-GB",
                              {
                                hour12: false,
                              }
                            );

                        

                        const objectNames =
                          detections.map(
                            (item) =>
                              String(
                                item.class_name ||
                                item.label ||
                                item.name ||
                                "object"
                              ).toLowerCase()
                          );
                        const frameWidth =
                          Number(lastRealInference.image_width) || 1280;

                        const spatialDescriptions =
                          detections.map((item, index) => {
                            const name = String(
                              item.class_name ||
                                item.label ||
                                item.name ||
                                "object"
                            ).toUpperCase();

                            const box =
                              item.bounding_box || {};

                            const x1 = Number(box.x1 ?? 0);
                            const x2 = Number(box.x2 ?? 0);
                            const centreX = (x1 + x2) / 2;

                            let region = "centre";

                            if (centreX < frameWidth / 3) {
                              region = "left";
                            } else if (
                              centreX > (frameWidth * 2) / 3
                            ) {
                              region = "right";
                            }

                            return `${index + 1}. ${name}: ${region} region of the monitored frame.`;
                          });

                        const validDetections =
                          detections.length;

                        const yoloSceneSummary =
                          detections.length === 0
                            ? `OCCUPANCY
No people, vehicles or other recognised objects were detected in the monitored area.

SPATIAL ANALYSIS
No objects exceeded the configured confidence threshold.

PARKING / VEHICLE ACTIVITY
No vehicle presence or parking-related activity was identified in the current frame.

DETECTION QUALITY
0 valid detections exceeded the configured confidence threshold.
Confidence threshold: ${Number(confidence).toFixed(2)}.

SYSTEM ASSESSMENT
The monitored scene is currently clear of recognised people and vehicle activity.`
                            : `OCCUPANCY
${peopleCount} ${peopleCount === 1 ? "person was" : "people were"} detected within the monitored area.
${vehicleCount} vehicle${vehicleCount === 1 ? "" : "s"} detected.
${otherCount} additional recognised object${otherCount === 1 ? "" : "s"} detected.

SPATIAL ANALYSIS
${spatialDescriptions.join("\n")}

PARKING / VEHICLE ACTIVITY
${
  vehicleCount > 0
    ? `${vehicleCount} vehicle${vehicleCount === 1 ? "" : "s"} detected within the monitored camera field. Vehicle presence is confirmed for this frame.`
    : "No vehicles were detected. No parking-related activity was identified in the current frame."
}

DETECTION QUALITY
${validDetections} valid detection${validDetections === 1 ? "" : "s"} exceeded the configured confidence threshold.
Confidence threshold: ${Number(confidence).toFixed(2)}.
Recognised classes: ${[...new Set(objectNames)].join(", ")}.

SYSTEM ASSESSMENT
${
  vehicleCount > 0 && peopleCount > 0
    ? `The monitored scene indicates simultaneous pedestrian and vehicle presence. ${peopleCount} ${peopleCount === 1 ? "person" : "people"} and ${vehicleCount} vehicle${vehicleCount === 1 ? "" : "s"} were identified in the current analysis.`
    : vehicleCount > 0
      ? `The monitored scene indicates vehicle presence without detected pedestrian activity. ${vehicleCount} vehicle${vehicleCount === 1 ? "" : "s"} were identified.`
      : peopleCount > 0
        ? `The monitored scene indicates human presence without vehicle activity. ${peopleCount} ${peopleCount === 1 ? "person was" : "people were"} identified and no parking occupancy event was detected during this analysis.`
        : `The monitored scene contains ${otherCount} recognised object${otherCount === 1 ? "" : "s"} without detected pedestrian or vehicle activity.`
}`;
                        const florenceDescription =
                          typeof lastRealInference.description === "string"
                            ? lastRealInference.description
                            : lastRealInference.description
                              ? JSON.stringify(
                                  lastRealInference.description
                                )
                              : "No visual assessment was returned.";

                        const sceneSummary =
                          lastRealInference.workload_type === "yolo"
                            ? yoloSceneSummary
                            : `The captured frame was analysed for visual maintenance conditions. ${florenceDescription}`;
return (
                          <div className="mt-3 border border-[#27272A] p-4">
                            <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                              <Overline>
                                RESULT DETAILS
                              </Overline>

                              <div className="font-mono text-[9px] text-[#A1A1AA]">
                                {dateText}
                                {" | "}
                                {timeText}
                              </div>
                            </div>

                            <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mt-4">
                              <div className="border border-[#18181B] p-2">
                                <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                  Model
                                </div>
                                <div className="font-mono text-[9px] text-white mt-1">
                                  {lastRealInference.model_name ||
                                    (lastRealInference.workload_type === "yolo"
                                      ? "YOLOv8"
                                      : "Florence-2")}
                                </div>
                              </div>

                              <div className="border border-[#18181B] p-2">
                                <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                  Route
                                </div>
                                <div className="font-mono text-[9px] text-[#39FF88] mt-1 uppercase">
                                  {lastRealInference.mode || mode}
                                </div>
                              </div>

                              <div className="border border-[#18181B] p-2">
                                <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                  Source
                                </div>
                                <div className="font-mono text-[9px] text-white mt-1 uppercase">
                                  {lastRealInference.source === "pc_camera"
                                    ? "PC CAMERA"
                                    : lastRealInference.source || "--"}
                                </div>
                              </div>

                              <div className="border border-[#18181B] p-2">
                                <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                  Latency
                                </div>
                                <div className="font-mono text-[9px] text-white mt-1">
                                  {lastRealInference.latency_ms != null
                                    ? `${Number(
                                        lastRealInference.latency_ms
                                      ).toFixed(2)} ms`
                                    : "--"}
                                </div>
                              </div>

                              <div className="border border-[#18181B] p-2">
                                <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                  Threshold
                                </div>
                                <div className="font-mono text-[9px] text-white mt-1">
                                  {lastRealInference.confidence_threshold ??
                                    confidence}
                                </div>
                              </div>
                            </div>

                            {lastRealInference.workload_type === "yolo" ? (
                              <>
                                <div className="mt-4 border border-[#18181B] p-3">
                                  <div className="font-mono text-[8px] uppercase text-[#39FF88]">
                                    Analysis Summary
                                  </div>

                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
                                    <div>
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        People
                                      </div>
                                      <div className="font-mono text-sm text-white mt-1">
                                        {peopleCount}
                                      </div>
                                    </div>

                                    <div>
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        Vehicles
                                      </div>
                                      <div className="font-mono text-sm text-white mt-1">
                                        {vehicleCount}
                                      </div>
                                    </div>

                                    <div>
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        Other Objects
                                      </div>
                                      <div className="font-mono text-sm text-white mt-1">
                                        {otherCount}
                                      </div>
                                    </div>

                                    <div>
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        Total Objects
                                      </div>
                                      <div className="font-mono text-sm text-white mt-1">
                                        {detections.length}
                                      </div>
                                    </div>
                                  </div>
                                </div>

                                <div className="mt-4">
                                  <div className="font-mono text-[8px] uppercase text-[#39FF88]">
                                    Detected Objects
                                  </div>

                                  {detections.length > 0 ? (
                                    <div className="mt-3 space-y-3">
                                      {detections.map(
                                        (
                                          detection,
                                          index
                                        ) => {
                                          const box =
                                            detection.bounding_box || {};

                                          const className =
                                            String(
                                              detection.class_name ||
                                              detection.label ||
                                              detection.name ||
                                              "Object"
                                            );

                                          const isVehicle =
                                            vehicleClasses.includes(
                                              className.toLowerCase()
                                            );

                                          return (
                                            <div
                                              key={index}
                                              className="border border-[#18181B] p-3"
                                            >
                                              <div className="flex items-center justify-between">
                                                <div className="font-mono text-[10px] text-white uppercase">
                                                  {index + 1}. {className}
                                                </div>

                                                <div className="font-mono text-[10px] text-[#39FF88]">
                                                  {detection.confidence != null
                                                    ? `${(
                                                        Number(
                                                          detection.confidence
                                                        ) * 100
                                                      ).toFixed(2)}%`
                                                    : "--"}
                                                </div>
                                              </div>

                                              {isVehicle &&
                                                (
                                                  detection.colour ||
                                                  detection.color ||
                                                  detection.make ||
                                                  detection.model ||
                                                  detection.plate
                                                ) && (
                                                  <div className="mt-3 border border-[#18181B] p-2">
                                                    <div className="font-mono text-[7px] uppercase text-[#39FF88] mb-2">
                                                      Vehicle Details
                                                    </div>

                                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 font-mono text-[9px] text-[#D4D4D8]">
                                                      {(detection.colour ||
                                                        detection.color) && (
                                                        <span>
                                                          Colour:{" "}
                                                          {detection.colour ||
                                                            detection.color}
                                                        </span>
                                                      )}

                                                      {detection.make && (
                                                        <span>
                                                          Make:{" "}
                                                          {detection.make}
                                                        </span>
                                                      )}

                                                      {detection.model && (
                                                        <span>
                                                          Model:{" "}
                                                          {detection.model}
                                                        </span>
                                                      )}

                                                      {detection.plate && (
                                                        <span>
                                                          Plate:{" "}
                                                          {detection.plate}
                                                        </span>
                                                      )}
                                                    </div>
                                                  </div>
                                                )}

                                              <div className="mt-3">
                                                <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                                  Object Location / Bounding Box
                                                </div>

                                                <div className="grid grid-cols-1 md:grid-cols-4 gap-2 mt-2">
                                                  <div className="border border-[#18181B] p-2">
                                                    <div className="font-mono text-[7px] text-[#52525B]">
                                                      Left edge (x1)
                                                    </div>
                                                    <div className="font-mono text-[9px] text-white mt-1">
                                                      {box.x1 != null
                                                        ? `${Number(
                                                            box.x1
                                                          ).toFixed(1)} px`
                                                        : "--"}
                                                    </div>
                                                  </div>

                                                  <div className="border border-[#18181B] p-2">
                                                    <div className="font-mono text-[7px] text-[#52525B]">
                                                      Top edge (y1)
                                                    </div>
                                                    <div className="font-mono text-[9px] text-white mt-1">
                                                      {box.y1 != null
                                                        ? `${Number(
                                                            box.y1
                                                          ).toFixed(1)} px`
                                                        : "--"}
                                                    </div>
                                                  </div>

                                                  <div className="border border-[#18181B] p-2">
                                                    <div className="font-mono text-[7px] text-[#52525B]">
                                                      Right edge (x2)
                                                    </div>
                                                    <div className="font-mono text-[9px] text-white mt-1">
                                                      {box.x2 != null
                                                        ? `${Number(
                                                            box.x2
                                                          ).toFixed(1)} px`
                                                        : "--"}
                                                    </div>
                                                  </div>

                                                  <div className="border border-[#18181B] p-2">
                                                    <div className="font-mono text-[7px] text-[#52525B]">
                                                      Bottom edge (y2)
                                                    </div>
                                                    <div className="font-mono text-[9px] text-white mt-1">
                                                      {box.y2 != null
                                                        ? `${Number(
                                                            box.y2
                                                          ).toFixed(1)} px`
                                                        : "--"}
                                                    </div>
                                                  </div>
                                                </div>
                                              </div>
                                            </div>
                                          );
                                        }
                                      )}
                                    </div>
                                  ) : (
                                    <div className="font-mono text-[9px] text-[#52525B] mt-3">
                                      No objects detected above the selected confidence threshold.
                                    </div>
                                  )}
                                </div>
                              </>
                            ) : (
                              <div className="mt-4 border border-[#18181B] p-3">
                                <div className="font-mono text-[8px] uppercase text-[#39FF88]">
                                  Florence-2 Visual Assessment
                                </div>

                                <div className="font-mono text-[10px] text-[#D4D4D8] leading-5 whitespace-pre-wrap mt-3">
                                  {typeof lastRealInference.description ===
                                  "string"
                                    ? lastRealInference.description
                                    : JSON.stringify(
                                        lastRealInference.description,
                                        null,
                                        2
                                      )}
                                </div>
                              </div>
                            )}
                            {lastRealInference.requested_mode === "auto" &&
                              lastRealInference.decision && (
                                <div className="mt-4 border border-[#8B5CF6]/30 bg-[#8B5CF6]/5 p-3">
                                  <div className="flex items-center justify-between gap-3">
                                    <div>
                                      <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#A78BFA]">
                                        AI Orchestration Decision
                                      </div>

                                      <div className="font-mono text-[13px] text-white mt-1 uppercase">
                                        AUTO
                                        {" ? "}
                                        {String(
                                          lastRealInference.orchestrated_route ||
                                            lastRealInference.decision?.route ||
                                            "--"
                                        ).toUpperCase()}
                                      </div>
                                    </div>

                                    <div className="text-right">
                                      <div className="font-mono text-[7px] uppercase text-[#71717A]">
                                        Confidence
                                      </div>

                                      <div className="font-mono text-[13px] text-[#A78BFA] mt-1">
                                        {lastRealInference.decision?.confidence != null
                                          ? `${(
                                              Number(
                                                lastRealInference.decision.confidence
                                              ) * 100
                                            ).toFixed(1)}%`
                                          : "--"}
                                      </div>
                                    </div>
                                  </div>

                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-3">
                                    <div className="border border-[#27272A] p-2">
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        Requested
                                      </div>
                                      <div className="font-mono text-[10px] text-white mt-1 uppercase">
                                        {lastRealInference.requested_mode || "--"}
                                      </div>
                                    </div>

                                    <div className="border border-[#27272A] p-2">
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        Selected Route
                                      </div>
                                      <div className="font-mono text-[10px] text-[#39FF88] mt-1 uppercase">
                                        {lastRealInference.orchestrated_route ||
                                          lastRealInference.decision?.route ||
                                          "--"}
                                      </div>
                                    </div>

                                    <div className="border border-[#27272A] p-2">
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        Engine
                                      </div>
                                      <div className="font-mono text-[10px] text-white mt-1">
                                        {lastRealInference.decision?.engine || "--"}
                                      </div>
                                    </div>

                                    <div className="border border-[#27272A] p-2">
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        Node
                                      </div>
                                      <div className="font-mono text-[10px] text-white mt-1">
                                        {lastRealInference.node || "--"}
                                      </div>
                                    </div>

                                    <div className="border border-[#27272A] p-2">
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        Environment
                                      </div>

                                      <div className="font-mono text-[10px] text-white mt-1 uppercase">
                                        {String(
                                          lastRealInference.orchestrated_route ||
                                            lastRealInference.execution ||
                                            lastRealInference.mode ||
                                            ""
                                        ).toLowerCase() === "hybrid"
                                          ? "VirtualBox Edge VM + AWS EC2"
                                          : String(
                                                lastRealInference.orchestrated_route ||
                                                  lastRealInference.execution ||
                                                  lastRealInference.mode ||
                                                  ""
                                              ).toLowerCase() === "cloud"
                                            ? "AWS EC2"
                                            : "VirtualBox Edge VM"}
                                      </div>
                                    </div>

                                    <div className="border border-[#27272A] p-2">
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        Platform
                                      </div>

                                      <div className="font-mono text-[10px] text-white mt-1">
                                        {String(
                                          lastRealInference.orchestrated_route ||
                                            lastRealInference.execution ||
                                            lastRealInference.mode ||
                                            ""
                                        ).toLowerCase() === "hybrid"
                                          ? `${
                                              lastRealInference.edge_result?.platform ||
                                              "--"
                                            } + ${
                                              lastRealInference.cloud_result?.platform ||
                                              "--"
                                            }`
                                          : lastRealInference.platform || "--"}
                                      </div>
                                    </div>

                                    <div className="border border-[#27272A] p-2">
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        Workload
                                      </div>

                                      <div className="font-mono text-[10px] text-white mt-1">
                                        {String(
                                          lastRealInference.orchestrated_route ||
                                            lastRealInference.execution ||
                                            lastRealInference.mode ||
                                            ""
                                        ).toLowerCase() === "hybrid"
                                          ? `${
                                              lastRealInference.edge_result?.model_name ||
                                              "YOLO"
                                            } + ${
                                              lastRealInference.cloud_result?.model_name ||
                                              "Florence-2"
                                            }`
                                          : lastRealInference.model_name ||
                                            lastRealInference.workload_type ||
                                            "--"}
                                      </div>
                                    </div>
                                  </div>

                                  <div className="mt-3 border border-[#27272A] p-3">
                                    <div className="font-mono text-[7px] uppercase text-[#71717A]">
                                      Why this route?
                                    </div>

                                    <div className="font-mono text-[9px] leading-5 text-[#D4D4D8] mt-2">
                                      {lastRealInference.decision?.reason ||
                                        "No decision explanation returned."}
                                    </div>
                                  </div>

                                  <div className="mt-3">
                                    <div className="font-mono text-[7px] uppercase text-[#71717A] mb-2">
                                      Route Probabilities
                                    </div>

                                    <div className="grid grid-cols-2 gap-2">
                                      {["edge", "cloud", "hybrid"].map((route) => {
                                        const value = Number(
                                          lastRealInference.decision?.probabilities?.[
                                            route
                                          ] || 0
                                        );

                                        return (
                                          <div
                                            key={route}
                                            className="border border-[#27272A] p-2"
                                          >
                                            <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                              {route}
                                            </div>

                                            <div className="font-mono text-[11px] text-white mt-1">
                                              {(value * 100).toFixed(1)}%
                                            </div>
                                          </div>
                                        );
                                      })}
                                    </div>
                                  </div>
                                </div>
                              )}


                                                        {String(
                              lastRealInference.orchestrated_route ||
                                lastRealInference.execution ||
                                lastRealInference.mode ||
                                ""
                            ).toLowerCase() === "hybrid" && (
                              <div className="mt-4 border border-[#FFCC00]/30 bg-[#FFCC00]/5 p-4">

                                <div className="flex items-start justify-between gap-4">
                                  <div>
                                    <div className="font-mono text-[9px] uppercase tracking-[0.14em] text-[#FFCC00]">
                                      Hybrid Execution Summary
                                    </div>

                                    <div className="font-mono text-[13px] text-white mt-1">
                                      Cooperative Edge–Cloud AI Inference
                                    </div>
                                  </div>

                                  <div className="font-mono text-[9px] text-[#FFCC00] border border-[#FFCC00]/30 px-3 py-2">
                                    PARALLEL / COOPERATIVE
                                  </div>
                                </div>


                                <div className="mt-4 border border-[#27272A] p-3">
                                  <div className="font-mono text-[7px] uppercase text-[#71717A]">
                                    Execution Strategy
                                  </div>

                                  <div className="font-mono text-[10px] text-white mt-2">
                                    Parallel Edge + Cloud Processing
                                  </div>

                                  <div className="font-mono text-[9px] text-[#A1A1AA] mt-2 leading-5">
                                    The selected AI workload is executed across both the Edge and Cloud
                                    infrastructures. The orchestration layer collects the outputs from
                                    both executions and aggregates them into the hybrid result.
                                  </div>
                                </div>


                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mt-3">

                                  <div className="border border-[#00E676]/30 p-3">
                                    <div className="font-mono text-[9px] uppercase text-[#3478FF]">
                                      Edge Execution
                                    </div>

                                    <div className="grid grid-cols-2 gap-2 mt-3 font-mono text-[9px]">
                                      <div className="text-[#71717A]">Role</div>
                                      <div className="text-white">Fast Object Detection</div>

                                      <div className="text-[#71717A]">Model</div>
                                      <div className="text-white">
                                        {lastRealInference.edge_result?.model_name || "YOLOv8n"}
                                      </div>

                                      <div className="text-[#71717A]">Node</div>
                                      <div className="text-white">
                                        {lastRealInference.edge_result?.node || "edge-node-01"}
                                      </div>

                                      <div className="text-[#71717A]">Environment</div>
                                      <div className="text-white">VirtualBox Edge VM</div>

                                      <div className="text-[#71717A]">Platform</div>
                                      <div className="text-white">
                                        {lastRealInference.edge_result?.platform || "--"}
                                      </div>

                                      <div className="text-[#71717A]">Latency</div>
                                      <div className="text-[#00E676]">
                                        {lastRealInference.edge_latency_ms != null
                                          ? `${Number(lastRealInference.edge_latency_ms).toFixed(1)} ms`
                                          : "--"}
                                      </div>

                                      <div className="text-[#71717A]">Output</div>
                                      <div className="text-white">
                                        {lastRealInference.edge_result?.detections_count ??
                                          lastRealInference.detections_count ??
                                          0}{" "}
                                        objects detected
                                      </div>
                                    </div>
                                  </div>


                                  <div className="border border-[#0055FF]/30 p-3">
                                    <div className="font-mono text-[9px] uppercase text-[#5B8CFF]">
                                      Cloud Execution
                                    </div>

                                    <div className="grid grid-cols-2 gap-2 mt-3 font-mono text-[9px]">
                                      <div className="text-[#71717A]">Role</div>
                                      <div className="text-white">
                                        Semantic Scene Understanding
                                      </div>

                                      <div className="text-[#71717A]">Model</div>
                                      <div className="text-white">
                                        {lastRealInference.cloud_result?.model_name || "Florence-2"}
                                      </div>

                                      <div className="text-[#71717A]">Node</div>
                                      <div className="text-white">
                                        {lastRealInference.cloud_result?.node || "cloud-node-01"}
                                      </div>

                                      <div className="text-[#71717A]">Environment</div>
                                      <div className="text-white">AWS EC2</div>

                                      <div className="text-[#71717A]">Platform</div>
                                      <div className="text-white">
                                        {lastRealInference.cloud_result?.platform || "--"}
                                      </div>

                                      <div className="text-[#71717A]">Latency</div>
                                      <div className="text-[#5B8CFF]">
                                        {lastRealInference.cloud_latency_ms != null
                                          ? `${Number(lastRealInference.cloud_latency_ms).toFixed(1)} ms`
                                          : "--"}
                                      </div>

                                      <div className="text-[#71717A]">Output</div>
                                      <div className="text-white">
                                        Detailed visual scene interpretation
                                      </div>
                                    </div>
                                  </div>

                                </div>


                                <div className="mt-3 border border-[#27272A] p-3">
                                  <div className="font-mono text-[7px] uppercase text-[#71717A]">
                                    Workload Distribution
                                  </div>

                                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-3">
                                    <div className="border border-[#00E676]/20 p-2 font-mono text-[9px]">
                                      <span className="text-white">YOLOv8n</span>
                                      <span className="text-[#71717A]"> → </span>
                                      <span className="text-[#00E676]">EDGE</span>
                                    </div>

                                    <div className="border border-[#0055FF]/20 p-2 font-mono text-[9px]">
                                      <span className="text-white">Florence-2</span>
                                      <span className="text-[#71717A]"> → </span>
                                      <span className="text-[#5B8CFF]">CLOUD</span>
                                    </div>
                                  </div>
                                </div>


                                <div className="mt-3 border border-[#27272A] p-3">
                                  <div className="font-mono text-[7px] uppercase text-[#FFCC00]">
                                    Why Hybrid?
                                  </div>

                                  <div className="font-mono text-[9px] leading-5 text-[#D4D4D8] mt-2">
                                    {lastRealInference.requested_mode === "auto"
                                      ? lastRealInference.decision?.reason ||
                                        "The AI orchestration engine selected cooperative Edge–Cloud execution based on the current workload and infrastructure conditions."
                                      : "Hybrid execution was manually requested by the user. The workload was distributed across the Edge and AWS Cloud environments using the configured cooperative execution strategy."}
                                  </div>
                                </div>


                                <div className="mt-3 border border-[#27272A] p-3">
                                  <div className="font-mono text-[7px] uppercase text-[#71717A]">
                                    Combined AI Result
                                  </div>

                                  <div className="font-mono text-[9px] text-white mt-2">
                                    Edge Detection:
                                    {" "}
                                    {lastRealInference.edge_result?.detections_count ??
                                      lastRealInference.detections_count ??
                                      0}
                                    {" "}objects identified by YOLOv8n
                                  </div>

                                  <div className="font-mono text-[9px] text-[#A1A1AA] leading-5 mt-2">
                                    Cloud Understanding:{" "}
                                    {typeof lastRealInference.cloud_result?.description === "string"
                                      ? lastRealInference.cloud_result.description
                                      : typeof lastRealInference.description === "string"
                                        ? lastRealInference.description
                                        : "Florence-2 semantic analysis completed."}
                                  </div>
                                </div>


                                <div className="mt-3">
                                  <div className="font-mono text-[7px] uppercase text-[#71717A] mb-2">
                                    Performance
                                  </div>

                                  <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                                    <div className="border border-[#27272A] p-2">
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        Edge Latency
                                      </div>
                                      <div className="font-mono text-[11px] text-[#00E676] mt-1">
                                        {lastRealInference.edge_latency_ms != null
                                          ? `${Number(lastRealInference.edge_latency_ms).toFixed(1)} ms`
                                          : "--"}
                                      </div>
                                    </div>

                                    <div className="border border-[#27272A] p-2">
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        Cloud Latency
                                      </div>
                                      <div className="font-mono text-[11px] text-[#5B8CFF] mt-1">
                                        {lastRealInference.cloud_latency_ms != null
                                          ? `${Number(lastRealInference.cloud_latency_ms).toFixed(1)} ms`
                                          : "--"}
                                      </div>
                                    </div>

                                    <div className="border border-[#27272A] p-2">
                                      <div className="font-mono text-[7px] uppercase text-[#52525B]">
                                        Overall Hybrid Latency
                                      </div>
                                      <div className="font-mono text-[11px] text-[#FFCC00] mt-1">
                                        {lastRealInference.latency_ms != null
                                          ? `${Number(lastRealInference.latency_ms).toFixed(1)} ms`
                                          : "--"}
                                      </div>
                                    </div>
                                  </div>
                                </div>


                                <div className="mt-3 border border-[#FFCC00]/20 p-3">
                                  <div className="font-mono text-[7px] uppercase text-[#71717A]">
                                    Execution Pattern
                                  </div>

                                  <div className="font-mono text-[9px] mt-3 space-y-2">
                                    <div>
                                      <span className="text-[#00E676]">EDGE</span>
                                      <span className="text-[#71717A]"> -- YOLOv8n -------+</span>
                                    </div>

                                    <div className="text-[#FFCC00] pl-28">
                                      +--? COMBINED RESULT
                                    </div>

                                    <div>
                                      <span className="text-[#5B8CFF]">CLOUD</span>
                                      <span className="text-[#71717A]"> -- Florence-2 -----+</span>
                                    </div>
                                  </div>
                                </div>

                              </div>
                            )}
<div className="mt-4 border-t border-[#27272A] pt-4">
                              <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#39FF88]">
                                Scene Summary
                              </div>

                              <div className="mt-3 border border-[#18181B] p-3">
                                <p className="font-mono text-[10px] leading-5 text-[#D4D4D8] whitespace-pre-wrap">
                                  {sceneSummary}
                                </p>
                              </div>
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  ) : (
                    <div className="font-mono text-[10px] text-[#52525B] text-center py-16">
                      No real workload has been executed yet.
                    </div>
                  )}
                </Cell>
              </div>
            </section>

            {/* =================================================
                SYSTEM LOGS
            ================================================== */}

            
            <section
              id="routing-section"
              className="scroll-mt-24"
            >
              <Cell className="p-4 md:p-5">
                <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">

                  <div>
                    <Overline>
                      LIVE ROUTING RESULT
                    </Overline>

                    <div className="flex flex-wrap items-center gap-3 mt-3">
                      <span
                        className={`font-mono text-xl uppercase ${routeTone(
                          currentPolicyRoute
                        )}`}
                      >
                        {currentPolicyRoute}
                      </span>

                      <span className="text-[#52525B]">
                        ->
                      </span>

                      <span
                        className={`font-mono text-xl uppercase ${routeTone(
                          currentExecutedRoute
                        )}`}
                      >
                        {currentExecutedRoute}
                      </span>

                      <span
                        className={`font-mono text-[9px] px-2 py-1 border ${
                          executionStatus ===
                          "SUCCESS"
                            ? "text-[#00E676] border-[#00E676]/30"
                            : executionStatus ===
                                "FAILED"
                              ? "text-[#FF3333] border-[#FF3333]/30"
                              : "text-[#71717A] border-[#27272A]"
                        }`}
                      >
                        {executionStatus}
                      </span>
                    </div>

                    <div className="font-mono text-[10px] text-[#71717A] mt-3 max-w-4xl">
                      {liveDecision
                        ?.reason ||
                        lastDecision
                          ?.reason ||
                        "Waiting for a live orchestration decision..."}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 min-w-[360px]">

                    <div className="border border-[#27272A] px-3 py-2">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        Latency
                      </div>

                      <div className="font-mono text-sm mt-1">
                        {formatMetric(
                          liveExecution
                            ?.latency_ms,
                          2
                        )}
                        <span className="text-[8px] text-[#71717A] ml-1">
                          ms
                        </span>
                      </div>
                    </div>

                    <div className="border border-[#27272A] px-3 py-2">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        Failover
                      </div>

                      <div
                        className={`font-mono text-sm mt-1 ${
                          failoverApplied
                            ? "text-[#FFCC00]"
                            : "text-[#00E676]"
                        }`}
                      >
                        {failoverApplied
                          ? "YES"
                          : "NO"}
                      </div>
                    </div>

                    <div className="border border-[#27272A] px-3 py-2">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        Batch
                      </div>

                      <div className="font-mono text-sm mt-1">
                        {liveDecision
                          ?.context
                          ?.batch_size ??
                          1}
                      </div>
                    </div>

                    <div className="border border-[#27272A] px-3 py-2">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        Policy
                      </div>

                      <div className="font-mono text-[10px] mt-1">
                        {liveDecision
                          ?.engine ||
                          engineFullName(
                            engineName
                          )}
                      </div>
                    </div>
                  </div>
                </div>

                {failoverApplied &&
                  failoverReason && (
                    <div className="mt-4 border border-[#FFCC00]/30 bg-[#FFCC00]/5 px-3 py-2 font-mono text-[10px] text-[#FFCC00]">
                      FAILOVER:{" "}
                      {failoverReason}
                    </div>
                  )}
              </Cell>
            </section>

            {/* =================================================
                DECISION EXPLANATION (XAI)
            ================================================== */}

            <section className="border border-[#1F1F22] bg-[#08090B] p-4">
              <div className="flex items-start justify-between gap-4 mb-4">
                <div>
                  <div className="font-mono text-[9px] uppercase tracking-[0.14em] text-[#8B5CF6]">
                    Decision Explanation (XAI)
                  </div>

                  <div className="text-[8px] text-[#596579] mt-1">
                    Why the intelligent policy selected the current execution route
                  </div>
                </div>

                <div className="flex items-center gap-5 text-right">

                  <div>
                    <div className="font-mono text-[7px] uppercase text-[#71717A]">
                      AI Selected Route
                    </div>

                    <div className={`font-mono text-lg mt-1 ${routeTone(xaiRoute)}`}>
                      {xaiRoute}
                    </div>
                  </div>

                  <div className="font-mono text-[#52525B] text-lg">
                    →
                  </div>

                  <div>
                    <div className="font-mono text-[7px] uppercase text-[#71717A]">
                      Executed Route
                    </div>

                    <div className={`font-mono text-lg mt-1 ${routeTone(currentExecutedRoute)}`}>
                      {currentExecutedRoute}
                    </div>
                  </div>

                  <div>
                    <div className="font-mono text-[7px] uppercase text-[#71717A]">
                      Failover
                    </div>

                    <div
                      className={`font-mono text-[10px] mt-1 ${
                        failoverApplied
                          ? "text-[#FACC15]"
                          : "text-[#A3E635]"
                      }`}
                    >
                      {failoverApplied ? "APPLIED" : "NO"}
                    </div>
                  </div>

                </div>
              </div>

              <div className="grid grid-cols-[1.15fr_1fr_0.55fr] gap-3">

                {/* KEY FACTORS */}
                <div className="border border-[#1F1F22] p-3">
                  <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#71717A] mb-3">
                    Key Factors
                  </div>

                  <div className="space-y-2 font-mono text-[8px]">

                    <div className="grid grid-cols-[1.2fr_0.8fr_0.8fr] gap-2">
                      <span className="text-[#A1A1AA]">Network Latency</span>
                      <span>{xaiNetworkLatency.toFixed(1)} ms</span>
                      <span className={xaiNetworkLatency > 150 ? "text-[#FF6B6B]" : xaiNetworkLatency > 80 ? "text-[#FACC15]" : "text-[#A3E635]"}>
                        {xaiNetworkLatency > 150 ? "HIGH" : xaiNetworkLatency > 80 ? "MEDIUM" : "LOW"}
                      </span>
                    </div>

                    <div className="grid grid-cols-[1.2fr_0.8fr_0.8fr] gap-2">
                      <span className="text-[#A1A1AA]">Edge CPU</span>
                      <span>{xaiEdgeCpu.toFixed(1)} %</span>
                      <span className={xaiEdgeCpu < 40 ? "text-[#A3E635]" : xaiEdgeCpu < 75 ? "text-[#FACC15]" : "text-[#FF6B6B]"}>
                        {xaiEdgeCpu < 40 ? "LOW" : xaiEdgeCpu < 75 ? "MEDIUM" : "HIGH"}
                      </span>
                    </div>

                    <div className="grid grid-cols-[1.2fr_0.8fr_0.8fr] gap-2">
                      <span className="text-[#A1A1AA]">Memory Available</span>
                      <span>{xaiMemoryAvailable.toFixed(1)} %</span>
                      <span className="text-[#A3E635]">NORMAL</span>
                    </div>

                    <div className="grid grid-cols-[1.2fr_0.8fr_0.8fr] gap-2">
                      <span className="text-[#A1A1AA]">Connectivity</span>
                      <span>{xaiConnectivity === 1 ? "ONLINE" : "OFFLINE"}</span>
                      <span className={xaiConnectivity === 1 ? "text-[#A3E635]" : "text-[#FF6B6B]"}>
                        {xaiConnectivity === 1 ? "POSITIVE" : "NEGATIVE"}
                      </span>
                    </div>

                    <div className="grid grid-cols-[1.2fr_0.8fr_0.8fr] gap-2">
                      <span className="text-[#A1A1AA]">Priority</span>
                      <span>{xaiPriority}</span>
                      <span className="text-[#FACC15]">
                        {xaiPriority >= 4 ? "HIGH" : xaiPriority >= 2 ? "MEDIUM" : "LOW"}
                      </span>
                    </div>

                    <div className="grid grid-cols-[1.2fr_0.8fr_0.8fr] gap-2">
                      <span className="text-[#A1A1AA]">Cost Budget</span>
                      <span>${xaiCostBudget.toFixed(2)}</span>
                      <span className="text-[#A3E635]">WITHIN LIMIT</span>
                    </div>

                  </div>
                </div>

                {/* WHY */}
                <div className="border border-[#1F1F22] p-3">
                  <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#71717A] mb-3">
                    Why {xaiRoute}?
                  </div>

                  <p className="text-[9px] leading-5 text-[#A1A1AA]">
                    {xaiWhyText}
                  </p>

                  <div className="mt-4 pt-3 border-t border-[#1F1F22]">
                    <div className="font-mono text-[7px] uppercase text-[#596579]">
                      Decision Engine
                    </div>

                    <div className="font-mono text-[10px] text-[#8B5CF6] mt-1">
                      {engineLabel(engineName)}
                    </div>
                  </div>
                </div>

                {/* CONFIDENCE */}
                <div className="border border-[#1F1F22] p-3 flex flex-col justify-between">
                  <div>
                    <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#71717A]">
                      Confidence
                    </div>

                    <div className="font-mono text-2xl text-white mt-3">
                      {xaiConfidence.toFixed(1)}
                      <span className="text-[9px] text-[#71717A] ml-1">%</span>
                    </div>
                  </div>

                  <div className="mt-4">
                    <div className="h-[3px] bg-[#151515]">
                      <div
                        className="h-full bg-[#8B5CF6]"
                        style={{ width: `${Math.min(100, Math.max(0, xaiConfidence))}%` }}
                      />
                    </div>

                    <div className="font-mono text-[7px] text-[#71717A] mt-2">
                      Policy certainty
                    </div>
                  </div>
                </div>

              </div>
            </section>


            {/* =================================================
                KPI CARDS
            ================================================== */}

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <KPI
                label="Execution Latency"
                value={formatMetric(
                  liveExecution
                    ?.latency_ms,
                  2
                )}
                unit="ms"
                tone="good"
              />

              <KPI
                label="AWS RTT"
                value={formatMetric(
                  cloudRtt,
                  2
                )}
                unit="ms"
                tone="accent"
              />

              <KPI
                label="Edge CPU"
                value={formatMetric(
                  edgeCpu,
                  1
                )}
                unit="%"
                tone="warn"
              />

              <KPI
                label="Decision Time"
                value={formatMetric(
                  liveDecision
                    ?.decision_latency_us,
                  1
                )}
                unit="us"
                tone="default"
              />
            </div>
                        {/* =================================================
                LIVE PERFORMANCE CHARTS
            ================================================== */}

            <section
              id="telemetry-section"
              className="scroll-mt-24"
            >
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-3">

                {/* Execution Latency */}
                <Cell className="p-4 xl:col-span-2">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <Overline>
                        EXECUTION LATENCY
                      </Overline>

                      <div className="font-mono text-[9px] text-[#52525B] mt-1">
                        Real Edge vs Cloud execution | milliseconds | timestamps in UTC
                      </div>
                    </div>

                    <div className="flex items-center gap-4 font-mono text-[9px]">
                      <span className="flex items-center gap-1 text-[#00E676]">
                        <span className="w-2 h-2 bg-[#00E676]" />
                        EDGE
                      </span>

                      <span className="flex items-center gap-1 text-[#0055FF]">
                        <span className="w-2 h-2 bg-[#0055FF]" />
                        CLOUD
                      </span>
                    </div>
                  </div>

                  <div className="h-[260px]">
                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                    >
                      <LineChart
                        data={
                          dashboardLatencyData
                        }
                        margin={{
                          top: 10,
                          right: 15,
                          left: -15,
                          bottom: 0,
                        }}
                      >
                        <CartesianGrid
                          stroke="#18181B"
                          vertical={false}
                        />

                        <XAxis
                          dataKey="t"
                          stroke="#52525B"
                          tick={{
                            fontFamily:
                              "JetBrains Mono",
                            fontSize: 9,
                          }}
                          tickLine={false}
                          axisLine={{
                            stroke:
                              "#27272A",
                          }}
                        />

                        <YAxis
                          stroke="#52525B"
                          tick={{
                            fontFamily:
                              "JetBrains Mono",
                            fontSize: 9,
                          }}
                          tickLine={false}
                          axisLine={false}
                        />

                        <Tooltip
                          contentStyle={{
                            background:
                              "#050505",
                            border:
                              "1px solid #27272A",
                            borderRadius:
                              0,
                            fontFamily:
                              "JetBrains Mono",
                            fontSize:
                              10,
                          }}
                        />

                        <Line
                          type="monotone"
                          dataKey="edge_latency"
                          name="Edge"
                          stroke="#00E676"
                          strokeWidth={2}
                          dot={false}
                          connectNulls
                          isAnimationActive={
                            false
                          }
                        />

                        <Line
                          type="monotone"
                          dataKey="cloud_latency"
                          name="Cloud"
                          stroke="#0055FF"
                          strokeWidth={2}
                          dot={false}
                          connectNulls
                          isAnimationActive={
                            false
                          }
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Cell>

                {/* Resource Utilisation */}
                <Cell className="p-4">
                  <div>
                    <Overline>
                      RESOURCE UTILISATION
                    </Overline>

                    <div className="font-mono text-[9px] text-[#52525B] mt-1">
                      Current infrastructure load
                    </div>
                  </div>

                  <div className="h-[260px] mt-4">
                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                    >
                      <BarChart
                        data={
                          utilisationData
                        }
                        layout="vertical"
                        margin={{
                          top: 0,
                          right: 10,
                          left: 10,
                          bottom: 0,
                        }}
                      >
                        <CartesianGrid
                          stroke="#18181B"
                          horizontal={false}
                        />

                        <XAxis
                          type="number"
                          domain={[
                            0,
                            100,
                          ]}
                          stroke="#52525B"
                          tick={{
                            fontFamily:
                              "JetBrains Mono",
                            fontSize: 8,
                          }}
                        />

                        <YAxis
                          type="category"
                          dataKey="name"
                          width={82}
                          stroke="#52525B"
                          tick={{
                            fontFamily:
                              "JetBrains Mono",
                            fontSize: 8,
                          }}
                        />

                        <Tooltip
                          contentStyle={{
                            background:
                              "#050505",
                            border:
                              "1px solid #27272A",
                            borderRadius:
                              0,
                            fontFamily:
                              "JetBrains Mono",
                            fontSize:
                              10,
                          }}
                          cursor={false}
                        />

                        <Bar
                          dataKey="value"
                          fill="#0055FF"
                          barSize={13}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Cell>
              </div>

              {/* Route Distribution */}
              <div className="grid grid-cols-1 lg:grid-cols-[0.8fr_1.2fr] gap-3 mt-3">
                <Cell className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Overline>
                        DECISIONS BY ROUTE — EDGE • CLOUD • HYBRID
                      </Overline>

                      <div className="font-mono text-[9px] text-[#52525B] mt-1">
                        Selected policy | last 30 minutes
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="font-mono text-[7px] uppercase text-[#71717A]">
                        Total Decisions
                      </div>

                      <div className="font-mono text-lg text-white mt-1">
                        {totalDistribution}
                      </div>

                      <div className="font-mono text-[7px] text-[#52525B] mt-1 max-w-[180px]">
                        Routing decisions made by the orchestration policy
                      </div>
                    </div>
                  </div>

                  <div className="h-[230px] mt-2">
                    <ResponsiveContainer
                      width="100%"
                      height="100%"
                    >
                      <PieChart>
                        <Pie
                          data={
                            distributionData
                          }
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={50}
                          outerRadius={82}
                          paddingAngle={3}
                        >
                          {distributionData.map(
                            (item) => (
                              <RCell
                                key={
                                  item.name
                                }
                                fill={
                                  ROUTE_COLORS[
                                    item.name
                                  ]
                                }
                              />
                            )
                          )}
                        </Pie>

                        <Tooltip
                          contentStyle={{
                            background: "#050505",
                            border: "1px solid #27272A",
                            borderRadius: 0,
                            fontFamily: "JetBrains Mono",
                            fontSize: 10,
                            color: "#FFFFFF",
                          }}
                          itemStyle={{
                            color: "#FFFFFF",
                            fontFamily: "JetBrains Mono",
                            fontSize: 10,
                          }}
                          labelStyle={{
                            color: "#A1A1AA",
                            fontFamily: "JetBrains Mono",
                            fontSize: 9,
                          }}
                          cursor={false}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                    {distributionData.map((item) => {
                      const route = String(item.name || "").toLowerCase();

                      const explanation =
                        route === "edge"
                          ? "Routing Decisions Selected Edge Execution"
                          : route === "cloud"
                            ? "Routing Decisions Selected Cloud Execution"
                            : "Routing Decisions Selected Cooperative Edge–Cloud Execution";

                      return (
                        <div
                          key={item.name}
                          className="border border-[#27272A] p-3"
                        >
                          <div
                            className="font-mono text-[8px] uppercase"
                            style={{
                              color: ROUTE_COLORS[item.name],
                            }}
                          >
                            {item.name}
                          </div>

                          <div className="font-mono text-lg mt-1">
                            {item.value}
                          </div>

                          <div className="font-mono text-[7px] text-[#71717A] leading-4 mt-2">
                            {explanation}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="mt-3 border border-[#27272A] p-3">
                    <div className="font-mono text-[7px] uppercase text-[#71717A]">
                      Routing Summary
                    </div>

                    <div className="font-mono text-[9px] text-white mt-2">
                      {totalDistribution} Total Routing Decisions:{" "}
                      {distributionData
                        .map(
                          (item) =>
                            `${item.value} ${String(item.name).toUpperCase()}`
                        )
                        .join(" + ")}
                    </div>
                  </div>
                </Cell>

                {/* Latest Infrastructure Snapshot */}
                <Cell className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Overline>
                        INFRASTRUCTURE SNAPSHOT
                      </Overline>

                      <div className="font-mono text-[9px] text-[#52525B] mt-1">
                        Live telemetry used by the decision policy
                      </div>
                    </div>

                    <Broadcast
                      size={16}
                      className="text-[#00E676]"
                    />
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-5">
                    <div className="border border-[#27272A] p-3">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        Edge CPU
                      </div>

                      <div className="font-mono text-xl text-[#00E676] mt-1">
                        {formatMetric(
                          edgeCpu,
                          1
                        )}
                        <span className="text-[9px] text-[#71717A] ml-1">
                          %
                        </span>
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-3">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        Edge RAM
                      </div>

                      <div className="font-mono text-xl mt-1">
                        {formatMetric(
                          edgeMemory,
                          1
                        )}
                        <span className="text-[9px] text-[#71717A] ml-1">
                          %
                        </span>
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-3">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        Cloud CPU
                      </div>

                      <div className="font-mono text-xl text-[#0055FF] mt-1">
                        {formatMetric(
                          cloudCpu,
                          1
                        )}
                        <span className="text-[9px] text-[#71717A] ml-1">
                          %
                        </span>
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-3">
                      <div className="font-mono text-[8px] uppercase text-[#52525B]">
                        AWS RTT
                      </div>

                      <div className="font-mono text-xl mt-1">
                        {formatMetric(
                          cloudRtt,
                          1
                        )}
                        <span className="text-[9px] text-[#71717A] ml-1">
                          ms
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2">
                    <div className="border border-[#27272A] p-3 font-mono text-[9px]">
                      <div className="text-[#52525B] uppercase">
                        Edge
                      </div>

                      <div className="mt-2 text-[#A1A1AA]">
                        CPU cores:{" "}
                        <span className="text-white">
                          {edgeInfra
                            ?.cpu_count ??
                            "--"}
                        </span>
                      </div>

                      <div className="mt-1 text-[#A1A1AA]">
                        Memory total:{" "}
                        <span className="text-white">
                          {formatMetric(
                            edgeInfra
                              ?.memory_total_gb,
                            2
                          )}{" "}
                          GB
                        </span>
                      </div>

                      <div className="mt-1 text-[#A1A1AA]">
                        Disk free:{" "}
                        <span className="text-white">
                          {formatMetric(
                            edgeInfra
                              ?.disk_free_gb,
                            2
                          )}{" "}
                          GB
                        </span>
                      </div>
                    </div>

                    <div className="border border-[#27272A] p-3 font-mono text-[9px]">
                      <div className="text-[#52525B] uppercase">
                        Cloud
                      </div>

                      <div className="mt-2 text-[#A1A1AA]">
                        CPU cores:{" "}
                        <span className="text-white">
                          {cloudInfra
                            ?.cpu_count ??
                            "--"}
                        </span>
                      </div>

                      <div className="mt-1 text-[#A1A1AA]">
                        Memory total:{" "}
                        <span className="text-white">
                          {formatMetric(
                            cloudInfra
                              ?.memory_total_gb,
                            2
                          )}{" "}
                          GB
                        </span>
                      </div>

                      <div className="mt-1 text-[#A1A1AA]">
                        Disk free:{" "}
                        <span className="text-white">
                          {formatMetric(
                            cloudInfra
                              ?.disk_free_gb,
                            2
                          )}{" "}
                          GB
                        </span>
                      </div>
                    </div>
                  </div>
                </Cell>
              </div>
            </section>

            {/* =================================================
                DECISION ENGINE
            ================================================== */}

            
            {/* Infrastructure overview */}
              <div className="grid grid-cols-1 xl:grid-cols-[1fr_42px_0.9fr_42px_1fr] gap-2 items-stretch min-w-0">

                {/* EDGE COMPUTE */}
                <Cell className="relative overflow-hidden bg-[#090A0C] border-[#24272C]">
                  <div className="h-[3px] bg-[#35D07F]" />

                  <div className="p-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full border border-[#235A3C] bg-[#0A1C13] flex items-center justify-center">
                          <Cpu
                            size={20}
                            className="text-[#56E39A]"
                            weight="duotone"
                          />
                        </div>

                        <div>
                          <div className="font-mono text-[9px] tracking-[0.12em] text-[#56E39A] uppercase">
                            Edge Compute
                          </div>

                          <div className="font-mono text-[12px] text-white mt-1">
                            {edgeInfra?.node || "edge-node-01"}
                          </div>

                          <div className="font-mono text-[8px] text-[#7C8B82] mt-1">
                            {edgeInfra?.platform || "Linux"} | Local inference node
                          </div>
                        </div>
                      </div>

                      <div
                        className={`flex items-center gap-1.5 font-mono text-[8px] px-2 py-1 border rounded-sm ${
                          edgeConnected
                            ? "border-[#246B43] bg-[#0B2115] text-[#6DE8A4]"
                            : "border-[#6B3030] bg-[#211010] text-[#FF6B6B]"
                        }`}
                      >
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            edgeConnected
                              ? "bg-[#57E389]"
                              : "bg-[#FF5252]"
                          }`}
                        />

                        {edgeConnected ? "ONLINE" : "OFFLINE"}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-x-4 gap-y-2 mt-3">
                      <div>
                        <div className="flex justify-between font-mono text-[8px] text-[#78847D]">
                          <span>CPU LOAD</span>
                          <span className="text-white">
                            {formatMetric(edgeCpu)}%
                          </span>
                        </div>

                        <div className="h-1 bg-[#122219] mt-1.5 overflow-hidden">
                          <div
                            className="h-full bg-[#35D07F]"
                            style={{
                              width: `${Math.min(
                                100,
                                Math.max(0, Number(edgeCpu || 0))
                              )}%`,
                            }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between font-mono text-[8px] text-[#78847D]">
                          <span>MEMORY</span>
                          <span className="text-white">
                            {formatMetric(edgeMemory)}%
                          </span>
                        </div>

                        <div className="h-1 bg-[#122219] mt-1.5 overflow-hidden">
                          <div
                            className="h-full bg-[#35D07F]"
                            style={{
                              width: `${Math.min(
                                100,
                                Math.max(0, Number(edgeMemory || 0))
                              )}%`,
                            }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="font-mono text-[8px] text-[#78847D]">
                          EDGE RTT
                        </div>

                        <div className="font-mono text-[12px] text-white mt-1">
                          {formatMetric(edgeRtt)}
                          <span className="text-[8px] text-[#78847D] ml-1">
                            ms
                          </span>
                        </div>
                      </div>

                      <div>
                        <div className="font-mono text-[8px] text-[#78847D]">
                          EXECUTION
                        </div>

                        <div className="font-mono text-[10px] text-[#56E39A] mt-1">
                          LOCAL / REAL
                        </div>
                      </div>
                    </div>
                  </div>
                </Cell>

                {/* ENGINE -> EDGE */}
                <div className="hidden lg:flex items-center justify-center">
                  <div className="flex items-center w-full">
                    
                    <span className="font-mono text-[#57E389] text-lg px-1">
                      &lt;-
                    </span>
                  </div>
                </div>

                {/* DECISION ENGINE */}
                <Cell className="relative overflow-hidden bg-[#090A0C] border-[#34301F]">
                  <div className="h-[3px] bg-[#F2C94C]" />

                  <div className="p-3">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full border border-[#62531C] bg-[#211C0B] flex items-center justify-center">
                          <Brain
                            size={20}
                            className="text-[#F2C94C]"
                            weight="duotone"
                          />
                        </div>

                        <div>
                          <div className="font-mono text-[9px] tracking-[0.12em] text-[#F2C94C] uppercase">
                            Decision Engine
                          </div>

                          <div className="font-mono text-[12px] text-white mt-1">
                            {engineLabel(engineName)}
                          </div>

                          <div className="font-mono text-[8px] text-[#8D8566] mt-1">
                            Intelligent routing policy
                          </div>
                        </div>
                      </div>

                      <span
                        className={`w-2 h-2 rounded-full mt-1 ${
                          activePolicyAvailable
                            ? "bg-[#57E389]"
                            : "bg-[#FF5252]"
                        }`}
                      />
                    </div>

                    <div className="mt-3 border border-[#302D20] bg-[#0B0B0D] p-2.5">
                      <div className="font-mono text-[8px] text-[#8D8566] uppercase">
                        Active Route
                      </div>

                      <div
                        className={`font-mono text-xl uppercase mt-1 ${routeTone(
                          currentPolicyRoute
                        )}`}
                      >
                        {currentPolicyRoute}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 mt-2">
                      <div className="border border-[#302A14] p-2">
                        <div className="font-mono text-[7px] text-[#8D8566] uppercase">
                          Confidence
                        </div>

                        <div className="font-mono text-[11px] text-white mt-1">
                          {liveDecision?.confidence != null
                            ? `${(
                                Number(liveDecision.confidence) * 100
                              ).toFixed(1)}%`
                            : "--"}
                        </div>
                      </div>

                      <div className="border border-[#302A14] p-2">
                        <div className="font-mono text-[7px] text-[#8D8566] uppercase">
                          Decision Time
                        </div>

                        <div className="font-mono text-[11px] text-white mt-1">
                          {formatMetric(
                            liveDecision?.decision_latency_us,
                            1
                          )}
                          <span className="text-[7px] text-[#8D8566] ml-1">
                            us
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between mt-3 font-mono text-[8px]">
                      <span className="text-[#56E39A]">EDGE</span>
                      <span className="text-[#596579]">|</span>
                      <span className="text-[#F2C94C]">HYBRID</span>
                      <span className="text-[#596579]">|</span>
                      <span className="text-[#5B8CFF]">CLOUD</span>
                    </div>
                  </div>
                </Cell>

                {/* ENGINE -> CLOUD */}
                <div className="hidden lg:flex items-center justify-center">
                  <div className="flex items-center w-full">
                    
                    <span className="font-mono text-[#5B8CFF] text-lg px-1">
                      ->
                    </span>
                  </div>
                </div>

                {/* AWS CLOUD */}
                <Cell className="relative overflow-hidden bg-[#090A0C] border-[#242A36]">
                  <div className="h-[3px] bg-[#4D7CFE]" />

                  <div className="p-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full border border-[#254B82] bg-[#0A1930] flex items-center justify-center">
                          <CloudArrowUp
                            size={20}
                            className="text-[#6D9CFF]"
                            weight="duotone"
                          />
                        </div>

                        <div>
                          <div className="flex items-center gap-2">
                            <div className="font-mono text-[9px] tracking-[0.12em] text-[#6D9CFF] uppercase">
                              AWS Cloud
                            </div>

                            <span className="font-mono text-[7px] px-1.5 py-0.5 border border-[#6D4B1D] text-[#FFB84D] bg-[#201407]">
                              AWS
                            </span>
                          </div>

                          <div className="font-mono text-[12px] text-white mt-1">
                            {cloudInfra?.node || "cloud-node-01"}
                          </div>

                          <div className="font-mono text-[8px] text-[#78849A] mt-1">
                            EC2 | eu-west-2 | Remote execution
                          </div>
                        </div>
                      </div>

                      <div
                        className={`flex items-center gap-1.5 font-mono text-[8px] px-2 py-1 border rounded-sm ${
                          cloudConnected
                            ? "border-[#27588B] bg-[#0B192A] text-[#7CABFF]"
                            : "border-[#6B3030] bg-[#211010] text-[#FF6B6B]"
                        }`}
                      >
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            cloudConnected
                              ? "bg-[#57E389]"
                              : "bg-[#FF5252]"
                          }`}
                        />

                        {cloudConnected ? "ONLINE" : "OFFLINE"}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-x-4 gap-y-2 mt-3">
                      <div>
                        <div className="flex justify-between font-mono text-[8px] text-[#78849A]">
                          <span>CPU LOAD</span>
                          <span className="text-white">
                            {formatMetric(cloudCpu)}%
                          </span>
                        </div>

                        <div className="h-1 bg-[#101D32] mt-1.5 overflow-hidden">
                          <div
                            className="h-full bg-[#4D7CFE]"
                            style={{
                              width: `${Math.min(
                                100,
                                Math.max(0, Number(cloudCpu || 0))
                              )}%`,
                            }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between font-mono text-[8px] text-[#78849A]">
                          <span>MEMORY</span>
                          <span className="text-white">
                            {formatMetric(cloudMemory)}%
                          </span>
                        </div>

                        <div className="h-1 bg-[#101D32] mt-1.5 overflow-hidden">
                          <div
                            className="h-full bg-[#4D7CFE]"
                            style={{
                              width: `${Math.min(
                                100,
                                Math.max(0, Number(cloudMemory || 0))
                              )}%`,
                            }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="font-mono text-[8px] text-[#78849A]">
                          AWS RTT
                        </div>

                        <div className="font-mono text-[12px] text-white mt-1">
                          {formatMetric(cloudRtt)}
                          <span className="text-[8px] text-[#78849A] ml-1">
                            ms
                          </span>
                        </div>
                      </div>

                      <div>
                        <div className="font-mono text-[8px] text-[#78849A]">
                          EXECUTION
                        </div>

                        <div className="font-mono text-[10px] text-[#6D9CFF] mt-1">
                          REMOTE / REAL
                        </div>
                      </div>
                    </div>
                  </div>
                </Cell>
              </div>
            </section>
            {/* =================================================
                MODELS + DEPLOYMENTS
            ================================================== */}

            <section className="scroll-mt-24">
              <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-3 mb-3">
                <div>
                  <Overline>
                    MODEL MANAGEMENT
                  </Overline>

                  <h2 className="font-mono text-lg mt-1">
                    AI Models & Edge–Cloud Deployment
                  </h2>

                  <p className="font-mono text-[9px] text-[#71717A] mt-1">
                    Manage AI models and deployment runtimes across the real Edge–Cloud orchestration infrastructure.
                  </p>
                </div>

                
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">

                {/* DEPLOYMENT OVERVIEW */}
                <Cell className="xl:col-span-2">
                  <div className="px-4 py-3 border-b border-[#27272A] flex items-center justify-between">
                    <div>
                      <Overline>DEPLOYMENT OVERVIEW</Overline>
                      <div className="font-mono text-[9px] text-[#71717A] mt-1">
                        Real Edge and Cloud deployment infrastructure
                      </div>
                    </div>

                    <TreeStructure
                      size={16}
                      weight="duotone"
                      className="text-[#FFCC00]"
                    />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-[#27272A]">

                    <div className="bg-[#0A0A0A] p-4">
                      <div className="flex items-center gap-2">
                        <Cpu
                          size={15}
                          weight="duotone"
                          className="text-[#00E676]"
                        />
                        <div className="font-mono text-[7px] uppercase tracking-[0.1em] text-[#71717A]">
                          Edge Deployment
                        </div>
                      </div>

                      <div className="flex items-center gap-2 mt-3">
                        <CheckCircle
                          size={14}
                          weight="fill"
                          className="text-[#00E676]"
                        />
                        <span className="font-mono text-[11px] text-[#00E676]">
                          COMPLETED
                        </span>
                      </div>

                      <div className="font-mono text-[8px] text-[#52525B] mt-1">
                        AWS IoT Greengrass
                      </div>
                    </div>

                    <div className="bg-[#0A0A0A] p-4">
                      <div className="flex items-center gap-2">
                        <Cloud
                          size={15}
                          weight="duotone"
                          className="text-[#5B8CFF]"
                        />
                        <div className="font-mono text-[7px] uppercase tracking-[0.1em] text-[#71717A]">
                          Cloud Service
                        </div>
                      </div>

                      <div className="flex items-center gap-2 mt-3">
                        <CheckCircle
                          size={14}
                          weight="fill"
                          className="text-[#5B8CFF]"
                        />
                        <span className="font-mono text-[11px] text-[#5B8CFF]">
                          CONFIGURED
                        </span>
                      </div>

                      <div className="font-mono text-[8px] text-[#52525B] mt-1">
                        AWS EC2 / systemd
                      </div>
                    </div>

                    <div className="bg-[#0A0A0A] p-4">
                      <div className="flex items-center gap-2">
                        <MapPin
                          size={15}
                          weight="duotone"
                          className="text-[#FFCC00]"
                        />
                        <div className="font-mono text-[7px] uppercase tracking-[0.1em] text-[#71717A]">
                          Deployment Targets
                        </div>
                      </div>

                      <div className="font-mono text-[8px] text-[#00E676] mt-3">
                        EDGE · edge-node-01-core
                      </div>

                      <div className="font-mono text-[8px] text-[#5B8CFF] mt-1">
                        CLOUD · cloud-node-01
                      </div>
                    </div>

                  </div>
                </Cell>

                {/* REAL DEPLOYMENT STATUS */}
                <Cell className="col-span-2">
                  <div className="px-4 py-3 border-b border-[#27272A] flex items-center justify-between">
                    <div>
                      <Overline>REAL DEPLOYMENT STATUS</Overline>
                      <div className="font-mono text-[9px] text-[#71717A] mt-1">
                        Verified Edge and Cloud runtime deployment
                      </div>
                    </div>

                    <Rocket
                      size={16}
                      weight="duotone"
                      className="text-[#FFCC00]"
                    />
                  </div>

                  <div className="p-4 grid grid-cols-1 xl:grid-cols-2 gap-3">

                    {/* EDGE */}
                    <div className="border border-[#1F3A29] bg-[#08110B] p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex items-center gap-3">
                          <div className="w-9 h-9 border border-[#245B38] flex items-center justify-center">
                            <Cpu
                              size={20}
                              weight="duotone"
                              className="text-[#00E676]"
                            />
                          </div>

                          <div>
                            <div className="font-mono text-[10px] font-semibold text-[#00E676]">
                              EDGE DEPLOYMENT
                            </div>
                            <div className="font-mono text-[8px] text-[#71717A] mt-1">
                              AWS IoT Greengrass
                            </div>
                          </div>
                        </div>

                        <div className="text-right space-y-2">

                          <div>
                            <div className="font-mono text-[7px] uppercase text-[#52525B]">
                              Deployment State
                            </div>

                            <div className="flex items-center justify-end gap-1.5 mt-1">
                              <CheckCircle
                                size={11}
                                weight="fill"
                                className="text-[#00E676]"
                              />

                              <span className="font-mono text-[8px] text-[#00E676]">
                                COMPLETED
                              </span>
                            </div>
                          </div>

                          <div>
                            <div className="font-mono text-[7px] uppercase text-[#52525B]">
                              Runtime State
                            </div>

                            <div
                              className={`font-mono text-[8px] mt-1 ${
                                edgeConnected
                                  ? "text-[#00E676]"
                                  : "text-[#FF5252]"
                              }`}
                            >
                              {edgeConnected ? "ONLINE" : "OFFLINE"}
                            </div>
                          </div>

                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-2 mt-4">
                        <div className="border border-[#18251D] p-2">
                          <div className="font-mono text-[7px] uppercase text-[#52525B]">
                            Component
                          </div>
                          <div className="font-mono text-[8px] text-[#A1A1AA] mt-1 break-all">
                            com.edgecloud.InferenceAgent
                          </div>
                        </div>

                        <div className="border border-[#18251D] p-2">
                          <div className="font-mono text-[7px] uppercase text-[#52525B]">
                            Version
                          </div>
                          <div className="font-mono text-[8px] text-white mt-1">
                            1.1.2
                          </div>
                        </div>

                        <div className="border border-[#18251D] p-2">
                          <div className="font-mono text-[7px] uppercase text-[#52525B]">
                            Target
                          </div>
                          <div className="font-mono text-[8px] text-[#00E676] mt-1">
                            edge-node-01-core
                          </div>
                        </div>

                        <div className="border border-[#18251D] p-2">
                          <div className="font-mono text-[7px] uppercase text-[#52525B]">
                            Platform
                          </div>
                          <div className="font-mono text-[8px] text-[#A1A1AA] mt-1">
                            Linux
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-[1fr_18px_1fr_18px_1fr] items-center mt-4">
                        <div className="border border-[#245B38] px-2 py-2 text-center">
                          <Database
                            size={14}
                            className="text-[#00E676] mx-auto"
                          />
                          <div className="font-mono text-[7px] text-[#A1A1AA] mt-1">
                            S3 ARTIFACT
                          </div>
                        </div>

                        <div className="font-mono text-[12px] text-[#00E676] text-center">
                          →
                        </div>

                        <div className="border border-[#245B38] px-2 py-2 text-center">
                          <TreeStructure
                            size={14}
                            className="text-[#00E676] mx-auto"
                          />
                          <div className="font-mono text-[7px] text-[#A1A1AA] mt-1">
                            GREENGRASS
                          </div>
                        </div>

                        <div className="font-mono text-[12px] text-[#00E676] text-center">
                          →
                        </div>

                        <div className="border border-[#245B38] px-2 py-2 text-center">
                          <Cpu
                            size={14}
                            className="text-[#00E676] mx-auto"
                          />
                          <div className="font-mono text-[7px] text-[#A1A1AA] mt-1">
                            EDGE RUNTIME
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* CLOUD */}
                    <div className="border border-[#1D2C55] bg-[#080C16] p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex items-center gap-3">
                          <div className="w-9 h-9 border border-[#294A8A] flex items-center justify-center">
                            <Cloud
                              size={20}
                              weight="duotone"
                              className="text-[#5B8CFF]"
                            />
                          </div>

                          <div>
                            <div className="font-mono text-[10px] font-semibold text-[#5B8CFF]">
                              CLOUD DEPLOYMENT
                            </div>
                            <div className="font-mono text-[8px] text-[#71717A] mt-1">
                              AWS EC2
                            </div>
                          </div>
                        </div>

                        <div className="text-right space-y-2">

                          <div>
                            <div className="font-mono text-[7px] uppercase text-[#52525B]">
                              Deployment State
                            </div>

                            <div className="font-mono text-[8px] text-[#5B8CFF] mt-1">
                              CONFIGURED
                            </div>
                          </div>

                          <div>
                            <div className="font-mono text-[7px] uppercase text-[#52525B]">
                              Runtime State
                            </div>

                            <div
                              className={`font-mono text-[8px] mt-1 ${
                                cloudConnected
                                  ? "text-[#00E676]"
                                  : "text-[#FF5252]"
                              }`}
                            >
                              {cloudConnected ? "ONLINE" : "OFFLINE"}
                            </div>
                          </div>

                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-2 mt-4">
                        <div className="border border-[#172039] p-2">
                          <div className="font-mono text-[7px] uppercase text-[#52525B]">
                            Target
                          </div>
                          <div className="font-mono text-[8px] text-[#5B8CFF] mt-1">
                            cloud-node-01
                          </div>
                        </div>

                        <div className="border border-[#172039] p-2">
                          <div className="font-mono text-[7px] uppercase text-[#52525B]">
                            Service
                          </div>
                          <div className="font-mono text-[8px] text-[#A1A1AA] mt-1">
                            cloud-node.service
                          </div>
                        </div>

                        <div className="border border-[#172039] p-2">
                          <div className="font-mono text-[7px] uppercase text-[#52525B]">
                            Runtime
                          </div>
                          <div className="font-mono text-[8px] text-[#A1A1AA] mt-1">
                            FastAPI / Uvicorn
                          </div>
                        </div>

                        <div className="border border-[#172039] p-2">
                          <div className="font-mono text-[7px] uppercase text-[#52525B]">
                            Port
                          </div>
                          <div className="font-mono text-[8px] text-white mt-1">
                            8000
                          </div>
                        </div>

                        <div className="border border-[#172039] p-2">
                          <div className="font-mono text-[7px] uppercase text-[#52525B]">
                            Startup
                          </div>
                          <div className="font-mono text-[8px] text-[#5B8CFF] mt-1">
                            ENABLED
                          </div>
                        </div>

                        <div className="border border-[#172039] p-2">
                          <div className="font-mono text-[7px] uppercase text-[#52525B]">
                            Recovery
                          </div>
                          <div className="font-mono text-[8px] text-[#A1A1AA] mt-1">
                            RESTART ALWAYS
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-[1fr_18px_1fr_18px_1fr] items-center mt-4">
                        <div className="border border-[#294A8A] px-2 py-2 text-center">
                          <Cloud
                            size={14}
                            className="text-[#5B8CFF] mx-auto"
                          />
                          <div className="font-mono text-[7px] text-[#A1A1AA] mt-1">
                            AWS EC2
                          </div>
                        </div>

                        <div className="font-mono text-[12px] text-[#5B8CFF] text-center">
                          →
                        </div>

                        <div className="border border-[#294A8A] px-2 py-2 text-center">
                          <HardDrives
                            size={14}
                            className="text-[#5B8CFF] mx-auto"
                          />
                          <div className="font-mono text-[7px] text-[#A1A1AA] mt-1">
                            SYSTEMD
                          </div>
                        </div>

                        <div className="font-mono text-[12px] text-[#5B8CFF] text-center">
                          →
                        </div>

                        <div className="border border-[#294A8A] px-2 py-2 text-center">
                          <Lightning
                            size={14}
                            className="text-[#5B8CFF] mx-auto"
                          />
                          <div className="font-mono text-[7px] text-[#A1A1AA] mt-1">
                            CLOUD RUNTIME
                          </div>
                        </div>
                      </div>
                    </div>

                  </div>
                </Cell>
              </div>

              {/* ROLE IN EDGE-CLOUD ORCHESTRATION */}
              <Cell className="mt-3">
                <div className="px-4 py-3 border-b border-[#27272A] flex items-center justify-between">
                  <div>
                    <Overline>
                      ROLE IN EDGE–CLOUD ORCHESTRATION
                    </Overline>

                    <div className="font-mono text-[9px] text-[#71717A] mt-1">
                      Deployed runtimes provide the execution environments selected by the orchestration policy.
                    </div>
                  </div>

                  <Brain
                    size={17}
                    weight="duotone"
                    className="text-[#FFCC00]"
                  />
                </div>

                <div className="p-4">

                  {/* EXECUTION RUNTIMES */}
                  <div className="grid grid-cols-1 md:grid-cols-[1fr_70px_1fr] gap-3 items-center">

                    {/* EDGE RUNTIME */}
                    <div className="border border-[#245B38] bg-[#08110B] p-4">
                      <div className="flex items-center gap-3">

                        <Cpu
                          size={22}
                          weight="duotone"
                          className="text-[#00E676]"
                        />

                        <div>
                          <div className="font-mono text-[9px] font-semibold text-[#00E676]">
                            EDGE RUNTIME
                          </div>

                          <div className="font-mono text-[8px] text-[#71717A] mt-1">
                            AWS IoT Greengrass · edge-node-01-core
                          </div>
                        </div>

                      </div>
                    </div>

                    {/* RUNTIME CONNECTION */}
                    <div className="hidden md:flex items-center justify-center">
                      <div className="font-mono text-[16px] text-[#756A98]">
                        ↔
                      </div>
                    </div>

                    {/* CLOUD RUNTIME */}
                    <div className="border border-[#294A8A] bg-[#080C16] p-4">
                      <div className="flex items-center gap-3">

                        <Cloud
                          size={22}
                          weight="duotone"
                          className="text-[#5B8CFF]"
                        />

                        <div>
                          <div className="font-mono text-[9px] font-semibold text-[#5B8CFF]">
                            CLOUD RUNTIME
                          </div>

                          <div className="font-mono text-[8px] text-[#71717A] mt-1">
                            AWS EC2 · cloud-node-01
                          </div>
                        </div>

                      </div>
                    </div>

                  </div>


                  {/* DOWNSTREAM FLOW */}
                  <div className="flex flex-col items-center mt-3">

                    <div className="font-mono text-[14px] text-[#756A98]">
                      ↓
                    </div>


                    {/* AI DECISION ENGINE */}
                    <div className="w-full max-w-5xl border border-[#665B2B] bg-[#151309] px-4 py-3">
                      <div className="flex items-center justify-center gap-3">

                        <Brain
                          size={19}
                          weight="duotone"
                          className="text-[#FFCC00]"
                        />

                        <div className="text-center">
                          <div className="font-mono text-[9px] font-semibold text-[#FFCC00]">
                            AI DECISION ENGINE
                          </div>

                          <div className="font-mono text-[7px] text-[#71717A] mt-1">
                            Evaluates workload and infrastructure context
                          </div>
                        </div>

                      </div>
                    </div>


                    <div className="font-mono text-[14px] text-[#756A98] mt-1">
                      ↓
                    </div>


                    {/* ROUTING POLICY */}
                    <div className="grid grid-cols-3 gap-2 w-full max-w-5xl">

                      <div className="border border-[#245B38] bg-[#08110B] px-3 py-2 text-center">
                        <div className="font-mono text-[8px] font-semibold text-[#00E676]">
                          EDGE
                        </div>

                        <div className="font-mono text-[7px] text-[#52525B] mt-1">
                          Local execution
                        </div>
                      </div>

                      <div className="border border-[#665B2B] bg-[#151309] px-3 py-2 text-center">
                        <div className="font-mono text-[8px] font-semibold text-[#FFCC00]">
                          HYBRID
                        </div>

                        <div className="font-mono text-[7px] text-[#52525B] mt-1">
                          Cooperative execution
                        </div>
                      </div>

                      <div className="border border-[#294A8A] bg-[#080C16] px-3 py-2 text-center">
                        <div className="font-mono text-[8px] font-semibold text-[#5B8CFF]">
                          CLOUD
                        </div>

                        <div className="font-mono text-[7px] text-[#52525B] mt-1">
                          Remote execution
                        </div>
                      </div>

                    </div>


                    <div className="font-mono text-[14px] text-[#756A98] mt-1">
                      ↓
                    </div>


                    {/* EXECUTION OUTPUT */}
                    <div className="grid grid-cols-1 md:grid-cols-[1fr_28px_1fr] gap-2 w-full max-w-5xl items-center">

                      <div className="border border-[#27272A] bg-[#0D0D0D] px-3 py-3 text-center">

                        <Lightning
                          size={15}
                          weight="duotone"
                          className="text-[#A1A1AA] mx-auto"
                        />

                        <div className="font-mono text-[8px] text-white mt-1">
                          AI INFERENCE
                        </div>

                      </div>


                      <div className="hidden md:block font-mono text-[12px] text-[#756A98] text-center">
                        →
                      </div>


                      <div className="border border-[#27272A] bg-[#0D0D0D] px-3 py-3 text-center">

                        <ChartLineUp
                          size={15}
                          weight="duotone"
                          className="text-[#A1A1AA] mx-auto"
                        />

                        <div className="font-mono text-[8px] text-white mt-1">
                          RESULT & TELEMETRY
                        </div>

                      </div>

                    </div>

                  </div>


                  {/* EXPLANATION */}
                  <div className="border-t border-[#18181B] mt-4 pt-3">
                    <div className="font-mono text-[8px] leading-relaxed text-[#71717A]">
                      Edge and Cloud deployments provide persistent execution runtimes.
                      The orchestration policy selects Edge, Cloud, or Hybrid execution
                      according to workload and infrastructure conditions.
                    </div>
                  </div>

                </div>
              </Cell>
            </section>

                        
            <section id="ai-assisted-section" className="scroll-mt-24">
              <Cell className="p-4">

                {/* HEADER */}
                <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-3">
                  <div>
                    <Overline>
                      AI-ASSISTED DECISION SUPPORT
                    </Overline>

                    <h2 className="font-mono text-base mt-1">
                      Edge–Cloud Orchestration Analysis & Performance Evaluation
                    </h2>

                    <p className="font-mono text-[9px] text-[#71717A] mt-1 max-w-4xl">
                      AI-assisted interpretation of orchestration decisions,
                      experimental results, policy behaviour and Edge–Cloud
                      system performance.
                    </p>
                  </div>

                  <span className="font-mono text-[8px] uppercase border border-[#1F4D32] text-[#00E676] px-2 py-1">
                    AI SUPPORT · LIVE
                  </span>
                </div>


                {/* QUICK ACTIONS + AI ASSISTANT */}
                <div className="grid grid-cols-1 lg:grid-cols-[0.9fr_1.1fr] gap-3 mt-4">

                  {/* QUICK ACTIONS */}
                  <div className="border border-[#1E3157] bg-[#080B12] p-4">

                    <div className="flex items-center gap-2 mb-4">
                      <Lightning
                        size={17}
                        weight="duotone"
                        className="text-[#3478FF]"
                      />

                      <div className="font-mono text-[9px] uppercase tracking-[0.14em] text-[#8BAEFF]">
                        Quick Actions
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">

                      {/* ANALYSE RESULTS */}
                      <button
                        type="button"
                        disabled={assistantLoading}
                        onClick={() =>
                          askAssistant(
                            "Analyse the latest experimental results and identify the most important performance findings."
                          )
                        }
                        className="group min-h-[84px] border border-[#245FD8] bg-[#09111F] hover:border-[#3478FF] px-4 py-3 flex items-center gap-3 text-left transition-colors disabled:opacity-40"
                      >
                        <ChartLineUp
                          size={24}
                          weight="duotone"
                          className="text-[#3478FF] shrink-0"
                        />

                        <div className="min-w-0 flex-1">
                          <div className="font-mono text-[10px] text-white">
                            Analyse Results
                          </div>

                          <div className="font-mono text-[7px] text-[#6B7B9A] mt-1">
                            Performance interpretation
                          </div>
                        </div>
                      </button>


                      {/* EXPLAIN LAST DECISION */}
                      <button
                        type="button"
                        disabled={assistantLoading}
                        onClick={() =>
                          askAssistant(
                            "Explain the latest routing decision, including the selected execution target and the factors that influenced the decision."
                          )
                        }
                        className="group min-h-[84px] border border-[#1E3157] bg-[#080D16] hover:border-[#3478FF] px-4 py-3 flex items-center gap-3 text-left transition-colors disabled:opacity-40"
                      >
                        <TreeStructure
                          size={24}
                          weight="duotone"
                          className="text-[#3478FF] shrink-0"
                        />

                        <div className="min-w-0 flex-1">
                          <div className="font-mono text-[10px] text-white">
                            Explain Last Decision
                          </div>

                          <div className="font-mono text-[7px] text-[#6B7B9A] mt-1">
                            Routing rationale
                          </div>
                        </div>
                      </button>


                      {/* COMPARE POLICIES */}
                      <button
                        type="button"
                        disabled={assistantLoading}
                        onClick={() =>
                          askAssistant(
                            "Explain the current Q-Learning routing preferences."
                          )
                        }
                        className="group min-h-[84px] border border-[#1E3157] bg-[#080D16] hover:border-[#3478FF] px-4 py-3 flex items-center gap-3 text-left transition-colors disabled:opacity-40"
                      >
                        <ArrowsClockwise
                          size={24}
                          weight="duotone"
                          className="text-[#3478FF] shrink-0"
                        />

                        <div className="min-w-0 flex-1">
                          <div className="font-mono text-[10px] text-white">
                            Explain Q-Learning
                          </div>

                          <div className="font-mono text-[7px] text-[#6B7B9A] mt-1">
                            Routing behaviour
                          </div>
                        </div>
                      </button>


                      {/* GENERATE REPORT */}
                      <button
                        type="button"
                        disabled={assistantLoading}
                        onClick={() =>
                          askAssistant(
                            "Generate a concise research-style report summarising the latest orchestration experiment, results, policy performance and key findings."
                          )
                        }
                        className="group min-h-[84px] border border-[#1E3157] bg-[#080D16] hover:border-[#3478FF] px-4 py-3 flex items-center gap-3 text-left transition-colors disabled:opacity-40"
                      >
                        <Terminal
                          size={24}
                          weight="duotone"
                          className="text-[#3478FF] shrink-0"
                        />

                        <div className="min-w-0 flex-1">
                          <div className="font-mono text-[10px] text-white">
                            Generate Report
                          </div>

                          <div className="font-mono text-[7px] text-[#6B7B9A] mt-1">
                            Research summary
                          </div>
                        </div>
                      </button>

                    </div>
                  </div>


                  {/* AI ASSISTANT */}
                  <div className="relative overflow-hidden border border-[#1E3157] bg-[#070B12] p-4">

                    {/* HEADER */}
                    <div className="flex items-center gap-2 relative z-20">
                      <Brain
                        size={17}
                        weight="duotone"
                        className="text-[#3478FF]"
                      />

                      <div className="font-mono text-[9px] uppercase tracking-[0.12em] text-[#8BAEFF]">
                        AI Assistant
                      </div>

                      <div className="flex items-center gap-1.5 font-mono text-[8px] text-[#72E06A]">
                        <span className="w-2 h-2 rounded-full bg-[#72E06A]" />
                        LIVE
                      </div>
                    </div>


                    <div className="grid grid-cols-1 md:grid-cols-[1fr_210px] gap-3 mt-4 items-center relative z-10">

                      {/* TEXT */}
                      <div className="border border-[#1E3157] bg-[#09111F] p-4 min-h-[92px] flex items-center">
                        <div className="font-mono text-[9px] leading-5 text-[#8B98B4]">
                          Ask about orchestration decisions, policy behaviour,
                          experimental results and Edge–Cloud performance.
                        </div>
                      </div>


                      {/* CSS ROBOT */}
                      <div className="relative h-[170px]">

                        {/* glow */}
                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,85,255,0.20),transparent_68%)]" />

                        {/* antenna left */}
                        <div className="absolute left-[52px] top-[3px] w-[2px] h-[37px] bg-[#6DA7FF]">
                          <div className="absolute -top-[3px] -left-[3px] w-2 h-2 rounded-full bg-[#55C7FF] shadow-[0_0_8px_#55C7FF]" />
                        </div>

                        {/* antenna right */}
                        <div className="absolute right-[51px] top-[3px] w-[2px] h-[37px] bg-[#6DA7FF]">
                          <div className="absolute -top-[3px] -left-[3px] w-2 h-2 rounded-full bg-[#55C7FF] shadow-[0_0_8px_#55C7FF]" />
                        </div>

                        {/* head shell */}
                        <div className="absolute left-1/2 -translate-x-1/2 top-[20px] w-[118px] h-[91px] rounded-[38px] border-[4px] border-[#B9C2D3] bg-gradient-to-b from-[#E1E5ED] via-[#9CA7BB] to-[#5E6879] shadow-[0_0_20px_rgba(44,126,255,0.22)]">

                          {/* ears */}
                          <div className="absolute -left-[15px] top-[28px] w-[16px] h-[38px] rounded-l-xl border border-[#7194C9] bg-[#75849D]" />

                          <div className="absolute -right-[15px] top-[28px] w-[16px] h-[38px] rounded-r-xl border border-[#7194C9] bg-[#75849D]" />

                          {/* face */}
                          <div className="absolute inset-[9px] rounded-[29px] bg-[#050A12] border border-[#27344A]">

                            {/* left eye */}
                            <div className="absolute left-[21px] top-[24px] w-[18px] h-[18px] rounded-full border-[3px] border-[#50C7FF] shadow-[0_0_8px_#3478FF]" />

                            {/* right eye */}
                            <div className="absolute right-[21px] top-[24px] w-[18px] h-[18px] rounded-full border-[3px] border-[#50C7FF] shadow-[0_0_8px_#3478FF]" />

                            {/* smile */}
                            <div className="absolute left-1/2 -translate-x-1/2 bottom-[17px] w-[18px] h-[8px] border-b-2 border-[#50C7FF] rounded-[50%]" />

                          </div>
                        </div>

                        {/* body */}
                        <div className="absolute left-1/2 -translate-x-1/2 top-[109px] w-[80px] h-[55px] rounded-t-[23px] rounded-b-[15px] border-[3px] border-[#94A0B5] bg-gradient-to-b from-[#CCD3DF] to-[#596475]">

                          <div className="absolute left-1/2 -translate-x-1/2 top-[16px] px-2 py-0.5 rounded-md border border-[#3478FF] bg-[#09111F] font-mono text-[8px] text-[#55C7FF] shadow-[0_0_8px_rgba(52,120,255,0.45)]">
                            AI
                          </div>
                        </div>

                        {/* arms */}
                        <div className="absolute left-[42px] top-[118px] w-[23px] h-[42px] rounded-l-[16px] border border-[#7A879B] bg-gradient-to-b from-[#AEB7C6] to-[#566172] rotate-[18deg]" />

                        <div className="absolute right-[42px] top-[118px] w-[23px] h-[42px] rounded-r-[16px] border border-[#7A879B] bg-gradient-to-b from-[#AEB7C6] to-[#566172] rotate-[-18deg]" />

                      </div>
                    </div>


                    {/* INPUT */}
                    <form
                      onSubmit={(e) => {
                        e.preventDefault();
                        askAssistant();
                      }}
                      className="flex gap-2 mt-3 relative z-20"
                    >
                      <input
                        type="text"
                        value={assistantInput}
                        onChange={(e) =>
                          setAssistantInput(
                            e.target.value
                          )
                        }
                        disabled={assistantLoading}
                        placeholder="Ask me anything about the orchestration system..."
                        className="flex-1 min-w-0 bg-[#070B12] border border-[#1E3157] px-4 py-3 font-mono text-[9px] text-white placeholder:text-[#52617C] outline-none focus:border-[#3478FF]"
                      />

                      <button
                        type="submit"
                        disabled={
                          assistantLoading ||
                          !assistantInput.trim()
                        }
                        className="w-12 h-12 border border-[#3478FF] bg-[#1747C7] text-white hover:bg-[#235CFF] disabled:opacity-40 flex items-center justify-center font-mono text-[20px]"
                      >
                        ?
                      </button>
                    </form>

                  </div>
                </div>

                {/* SUGGESTED QUESTIONS */}
                <div className="mt-4">

                  <div className="font-mono text-[8px] uppercase tracking-[0.12em] text-[#52525B] mb-2">
                    Suggested Questions
                  </div>

                  <div className="flex flex-wrap gap-1.5">

                    {[
                      "Explain the latest routing decision.",
                      "Summarise the latest infrastructure conditions.",
                      "Explain the current Q-Learning routing preferences.",
                      "Explain how Edge, Cloud and Hybrid execution differ in the implemented system.",
                      "Explain the Q-Learning state and Q-values.",
                    ].map((suggestion) => (

                      <button
                        key={suggestion}
                        type="button"
                        disabled={assistantLoading}
                        onClick={() =>
                          askAssistant(
                            suggestion
                          )
                        }
                        className="font-mono text-[8px] border border-[#27272A] text-[#71717A] hover:text-white hover:border-[#0055FF] px-2 py-1.5 disabled:opacity-40"
                      >
                        {suggestion}
                      </button>

                    ))}

                  </div>
                </div>


                {/* CONVERSATION */}
                <div className="border border-[#27272A] bg-black mt-4 max-h-[380px] overflow-y-auto thin-scroll">

                  {assistantMessages.length === 0 ? (

                    <div className="font-mono text-[9px] text-[#52525B] text-center py-6">
                      Select a quick action, choose a suggested question,
                      or ask the AI Assistant.
                    </div>

                  ) : (

                    <div>
                      {assistantMessages.map(
                        (
                          message,
                          index
                        ) => (

                          <div
                            key={index}
                            className="border-b border-[#18181B] p-4"
                          >

                            <div className="flex items-center gap-2">

                              <span className="font-mono text-[9px] text-[#0055FF]">
                                SOPHIA
                              </span>

                              <span className="font-mono text-[8px] text-[#52525B]">
                                {message.ts}
                              </span>

                            </div>

                            <div className="font-mono text-[10px] text-white mt-2">
                              {message.q}
                            </div>

                            {message.a ? (

                              <div className="mt-4">

                                <div className="font-mono text-[9px] text-[#00E676]">
                                  AI ASSISTANT
                                </div>

                                <div className="font-mono text-[10px] text-[#D4D4D8] leading-5 whitespace-pre-wrap mt-2">
                                  {message.a}
                                </div>

                              </div>

                            ) : (

                              <div className="flex items-center gap-2 mt-4">

                                <CircleNotch
                                  size={11}
                                  className="animate-spin text-[#0055FF]"
                                />

                                <span className="font-mono text-[9px] text-[#71717A]">
                                  Analysing...
                                </span>

                              </div>

                            )}

                          </div>

                        )
                      )}
                    </div>

                  )}

                </div>

              </Cell>
            </section>

{/* =================================================
            {/* =================================================
                ORCHESTRATION EVIDENCE
            ================================================== */}

            <section
              id="evidence-section"
              className="scroll-mt-24"
            >
              <div className="mb-3">
                <h2 className="font-mono text-lg mt-1">
                  Q-Learning Policy & Q-Table Analysis
                </h2>

                <p className="font-mono text-[9px] text-[#71717A] mt-1">
                  Analysis of learned Q-table states and routing preferences for Edge, Cloud and Hybrid execution based on real infrastructure conditions.
                </p>
              </div>
<div className="grid grid-cols-1 gap-3 mt-3">

                {/* Q TABLE */}
                <Cell className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <Overline>
                        Q-LEARNING ROUTING POLICY
                      </Overline>

                      <div className="font-mono text-[9px] text-[#71717A] mt-1">
                        Learned routing preferences based on real infrastructure conditions
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={
                        refreshQTable
                      }
                      className="font-mono text-[8px] uppercase px-2 py-1.5 border border-[#27272A] text-[#A1A1AA] hover:text-white"
                    >
                      Refresh
                    </button>
                  </div>

                  {qtable.length > 0 ? (
  <div className="max-h-[300px] overflow-y-auto thin-scroll">
    <table className="w-full font-mono text-[9px]">
      <thead className="text-[#71717A] border-b border-[#27272A]">
        <tr>
          <th className="text-left py-2 pr-3">Infrastructure Condition</th>
          <th className="text-right py-2 px-2">Edge Score</th>
          <th className="text-right py-2 px-2">Cloud Score</th>
          <th className="text-right py-2 px-2">Hybrid Score</th>
          <th className="text-right py-2 px-2">Best Route</th>
        </tr>
      </thead>

      <tbody>
        {qtable
          .filter(
            (row) =>
              Array.isArray(row.q_values) &&
              row.q_values.some(
                (value) => Number(value) !== 0
              )
          )
          .slice(0, 20)
          .map((row, index) => (
            <tr
              key={`${infrastructureProfile(row.state)}-${index}`}
              className="border-b border-[#18181B]"
            >
              <td className="py-2 pr-3 text-[#E4E4E7]">
                {infrastructureProfile(row.state)}
              </td>

              <td className="py-2 px-2 text-right text-[#A1A1AA]">
                {Number(row.q_values?.[0] ?? 0).toFixed(4)}
              </td>

              <td className="py-2 px-2 text-right text-[#A1A1AA]">
                {Number(row.q_values?.[1] ?? 0).toFixed(4)}
              </td>

              <td className="py-2 px-2 text-right text-[#A1A1AA]">
                {Number(row.q_values?.[2] ?? 0).toFixed(4)}
              </td>

              <td className="py-2 px-2 text-right uppercase text-white">
                {row.best_route}
              </td>
            </tr>
          ))}
      </tbody>
    </table>

    <div className="mt-5 border-t border-[#27272A] pt-4">
                      <p className="font-mono text-[9px] text-[#8B95A7] leading-5 mb-4">The table presents selected learned Q-table states and their corresponding Q-values for the Edge, Cloud and Hybrid routing actions. The action with the highest Q-value represents the learned preferred route for that state.</p>

<div className="font-mono text-[8px] uppercase tracking-[0.18em] text-[#71717A] mb-3">
        About These Results
      </div>

      <div className="font-mono text-[9px] text-[#71717A] leading-5 space-y-2">
        <p>
          <span className="text-[#E4E4E7]">Infrastructure Profiles</span>
          {" "}are derived from the real conditions used by Q-Learning. Network conditions are based on measured AWS network latency, while CPU availability comes from Edge node telemetry.
        </p>

        <p>
          <span className="text-[#E4E4E7]">Edge, Cloud and Hybrid Scores</span>
          {" "}are learned Q-values from real executions.
          {" "}
          <span className="text-[#E4E4E7]">Best Route</span>
          {" "}is the route with the highest learned score.
        </p>

        <p>
          <span className="text-[#FFCC00]">Q-values are learned decision signals</span>
          {" "}— not latency, percentages, probabilities or model accuracy.
        </p>
      </div>
    </div>
  </div>
) : (
  <div className="font-mono text-[10px] text-[#52525B] text-center py-10">
    Q-table not loaded.
  </div>
)}
                </Cell>
              </div>
            </section>

            <section id="experimental-evidence-section" className="scroll-mt-24 mb-6">
              <div className="border border-[#27272A] px-4 py-4 flex items-center justify-between gap-4">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-[#D4D4D8]">
                    EXPERIMENTAL EVIDENCE
                  </div>
                  <div className="font-mono text-[8px] text-[#71717A] mt-1">
                    Measured Edge, Cloud and Hybrid evaluation data
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => {
                    window.open(
                      "http://127.0.0.1:8001/api/report/real-evidence",
                      "_blank"
                    );
                  }}
                  className="font-mono text-[8px] uppercase px-4 py-2.5 border border-[#52525B] text-[#E4E4E7] hover:border-[#A1A1AA] hover:text-white whitespace-nowrap"
                >
                  <span className="inline-flex items-center gap-2">
                    <FilePdf size={14} />
                    Download Experimental Evidence
                  </span>
                </button>
              </div>
            </section>


            <section
              id="logs-section"
              className="scroll-mt-24"
            >
              <Cell>
                <div className="px-4 py-3 border-b border-[#27272A] flex items-center justify-between">
                  <div>
                    <Overline>
                      SYSTEM LOGS | LIVE
                    </Overline>

                    <div className="font-mono text-[9px] text-[#71717A] mt-1">
                      Frontend orchestration, workload and experiment events.
                    </div>
                  </div>

                  <Terminal
                    size={15}
                    className="text-[#71717A]"
                  />
                </div>

                <div
                  ref={
                    logsRef
                  }
                  className="h-[290px] overflow-y-auto thin-scroll bg-black p-3 font-mono text-[9px]"
                >
                  {logs.length >
                  0 ? (
                    logs.map(
                      (
                        log,
                        index
                      ) => (
                        <div
                          key={
                            index
                          }
                          className="flex gap-3 py-1 border-b border-[#0F0F0F]"
                        >
                          <span className="text-[#52525B] shrink-0">
                            {
                              log.t
                            }
                          </span>

                          <span
                            className={`shrink-0 uppercase ${
                              log.level ===
                              "ok"
                                ? "text-[#00E676]"
                                : log.level ===
                                    "err"
                                  ? "text-[#FF3333]"
                                  : log.level ===
                                      "warn"
                                    ? "text-[#FFCC00]"
                                    : "text-[#0055FF]"
                            }`}
                          >
                            [
                            {
                              log.level
                            }
                            ]
                          </span>

                          <span className="text-[#A1A1AA] break-all">
                            {
                              log.msg
                            }
                          </span>
                        </div>
                      )
                    )
                  ) : (
                    <div className="text-[#52525B]">
                      Waiting for system events...
                    </div>
                  )}
                </div>
              </Cell>
            </section>

            {/* =================================================
                AI-ASSISTED DECISION SUPPORT
            ================================================== */}

            
            
            {/* =================================================
                SYSTEM SETTINGS
            ================================================== */}

            <section className="scroll-mt-24">
              <div className="mb-3">
                <Overline>
                  SYSTEM SETTINGS
                </Overline>

                <h2 className="font-mono text-lg mt-1">
                  Orchestration Controls
                </h2>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">

                {/* LIVE TICK */}
                <Cell className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Overline>
                        LIVE ORCHESTRATION
                      </Overline>

                      <div className="font-mono text-[9px] text-[#71717A] mt-1">
                        Automatic policy evaluation using live infrastructure.
                      </div>
                    </div>

                    <Broadcast
                      size={16}
                      className={
                        running
                          ? "text-[#00E676]"
                          : "text-[#FF3333]"
                      }
                    />
                  </div>

                  <button
                    type="button"
                    onClick={() =>
                      setRunning(
                        (
                          value
                        ) =>
                          !value
                      )
                    }
                    className={`w-full mt-5 border px-3 py-3 font-mono text-[9px] uppercase ${
                      running
                        ? "border-[#00E676]/30 text-[#00E676]"
                        : "border-[#FF3333]/30 text-[#FF3333]"
                    }`}
                  >
                    {running
                      ? "Orchestration Running"
                      : "Orchestration Paused"}
                  </button>
                </Cell>

                {/* ARCHITECTURE */}
                <Cell className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Overline>
                        ARCHITECTURE
                      </Overline>

                      <div className="font-mono text-[9px] text-[#71717A] mt-1">
                        Inspect the complete Edge-Cloud topology.
                      </div>
                    </div>

                    <TreeStructure
                      size={16}
                      className="text-[#FFCC00]"
                    />
                  </div>

                  <button
                    type="button"
                    onClick={() =>
                      setShowArch(
                        true
                      )
                    }
                    className="w-full mt-5 border border-[#27272A] hover:border-[#FFCC00] px-3 py-3 font-mono text-[9px] uppercase text-[#A1A1AA] hover:text-white"
                  >
                    Open Architecture
                  </button>
                </Cell>
              </div>
            </section>

            {/* =================================================
                SYSTEM SUMMARY
            ================================================== */}

            <section>
              <Cell className="p-4">
                <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3">

                  <div>
                    <div className="font-mono text-[7px] uppercase text-[#52525B]">
                      Edge
                    </div>

                    <div
                      className={`font-mono text-[9px] mt-1 ${
                        edgeConnected
                          ? "text-[#00E676]"
                          : "text-[#FF3333]"
                      }`}
                    >
                      {edgeConnected
                        ? "ONLINE"
                        : "OFFLINE"}
                    </div>
                  </div>

                  <div>
                    <div className="font-mono text-[7px] uppercase text-[#52525B]">
                      Cloud
                    </div>

                    <div
                      className={`font-mono text-[9px] mt-1 ${
                        cloudConnected
                          ? "text-[#00E676]"
                          : "text-[#FF3333]"
                      }`}
                    >
                      {cloudConnected
                        ? "ONLINE"
                        : "OFFLINE"}
                    </div>
                  </div>

                  <div>
                    <div className="font-mono text-[7px] uppercase text-[#52525B]">
                      Policy
                    </div>

                    <div className="font-mono text-[9px] mt-1 text-white">
                      {engineLabel(
                        engineName
                      )}
                    </div>
                  </div>

                  <div>
                    <div className="font-mono text-[7px] uppercase text-[#52525B]">
                      Route
                    </div>

                    <div
                      className={`font-mono text-[9px] uppercase mt-1 ${routeTone(
                        currentExecutedRoute
                      )}`}
                    >
                      {
                        currentExecutedRoute
                      }
                    </div>
                  </div>

                  <div>
                    <div className="font-mono text-[7px] uppercase text-[#52525B]">
                      Failover
                    </div>

                    <div
                      className={`font-mono text-[9px] mt-1 ${
                        failoverApplied
                          ? "text-[#FFCC00]"
                          : "text-[#00E676]"
                      }`}
                    >
                      {failoverApplied
                        ? "YES"
                        : "NO"}
                    </div>
                  </div>

                  <div>
                    <div className="font-mono text-[7px] uppercase text-[#52525B]">
                      Execution
                    </div>

                    <div
                      className={`font-mono text-[9px] mt-1 ${
                        executionStatus ===
                        "SUCCESS"
                          ? "text-[#00E676]"
                          : executionStatus ===
                              "FAILED"
                            ? "text-[#FF3333]"
                            : "text-[#71717A]"
                      }`}
                    >
                      {
                        executionStatus
                      }
                    </div>
                  </div>

                  <div>
                    <div className="font-mono text-[7px] uppercase text-[#52525B]">
                      Scenarios
                    </div>

                    <div className="font-mono text-[9px] text-white mt-1">
                      {scenarios.length}
                    </div>
                  </div>

                  <div>
                    <div className="font-mono text-[7px] uppercase text-[#52525B]">
                      Models
                    </div>

                    <div className="font-mono text-[9px] text-white mt-1">
                      {models.length}
                    </div>
                  </div>
                </div>
              </Cell>
            </section>

            {/* =================================================
                FOOTER
            ================================================== */}

            <footer className="border-t border-[#27272A] pt-5 pb-8 mt-6">
              <div className="flex flex-col xl:flex-row xl:items-center xl:justify-between gap-4">

                <div>
                  <div className="font-mono text-[10px] tracking-[0.12em] text-white uppercase">
                    EDGE-CLOUD // ORCHESTRATOR
                  </div>

                  <div className="font-mono text-[8px] text-[#71717A] mt-1">
                    AI-Driven Infrastructure Orchestration for Edge-Cloud Environments
                  </div>
                </div>

                <div className="flex flex-wrap gap-x-5 gap-y-2 font-mono text-[8px] uppercase text-[#52525B]">
                  <span>
                    University Project
                  </span>

                  <span>
                    MSc Artificial Intelligence
                  </span>

                  <span>
                    University of Bedfordshire
                  </span>

                  <span>
                    Edge + AWS
                  </span>

                  <span>
                    2 Primary Policies
                  </span>
                </div>
              </div>

              <div className="mt-4 flex flex-col md:flex-row md:items-center md:justify-between gap-2 font-mono text-[7px] uppercase tracking-[0.12em] text-[#3F3F46]">
                <span>
                  Rule-Based | Q-Learning
                </span>

                <span>
                  Edge | Cloud | Hybrid | Automatic Failover
                </span>
              </div>
            </footer>

          </div>
        </main>
      </div>

      {/* =====================================================
          ARCHITECTURE MODAL
      ====================================================== */}

      {showArch && (
        <ArchitectureModal
          onClose={() =>
            setShowArch(
              false
            )
          }
          status={
            engineStatus
          }
          edgeInfra={
            edgeInfra
          }
          cloudInfra={
            cloudInfra
          }
          edgeConnected={
            edgeConnected
          }
          cloudConnected={
            cloudConnected
          }
        />
      )}
    </div>
  );
}
/* =========================================================
   ARCHITECTURE MODAL
========================================================= */

function ArchitectureModal({
  onClose,
  status,
  edgeInfra,
  cloudInfra,
  edgeConnected,
  cloudConnected,
}) {
  const architecturePolicyStatus = {
    RuleBased:
      status?.RuleBased || {},
    DecisionTree:
      status?.DecisionTree || {},
    RandomForest:
      status?.RandomForest || {},
    QLearning:
      status?.QLearning || {},
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-[#080808] border border-[#27272A] w-full max-w-6xl max-h-[92vh] overflow-y-auto thin-scroll"
        onClick={(e) =>
          e.stopPropagation()
        }
      >
        {/* Header */}
        <div className="sticky top-0 z-10 bg-[#080808] border-b border-[#27272A] px-5 md:px-7 py-5 flex items-start justify-between gap-4">
          <div>
            <Overline>
              SYSTEM ARCHITECTURE
            </Overline>

            <h2 className="font-mono text-xl md:text-2xl mt-2">
              EDGE-CLOUD // ORCHESTRATOR
            </h2>

            <p className="font-mono text-[9px] text-[#71717A] mt-2 max-w-3xl leading-5">
              Implemented Edge-Cloud infrastructure integrating intelligent
              orchestration policies for AI inference routing.
            </p>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="font-mono text-[9px] uppercase px-3 py-2 border border-[#27272A] text-[#A1A1AA] hover:text-white hover:border-white"
          >
            Close
          </button>
        </div>

        <div className="p-5 md:p-7 space-y-5">

          {/* =================================================
              HIGH-LEVEL TOPOLOGY
          ================================================== */}

          <section>
            <Overline>
              LIVE TOPOLOGY
            </Overline>

            <div className="grid grid-cols-1 lg:grid-cols-[1fr_0.8fr_1fr] gap-3 mt-3">

              {/* Edge */}
              <Cell className="p-5 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-[2px] bg-[#00E676]" />

                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 border border-[#00E676]/40 flex items-center justify-center">
                      <Cpu
                        size={20}
                        className="text-[#00E676]"
                      />
                    </div>

                    <div>
                      <div className="font-mono text-[9px] text-[#71717A] uppercase">
                        Edge Compute
                      </div>

                      <div className="font-mono text-sm text-white mt-1">
                        {edgeInfra?.node ||
                          "edge-node-01"}
                      </div>
                    </div>
                  </div>

                  <span
                    className={`font-mono text-[8px] ${
                      edgeConnected
                        ? "text-[#00E676]"
                        : "text-[#FF3333]"
                    }`}
                  >
                    {edgeConnected
                      ? "ONLINE"
                      : "OFFLINE"}
                  </span>
                </div>

                <div className="mt-5 space-y-2 font-mono text-[9px]">
                  <div className="flex items-center justify-between border-b border-[#18181B] pb-2">
                    <span className="text-[#71717A]">
                      Node
                    </span>

                    <span className="text-white">
                      {edgeInfra?.node ||
                        "edge-node-01"}
                    </span>
                  </div>

                  <div className="flex items-center justify-between border-b border-[#18181B] pb-2">
                    <span className="text-[#71717A]">
                      Platform
                    </span>

                    <span className="text-white">
                      {edgeInfra?.platform ||
                        "Linux"}
                    </span>
                  </div>

                  <div className="flex items-center justify-between border-b border-[#18181B] pb-2">
                    <span className="text-[#71717A]">
                      CPU Cores
                    </span>

                    <span className="text-white">
                      {edgeInfra?.cpu_count ??
                        "--"}
                    </span>
                  </div>

                  <div className="flex items-center justify-between border-b border-[#18181B] pb-2">
                    <span className="text-[#71717A]">
                      Memory
                    </span>

                    <span className="text-white">
                      {edgeInfra?.memory_total_gb != null
                        ? `${Number(
                            edgeInfra.memory_total_gb
                          ).toFixed(2)} GB`
                        : "--"}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-[#71717A]">
                      Runtime
                    </span>

                    <span className="text-[#00E676]">
                      REAL INFERENCE
                    </span>
                  </div>
                </div>
              </Cell>

              {/* Orchestrator */}
              <Cell className="p-5 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-[2px] bg-[#FFCC00]" />

                <div className="flex justify-center">
                  <div className="w-12 h-12 border border-[#FFCC00]/40 flex items-center justify-center">
                    <Brain
                      size={24}
                      className="text-[#FFCC00]"
                    />
                  </div>
                </div>

                <div className="text-center mt-4">
                  <div className="font-mono text-[9px] text-[#71717A] uppercase">
                    Central Orchestrator
                  </div>

                  <div className="font-mono text-sm text-white mt-1">
                    FastAPI Decision Engine
                  </div>
                </div>

                <div className="mt-5 space-y-2">
                  {[
                    "Rule-Based",
                    "Q-Learning",
                  ].map(
                    (name) => (
                      <div
                        key={name}
                        className="border border-[#27272A] px-3 py-2 flex items-center justify-between"
                      >
                        <span className="font-mono text-[9px] text-[#A1A1AA]">
                          {name}
                        </span>

                        <span className="w-1.5 h-1.5 rounded-full bg-[#00E676]" />
                      </div>
                    )
                  )}
                </div>

                <div className="mt-5 flex items-center justify-center gap-2 font-mono text-[8px]">
                  <span className="text-[#00E676]">
                    EDGE
                  </span>

                  <span className="text-[#52525B]">
                    ->
                  </span>

                  <span className="text-[#FFCC00]">
                    HYBRID
                  </span>

                  <span className="text-[#52525B]">
                    ->
                  </span>

                  <span className="text-[#0055FF]">
                    CLOUD
                  </span>
                </div>
              </Cell>

              {/* Cloud */}
              <Cell className="p-5 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-[2px] bg-[#0055FF]" />

                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 border border-[#0055FF]/40 flex items-center justify-center">
                      <CloudArrowUp
                        size={20}
                        className="text-[#0055FF]"
                      />
                    </div>

                    <div>
                      <div className="font-mono text-[9px] text-[#71717A] uppercase">
                        AWS Cloud
                      </div>

                      <div className="font-mono text-sm text-white mt-1">
                        {cloudInfra?.node ||
                          "cloud-node-01"}
                      </div>
                    </div>
                  </div>

                  <span
                    className={`font-mono text-[8px] ${
                      cloudConnected
                        ? "text-[#00E676]"
                        : "text-[#FF3333]"
                    }`}
                  >
                    {cloudConnected
                      ? "ONLINE"
                      : "OFFLINE"}
                  </span>
                </div>

                <div className="mt-5 space-y-2 font-mono text-[9px]">
                  <div className="flex items-center justify-between border-b border-[#18181B] pb-2">
                    <span className="text-[#71717A]">
                      Provider
                    </span>

                    <span className="text-white">
                      AWS EC2
                    </span>
                  </div>

                  <div className="flex items-center justify-between border-b border-[#18181B] pb-2">
                    <span className="text-[#71717A]">
                      Region
                    </span>

                    <span className="text-white">
                      eu-west-2
                    </span>
                  </div>

                  <div className="flex items-center justify-between border-b border-[#18181B] pb-2">
                    <span className="text-[#71717A]">
                      Platform
                    </span>

                    <span className="text-white">
                      {cloudInfra?.platform ||
                        "Linux"}
                    </span>
                  </div>

                  <div className="flex items-center justify-between border-b border-[#18181B] pb-2">
                    <span className="text-[#71717A]">
                      CPU Cores
                    </span>

                    <span className="text-white">
                      {cloudInfra?.cpu_count ??
                        "--"}
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-[#71717A]">
                      Runtime
                    </span>

                    <span className="text-[#0055FF]">
                      REAL INFERENCE
                    </span>
                  </div>
                </div>
              </Cell>
            </div>
          </section>

          {/* =================================================
              DATA FLOW
          ================================================== */}

          <section>
            <Overline>
              ORCHESTRATION DATA FLOW
            </Overline>

            <Cell className="p-5 mt-3">
              <div className="grid grid-cols-1 md:grid-cols-5 gap-3 items-stretch">

                <div className="border border-[#27272A] p-4">
                  <Broadcast
                    size={18}
                    className="text-[#00E676]"
                  />

                  <div className="font-mono text-[9px] text-white mt-3">
                    1. Telemetry
                  </div>

                  <div className="font-mono text-[8px] text-[#71717A] leading-4 mt-2">
                    CPU, RAM, network RTT and connectivity are collected from the live infrastructure.
                  </div>
                </div>

                <div className="border border-[#27272A] p-4">
                  <Database
                    size={18}
                    className="text-[#A1A1AA]"
                  />

                  <div className="font-mono text-[9px] text-white mt-3">
                    2. Context
                  </div>

                  <div className="font-mono text-[8px] text-[#71717A] leading-4 mt-2">
                    Workload and infrastructure features form the decision context.
                  </div>
                </div>

                <div className="border border-[#FFCC00]/30 p-4">
                  <Brain
                    size={18}
                    className="text-[#FFCC00]"
                  />

                  <div className="font-mono text-[9px] text-white mt-3">
                    3. Policy
                  </div>

                  <div className="font-mono text-[8px] text-[#71717A] leading-4 mt-2">
                    The selected policy chooses Edge, Cloud or Hybrid execution.
                  </div>
                </div>

                <div className="border border-[#0055FF]/30 p-4">
                  <Lightning
                    size={18}
                    className="text-[#0055FF]"
                  />

                  <div className="font-mono text-[9px] text-white mt-3">
                    4. Execution
                  </div>

                  <div className="font-mono text-[8px] text-[#71717A] leading-4 mt-2">
                    The workload executes on the selected real compute node or Hybrid path.
                  </div>
                </div>

                <div className="border border-[#00E676]/30 p-4">
                  <ChartLineUp
                    size={18}
                    className="text-[#00E676]"
                  />

                  <div className="font-mono text-[9px] text-white mt-3">
                    5. Metrics
                  </div>

                  <div className="font-mono text-[8px] text-[#71717A] leading-4 mt-2">
                    Latency, CPU, memory, success and routing metrics are returned and persisted.
                  </div>
                </div>
              </div>
            </Cell>
          </section>

          {/* =================================================
              POLICY LAYER
          ================================================== */}

          <section>
            <Overline>
              INTELLIGENT DECISION LAYER
            </Overline>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">

              {/* Rule-Based */}
              <Cell className="p-4">
                <div className="flex items-center justify-between">
                  <Path
                    size={17}
                    className="text-[#A1A1AA]"
                  />

                  <span className="font-mono text-[8px] text-[#00E676]">
                    AVAILABLE
                  </span>
                </div>

                <div className="font-mono text-sm text-white mt-4">
                  Rule-Based
                </div>

                <p className="font-mono text-[8px] leading-4 text-[#71717A] mt-2">
                  Deterministic baseline policy using explicit thresholds for latency, capacity, workload and cost.
                </p>

                <div className="border-t border-[#27272A] mt-4 pt-3 font-mono text-[8px] text-[#71717A]">
                  Training:{" "}
                  <span className="text-white">
                    Not required
                  </span>
                </div>
              </Cell>

              {/* QL */}
              <Cell className="p-4 border-[#FFCC00]/30">
                <div className="flex items-center justify-between">
                  <Lightning
                    size={17}
                    className="text-[#FFCC00]"
                  />

                  <span
                    className={`font-mono text-[8px] ${
                      architecturePolicyStatus
                        .QLearning
                        ?.available ===
                      false
                        ? "text-[#FF3333]"
                        : "text-[#00E676]"
                    }`}
                  >
                    {architecturePolicyStatus
                      .QLearning
                      ?.available ===
                    false
                      ? "UNAVAILABLE"
                      : "AVAILABLE"}
                  </span>
                </div>

                <div className="font-mono text-sm text-white mt-4">
                  Q-Learning
                </div>

                <p className="font-mono text-[8px] leading-4 text-[#71717A] mt-2">
                  Reinforcement-learning agent selecting routes from learned state-action values.
                </p>

                <div className="border-t border-[#27272A] mt-4 pt-3 font-mono text-[8px] text-[#71717A]">
                  Learning:{" "}
                  <span className="text-[#FFCC00]">Execution Feedback</span>
                </div>
              </Cell>
            </div>
          </section>

          {/* =================================================
              RESILIENCE
          ================================================== */}

          <section>
            <Overline>
              RESILIENCE & FAILOVER
            </Overline>

            <Cell className="p-5 mt-3">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">

                <div className="border border-[#27272A] p-4">
                  <WifiHigh
                    size={18}
                    className="text-[#00E676]"
                  />

                  <div className="font-mono text-[9px] text-white mt-3">
                    Normal Operation
                  </div>

                  <p className="font-mono text-[8px] text-[#71717A] leading-4 mt-2">
                    The policy selects Edge, Cloud or Hybrid using the current infrastructure state.
                  </p>
                </div>

                <div className="border border-[#FFCC00]/30 p-4">
                  <Warning
                    size={18}
                    className="text-[#FFCC00]"
                  />

                  <div className="font-mono text-[9px] text-white mt-3">
                    Failure Detection
                  </div>

                  <p className="font-mono text-[8px] text-[#71717A] leading-4 mt-2">
                    Execution success and node connectivity are checked before and after routing.
                  </p>
                </div>

                <div className="border border-[#0055FF]/30 p-4">
                  <CloudArrowUp
                    size={18}
                    className="text-[#0055FF]"
                  />

                  <div className="font-mono text-[9px] text-white mt-3">
                    Automatic Failover
                  </div>

                  <p className="font-mono text-[8px] text-[#71717A] leading-4 mt-2">
                    Failed Edge or Hybrid execution can be rerouted to the healthy AWS Cloud node.
                  </p>
                </div>
              </div>
            </Cell>
          </section>

          {/* =================================================
              RESEARCH CONTEXT
          ================================================== */}

          <section>
            <Overline>
              RESEARCH CONTEXT
            </Overline>

            <Cell className="p-5 mt-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">

                <div>
                  <div className="font-mono text-[9px] text-white uppercase">
                    Project
                  </div>

                  <p className="font-mono text-[8px] text-[#71717A] leading-5 mt-2">
                    <span className="text-[#A1A1AA]">AI-Driven Infrastructure Orchestration for Edge-Cloud Environments</span>
                    <br />
                    Intelligent routing and execution across Edge, Cloud and Hybrid computing environments.
                  </p>
                </div>

                <div>
                  <div className="font-mono text-[9px] text-white uppercase">
                    Academic Context
                  </div>

                  <p className="font-mono text-[8px] text-[#71717A] leading-5 mt-2">
                    University Project | MSc Artificial Intelligence | University of Bedfordshire.
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-5">
                <div className="border border-[#18181B] p-3">
                  <div className="font-mono text-[7px] text-[#52525B] uppercase">
                    Policies
                  </div>

                  <div className="font-mono text-sm mt-1">
                    2
                  </div>
                </div>

                <div className="border border-[#18181B] p-3">
                  <div className="font-mono text-[7px] text-[#52525B] uppercase">
                    Routes
                  </div>

                  <div className="font-mono text-sm mt-1">
                    3
                  </div>
                </div>

                <div className="border border-[#18181B] p-3">
                  <div className="font-mono text-[7px] text-[#52525B] uppercase">
                    Infrastructure
                  </div>

                  <div className="font-mono text-sm mt-1">
                    EDGE + CLOUD
                  </div>
                </div>

                <div className="border border-[#18181B] p-3">
                  <div className="font-mono text-[7px] text-[#52525B] uppercase">
                    Cloud
                  </div>

                  <div className="font-mono text-sm mt-1">
                    AWS
                  </div>
                </div>
              </div>
            </Cell>
          </section>

          {/* Modal footer */}
          <div className="border-t border-[#27272A] pt-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div className="font-mono text-[8px] text-[#52525B] uppercase">
              EDGE-CLOUD // ORCHESTRATOR
            </div>

            <div className="font-mono text-[8px] text-[#52525B] uppercase">
              Edge | AWS Cloud | Hybrid | Intelligent Routing
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
















































































































































