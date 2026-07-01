import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import fs from "fs";

dotenv.config();

// Default seed data
const DEFAULT_USERS = [
  { name: "Đặng Thị Thanh Vy", email: "admin@agrismart.vn", role: "Quản trị hệ thống", roleTag: "SYSTEM ADMIN" },
  { name: "Đỗ Thị Hà Thanh", email: "analyst@agrismart.vn", role: "Chuyên gia dữ liệu", roleTag: "DATA ANALYST" },
  { name: "Hồ Thủy Hương", email: "farmer@agrismart.vn", role: "Hợp tác xã sầu riêng", roleTag: "FARMER COOP" },
  { name: "Lê Dương Minh Anh", email: "buyer@agrismart.vn", role: "Doanh nghiệp thu mua", roleTag: "BUYER ENTERPRISE" },
  { name: "Lê Hoàng Anh", email: "anhlh@agrismart.vn", role: "Cộng tác viên khảo sát", roleTag: "CONTRIBUTOR" },
];

const DEFAULT_COLLABORATORS = [
  {
    id: "CTV001",
    name: "Lê Hoàng Anh",
    email: "anhlh@agrismart.vn",
    phone: "0903.112.233",
    role: "Cộng tác viên khảo sát",
    region: "Đắk Lắk",
    crop: "Tiêu lốt",
    status: "Đang hoạt động",
    surveyCount: 18,
  },
  {
    id: "CTV002",
    name: "Trần Văn Hùng",
    email: "hungtv@agrismart.vn",
    phone: "0918.456.789",
    role: "Cộng tác viên Sầu riêng",
    region: "Bến Tre",
    crop: "Sầu riêng Ri6",
    status: "Đang hoạt động",
    surveyCount: 24,
  },
  {
    id: "CTV003",
    name: "Nguyễn Thị Mai",
    email: "maint@agrismart.vn",
    phone: "0932.789.012",
    role: "Cộng tác viên Thu hoạch",
    region: "Tiền Giang",
    crop: "Xoài Cát Hòa Lộc",
    status: "Đang hoạt động",
    surveyCount: 15,
  },
  {
    id: "CTV004",
    name: "Vương Đình Trung",
    email: "trungvd@agrismart.vn",
    phone: "0989.111.222",
    role: "Kỹ thuật viên sinh học",
    region: "Cần Thơ",
    crop: "Lúa mùa cao sản",
    status: "Ngoại tuyến",
    surveyCount: 31,
  },
];

const DEFAULT_NOTIFICATIONS = [
  {
    id: "notif-1",
    title: "Cảnh báo sâu bệnh sớm",
    description: "Độ ẩm Đắk Lắk giảm sút đột ngột dưới 50%. Nguy cơ rầy phấn trắng tăng cao trên sầu riêng.",
    time: "5 phút trước",
    type: "warning",
    read: false,
  },
  {
    id: "notif-2",
    title: "Đã tối ưu hóa thuật toán ML",
    description: "Huấn luyện lại mô hình dự báo sản lượng đạt độ chính xác R² = 0.94 vượt trội.",
    time: "42 phút trước",
    type: "success",
    read: false,
  },
  {
    id: "notif-3",
    title: "Thời tiết thay đổi liên vùng",
    description: "Hệ thống dự báo ghi nhận khả năng mưa giông lớn tại Lâm Đồng vào chiều tối nay.",
    time: "2 giờ trước",
    type: "info",
    read: false,
  },
  {
    id: "notif-4",
    title: "Kiểm duyệt công tác trực nhật",
    description: "Cộng tác viên Lê Hoàng Anh đã hoàn thành khảo sát thực địa sầu riêng.",
    time: "4 giờ trước",
    type: "success",
    read: true,
  },
];

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // JSON Storage paths
  const USERS_FILE = path.join(process.cwd(), "users_state.json");
  const COLLABORATORS_FILE = path.join(process.cwd(), "collaborators_state.json");
  const NOTIFICATIONS_FILE = path.join(process.cwd(), "notifications_state.json");

  // Helper functions for reading/writing with fallbacks
  const readJsonFile = (filePath: string, fallback: any) => {
    try {
      if (fs.existsSync(filePath)) {
        const raw = fs.readFileSync(filePath, "utf-8");
        return JSON.parse(raw);
      }
    } catch (e) {
      console.error(`Error reading ${filePath}:`, e);
    }
    return fallback;
  };

  const writeJsonFile = (filePath: string, data: any) => {
    try {
      fs.writeFileSync(filePath, JSON.stringify(data, null, 2), "utf-8");
      return true;
    } catch (e) {
      console.error(`Error writing to ${filePath}:`, e);
      return false;
    }
  };

  // Shared Gemini client (Lazy initialization)
  let ai: GoogleGenAI | null = null;
  const getGeminiClient = (): GoogleGenAI => {
    if (!ai) {
      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error("GEMINI_API_KEY environment variable is not configured. Please add it in the Secrets panel.");
      }
      ai = new GoogleGenAI({ apiKey });
    }
    return ai;
  };

  // API Health Endpoint
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", geminiConfigured: !!process.env.GEMINI_API_KEY });
  });

  // API Sync Endpoints for Real-time Cross-Machine sync
  app.get("/api/users", (req, res) => {
    const data = readJsonFile(USERS_FILE, DEFAULT_USERS);
    res.json({ data, isCustom: fs.existsSync(USERS_FILE) });
  });

  app.post("/api/users", (req, res) => {
    const list = Array.isArray(req.body) ? req.body : req.body.data;
    if (Array.isArray(list)) {
      writeJsonFile(USERS_FILE, list);
      res.json({ success: true, data: list });
    } else {
      res.status(400).json({ error: "Invalid data format. Expected array." });
    }
  });

  app.get("/api/collaborators", (req, res) => {
    const data = readJsonFile(COLLABORATORS_FILE, DEFAULT_COLLABORATORS);
    res.json({ data, isCustom: fs.existsSync(COLLABORATORS_FILE) });
  });

  app.post("/api/collaborators", (req, res) => {
    const list = Array.isArray(req.body) ? req.body : req.body.data;
    if (Array.isArray(list)) {
      writeJsonFile(COLLABORATORS_FILE, list);
      res.json({ success: true, data: list });
    } else {
      res.status(400).json({ error: "Invalid data format. Expected array." });
    }
  });

  app.get("/api/notifications", (req, res) => {
    const data = readJsonFile(NOTIFICATIONS_FILE, DEFAULT_NOTIFICATIONS);
    res.json({ data, isCustom: fs.existsSync(NOTIFICATIONS_FILE) });
  });

  app.post("/api/notifications", (req, res) => {
    const list = Array.isArray(req.body) ? req.body : req.body.data;
    if (Array.isArray(list)) {
      writeJsonFile(NOTIFICATIONS_FILE, list);
      res.json({ success: true, data: list });
    } else {
      res.status(400).json({ error: "Invalid data format. Expected array." });
    }
  });

  // API Analyze Endpoint (Proxy for Gemini)
  app.post("/api/analyze", async (req, res) => {
    try {
      const { prompt, systemInstruction } = req.body;
      if (!prompt) {
        res.status(400).json({ error: "Prompt is required" });
        return;
      }

      const client = getGeminiClient();
      const response = await client.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
        config: systemInstruction ? { systemInstruction } : undefined
      });

      res.json({ text: response.text });
    } catch (error: any) {
      console.error("Gemini Error:", error);
      res.status(500).json({ error: error.message || "Failed to communicate with Gemini API" });
    }
  });

  app.post("/api/chat", async (req, res) => {
    const { messages, systemInstruction } = req.body;
    try {
      const key = process.env.GEMINI_API_KEY;
      if (!key) {
        return res.status(500).json({ error: "GEMINI_API_KEY environment variable is required. Please set it in the Secrets panel." });
      }
      
      const client = getGeminiClient();
      const formattedContents = messages.map((m: any) => ({
        role: m.role === "user" ? "user" : "model",
        parts: [{ text: m.content }]
      }));

      const response = await client.models.generateContent({
        model: "gemini-2.5-flash",
        contents: formattedContents,
        config: systemInstruction ? { systemInstruction } : undefined
      });

      res.json({ text: response.text });
    } catch (error: any) {
      console.error("Gemini Chat Error:", error);
      res.status(500).json({ error: error.message || "Failed to communicate with Gemini API" });
    }
  });

  // Coordinates mapping for Vietnamese agricultural provinces
  const PROVINCE_COORDINATES: Record<string, { lat: number; lng: number }> = {
    "Đắk Lắk": { lat: 12.6667, lng: 108.0500 },
    "Tiền Giang": { lat: 10.3667, lng: 106.3500 },
    "Đồng Tháp": { lat: 10.4500, lng: 105.6333 },
    "An Giang": { lat: 10.5000, lng: 105.1667 },
    "Lâm Đồng": { lat: 11.9404, lng: 108.4583 },
    "Cần Thơ": { lat: 10.0333, lng: 105.7833 }
  };

  const getCoordinates = (prov: string) => {
    const normalized = prov.trim();
    for (const key of Object.keys(PROVINCE_COORDINATES)) {
      if (key.toLowerCase().includes(normalized.toLowerCase()) || normalized.toLowerCase().includes(key.toLowerCase())) {
        return PROVINCE_COORDINATES[key];
      }
    }
    return PROVINCE_COORDINATES["Đắk Lắk"]; // default fallback
  };

  // Weather forecast API endpoint
  app.get("/api/weather", async (req, res) => {
    try {
      const province = (req.query.province as string) || "Đắk Lắk";
      const coords = getCoordinates(province);
      
      const url = `https://api.open-meteo.com/v1/forecast?latitude=${coords.lat}&longitude=${coords.lng}&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_max,relative_humidity_2m_min&timezone=Asia/Bangkok`;
      
      const apiResponse = await fetch(url);
      if (!apiResponse.ok) {
        throw new Error(`Open-Meteo API responded with status ${apiResponse.status}`);
      }
      const data = await apiResponse.json();
      
      const daily = data.daily || {};
      const times = daily.time || [];
      const weatherCodes = daily.weather_code || [];
      const tempMaxs = daily.temperature_2m_max || [];
      const tempMins = daily.temperature_2m_min || [];
      const precipitations = daily.precipitation_sum || [];
      const humidityMaxs = daily.relative_humidity_2m_max || [];
      const humidityMins = daily.relative_humidity_2m_min || [];

      const forecast = times.map((time: string, idx: number) => {
        const code = weatherCodes[idx] ?? 0;
        let desc = "Nắng ráo";
        let icon = "sun";
        
        if (code === 0) {
          desc = "Trời trong xanh";
          icon = "sun";
        } else if (code >= 1 && code <= 3) {
          desc = "Ít mây, trời nắng";
          icon = "cloud-sun";
        } else if (code === 45 || code === 48) {
          desc = "Có sương mù";
          icon = "cloud";
        } else if (code >= 51 && code <= 55) {
          desc = "Mưa phùn nhẹ";
          icon = "cloud-drizzle";
        } else if (code >= 61 && code <= 65) {
          desc = "Mưa rào rải rác";
          icon = "cloud-rain";
        } else if (code >= 80 && code <= 82) {
          desc = "Mưa rào lớn";
          icon = "cloud-rain-wind";
        } else if (code >= 95 && code <= 99) {
          desc = "Giông bão có sấm sét";
          icon = "cloud-lightning";
        } else {
          desc = "Nhiều mây, âm u";
          icon = "cloudy";
        }

        const avgHumidity = Math.round(((humidityMaxs[idx] ?? 80) + (humidityMins[idx] ?? 50)) / 2);

        return {
          date: time,
          tempMax: tempMaxs[idx] ?? 30,
          tempMin: tempMins[idx] ?? 22,
          precipitation: precipitations[idx] ?? 0,
          humidity: avgHumidity,
          description: desc,
          icon,
          weatherCode: code
        };
      });

      res.json({
        province,
        coords,
        forecast
      });
    } catch (err: any) {
      console.error("Weather endpoint error:", err);
      res.status(500).json({ error: err.message || "Không thể lấy thông tin thời tiết" });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
