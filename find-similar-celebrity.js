import "dotenv/config";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_PATH   = path.join(__dirname, "data", "celebrities.json");
const BASE_URL  = "https://api-us.faceplusplus.com/facepp/v3";

const API_KEY    = process.env.FACEPP_API_KEY;
const API_SECRET = process.env.FACEPP_API_SECRET;

const MIN_CONFIDENCE  = 60;   // 이 값 미만은 결과에서 제외
const MAX_RESULTS     = 3;    // 반환할 최대 매칭 수

// ── face_token → celebrity 역색인 빌드 ────────────────────────────────────────

function buildTokenIndex(celebrities) {
  const index = new Map(); // face_token → celebrity 객체
  for (const c of celebrities) {
    for (const token of c.face_tokens ?? []) {
      index.set(token, c);
    }
  }
  return index;
}

// ── Face++ API 공통 POST ──────────────────────────────────────────────────────

async function faceppPost(endpoint, fields) {
  const form = new FormData();
  form.append("api_key",    API_KEY);
  form.append("api_secret", API_SECRET);
  for (const [k, v] of Object.entries(fields)) form.append(k, String(v));

  const res  = await fetch(`${BASE_URL}${endpoint}`, { method: "POST", body: form });
  const data = await res.json();

  if (!res.ok) {
    throw new Error(`Face++ ${endpoint} ${res.status}: ${data.error_message ?? JSON.stringify(data)}`);
  }
  return data;
}

// ── Step 1: 고객 사진 → face_token ───────────────────────────────────────────

async function detectFace(imagePath) {
  const form = new FormData();
  form.append("api_key",    API_KEY);
  form.append("api_secret", API_SECRET);

  const buf  = readFileSync(imagePath);
  const ext  = path.extname(imagePath).toLowerCase().replace(".", "") || "jpeg";
  const mime = ext === "png" ? "image/png" : ext === "webp" ? "image/webp" : "image/jpeg";
  form.append("image_file", new Blob([buf], { type: mime }), path.basename(imagePath));

  const res  = await fetch(`${BASE_URL}/detect`, { method: "POST", body: form });
  const data = await res.json();

  if (!res.ok) {
    throw new Error(`Detect ${res.status}: ${data.error_message ?? JSON.stringify(data)}`);
  }
  if (!data.faces || data.faces.length === 0) {
    throw new Error("얼굴을 찾을 수 없습니다. 정면 얼굴 사진을 사용해주세요.");
  }

  // 얼굴이 여럿이면 가장 큰 bounding box를 사용
  const largest = data.faces.reduce((best, f) => {
    const area = f.face_rectangle.width * f.face_rectangle.height;
    return area > best.area ? { area, token: f.face_token } : best;
  }, { area: 0, token: null });

  return largest.token;
}

// ── Step 2: FaceSet 검색 ──────────────────────────────────────────────────────

async function searchInFaceSet(faceToken) {
  const data = await faceppPost("/search", {
    face_token:          faceToken,
    outer_id:            "celebrity_db",
    return_result_count: MAX_RESULTS,
  });

  // results 배열: [{ face_token, confidence, user_id }]
  return data.results ?? [];
}

// ── Step 3+4: 검색 결과 → 연예인 정보 매핑 ───────────────────────────────────

function mapResults(searchResults, tokenIndex) {
  const matches = [];

  for (const r of searchResults) {
    if (r.confidence < MIN_CONFIDENCE) continue;

    const celeb = tokenIndex.get(r.face_token);
    if (!celeb) continue; // DB에 없는 token이면 스킵

    // 같은 연예인이 이미 추가됐으면 최고 confidence로 dedup
    const existing = matches.find((m) => m.name === celeb.name);
    if (existing) {
      if (r.confidence > existing.confidence) existing.confidence = r.confidence;
      continue;
    }

    matches.push({
      name:       celeb.name,
      name_en:    celeb.name_en,
      confidence: Math.round(r.confidence * 10) / 10, // 소수 1자리
      face_shape: celeb.face_shape,
      country:    celeb.country,
      style:      celeb.style,
    });
  }

  // confidence 내림차순 정렬
  return matches.sort((a, b) => b.confidence - a.confidence).slice(0, MAX_RESULTS);
}

// ── 공개 API ──────────────────────────────────────────────────────────────────

/**
 * 고객 이미지 경로를 받아 닮은 연예인 목록을 반환
 * @param {string} imagePath  - 절대 또는 상대 경로
 * @returns {Promise<{ matches: Array, query_face_token: string }>}
 */
export async function findSimilarCelebrity(imagePath) {
  if (!API_KEY || !API_SECRET) {
    throw new Error("FACEPP_API_KEY / FACEPP_API_SECRET 미설정");
  }

  const celebrities = JSON.parse(readFileSync(DB_PATH, "utf-8"));
  const tokenIndex  = buildTokenIndex(celebrities);

  // 1) Detect
  const queryToken = await detectFace(imagePath);

  // 2) Search
  const rawResults = await searchInFaceSet(queryToken);

  // 3+4) 매핑
  const matches = mapResults(rawResults, tokenIndex);

  return { matches, query_face_token: queryToken };
}

// ── CLI 진입점 ────────────────────────────────────────────────────────────────
// 직접 실행: node find-similar-celebrity.js <이미지경로>

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const imagePath = process.argv[2];
  if (!imagePath) {
    console.error("사용법: node find-similar-celebrity.js <이미지경로>");
    process.exit(1);
  }

  const absPath = path.resolve(process.cwd(), imagePath);

  console.log(`\n이미지 분석 중: ${absPath}`);
  console.log("=".repeat(60));

  findSimilarCelebrity(absPath)
    .then(({ matches, query_face_token }) => {
      console.log(`Face token: ${query_face_token}\n`);

      if (matches.length === 0) {
        console.log(`매칭 결과 없음 (confidence ${MIN_CONFIDENCE}% 이상 없음)`);
        return;
      }

      console.log(`매칭 연예인 ${matches.length}명:\n`);
      for (const [i, m] of matches.entries()) {
        console.log(`${i + 1}. ${m.name} (${m.name_en})  confidence: ${m.confidence}%`);
        console.log(`   얼굴형: ${m.face_shape} | 국가: ${m.country}`);
        console.log(`   Vibe  : ${m.style.vibe}`);
        console.log(`   Makeup: ${m.style.makeup.join(", ")}`);
        console.log(`   Hair  : ${m.style.hair.join(", ")}`);
        console.log(`   Outfit: ${m.style.outfit.join(", ")}`);
        console.log();
      }

      console.log("JSON 결과:");
      console.log(JSON.stringify({ matches }, null, 2));
    })
    .catch((err) => {
      console.error("오류:", err.message);
      process.exit(1);
    });
}
