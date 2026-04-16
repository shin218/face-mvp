import "dotenv/config";
import { readFileSync, writeFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_PATH    = path.join(__dirname, "data", "celebrities.json");

const API_KEY    = process.env.FACEPP_API_KEY;
const API_SECRET = process.env.FACEPP_API_SECRET;
const BASE_URL   = "https://api-us.faceplusplus.com/facepp/v3";
const FACESET_ID = "celebrity_db";

// Face++ 무료 플랜: 1 QPS → 호출마다 1.1s 대기
const API_DELAY_MS = 1100;

// ── 유틸 ──────────────────────────────────────────────────────────────────────

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// multipart/form-data POST (파일 없이 필드만)
async function faceppPost(endpoint, fields) {
  const form = new FormData();
  form.append("api_key",    API_KEY);
  form.append("api_secret", API_SECRET);
  for (const [k, v] of Object.entries(fields)) form.append(k, v);

  const res  = await fetch(`${BASE_URL}${endpoint}`, { method: "POST", body: form });
  const data = await res.json();
  return { status: res.status, data };
}

// multipart/form-data POST with image_file (Buffer → Blob)
async function faceDetect(imagePath) {
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
    throw new Error(`Detect API ${res.status}: ${data.error_message || JSON.stringify(data)}`);
  }
  return data; // { faces: [{ face_token, ... }] }
}

// ── Step 1: FaceSet 생성 ──────────────────────────────────────────────────────

async function ensureFaceSet() {
  console.log(`FaceSet 확인 중 (outer_id: ${FACESET_ID})...`);
  const { status, data } = await faceppPost("/faceset/create", {
    outer_id:    FACESET_ID,
    display_name: "Celebrity DB",
  });
  await sleep(API_DELAY_MS);

  if (status === 200) {
    console.log(`  ✓ FaceSet 생성 완료 (faceset_token: ${data.faceset_token})\n`);
    return;
  }
  if (data.error_message === "FACESET_EXIST") {
    console.log("  ✓ FaceSet 이미 존재 — 건너뜀\n");
    return;
  }
  throw new Error(`FaceSet 생성 실패: ${data.error_message || JSON.stringify(data)}`);
}

// ── Step 2: 이미지 Detect → face_token 수집 ───────────────────────────────────

async function detectFacesFromImages(imagePaths) {
  const tokens = [];

  for (const relPath of imagePaths) {
    const absPath = path.join(__dirname, relPath);
    try {
      const result = await faceDetect(absPath);
      await sleep(API_DELAY_MS);

      if (!result.faces || result.faces.length === 0) {
        process.stdout.write("·"); // 얼굴 미검출
        continue;
      }
      // 한 장에 여러 얼굴이 있으면 첫 번째(가장 크거나 첫 번째) 사용
      tokens.push(result.faces[0].face_token);
      process.stdout.write("✓");
    } catch (err) {
      process.stdout.write("✗");
      // 개별 이미지 실패는 건너뜀
    }
  }

  return tokens;
}

// ── Step 3: FaceSet에 face_token 추가 (최대 5개씩 배치) ─────────────────────

async function addFacesToSet(faceTokens) {
  if (faceTokens.length === 0) return;

  // Face++ AddFace: face_tokens 파라미터는 콤마 구분, 한 번에 최대 5개
  const BATCH = 5;
  for (let i = 0; i < faceTokens.length; i += BATCH) {
    const chunk = faceTokens.slice(i, i + BATCH).join(",");
    const { status, data } = await faceppPost("/faceset/addface", {
      outer_id:    FACESET_ID,
      face_tokens: chunk,
    });
    await sleep(API_DELAY_MS);

    if (status !== 200) {
      throw new Error(`AddFace 실패: ${data.error_message || JSON.stringify(data)}`);
    }
  }
}

// ── 메인 ─────────────────────────────────────────────────────────────────────

async function buildFaceSet() {
  if (!API_KEY || !API_SECRET) {
    console.error("FACEPP_API_KEY 또는 FACEPP_API_SECRET 미설정 (.env 확인)");
    process.exit(1);
  }

  const celebrities = JSON.parse(readFileSync(DB_PATH, "utf-8"));

  const toProcess = celebrities.filter(
    (c) =>
      c.image_paths?.length > 0 &&       // 이미지가 있고
      (!c.face_tokens || c.face_tokens.length === 0) // face_tokens 미등록
  );

  console.log("=".repeat(60));
  console.log(`Face++ FaceSet 빌더`);
  console.log(`엔드포인트: ${BASE_URL}`);
  console.log(`총 ${celebrities.length}명 중 ${toProcess.length}명 처리 대상`);
  console.log("=".repeat(60) + "\n");

  // Step 1
  await ensureFaceSet();

  let successCount = 0;
  let failCount    = 0;
  let totalTokens  = 0;

  for (let i = 0; i < toProcess.length; i++) {
    const celeb = toProcess[i];
    const idx   = i + 1;

    process.stdout.write(
      `[${idx}/${toProcess.length}] ${celeb.name} (${celeb.name_en}) `
    );
    process.stdout.write(`[${celeb.image_paths.length}장] → `);

    try {
      // Step 2: Detect
      const faceTokens = await detectFacesFromImages(celeb.image_paths);
      process.stdout.write(` → `);

      if (faceTokens.length === 0) {
        console.log("얼굴 미검출, 건너뜀");
        // face_tokens를 빈 배열로 마킹 (재시도 방지)
        const record = celebrities.find((c) => c.name === celeb.name);
        record.face_tokens = [];
        failCount++;
      } else {
        // Step 3: FaceSet 등록
        await addFacesToSet(faceTokens);

        const record = celebrities.find((c) => c.name === celeb.name);
        record.face_tokens = faceTokens;

        console.log(`face_token ${faceTokens.length}개 등록`);
        successCount++;
        totalTokens += faceTokens.length;
      }
    } catch (err) {
      console.log(`\n  ✗ 오류: ${err.message}`);
      failCount++;
    }

    // 5명마다 중간 저장
    if (idx % 5 === 0) {
      writeFileSync(DB_PATH, JSON.stringify(celebrities, null, 2), "utf-8");
      console.log(`  ── ${idx}명 완료, DB 중간 저장 ──`);
    }
  }

  // 최종 저장
  writeFileSync(DB_PATH, JSON.stringify(celebrities, null, 2), "utf-8");

  console.log("\n" + "=".repeat(60));
  console.log(`완료: 성공 ${successCount}명 / 얼굴미검출·오류 ${failCount}명`);
  console.log(`총 등록 face_token: ${totalTokens}개`);
  console.log(`DB 저장: ${DB_PATH}`);
}

buildFaceSet().catch(console.error);
