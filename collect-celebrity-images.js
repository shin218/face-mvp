import "dotenv/config";
import { readFileSync, writeFileSync, mkdirSync, existsSync, createWriteStream, unlinkSync, statSync, openSync, readSync, closeSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";
import https from "https";
import http from "http";
import sharp from "sharp";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_PATH = path.join(__dirname, "data", "celebrities.json");
const IMAGES_DIR = path.join(__dirname, "data", "images");

const SERPAPI_KEY = process.env.SERPAPI_KEY;
const TARGET_VALID   = 3;    // 연예인당 확보할 유효 이미지 수
const CANDIDATE_FETCH = 15;  // SerpApi에서 한 번에 가져올 후보 수
const MIN_FILE_SIZE  = 10 * 1024;  // 10KB
const MIN_WIDTH      = 100;
const MIN_HEIGHT     = 100;
const REQUEST_DELAY_MS = 1000;

if (!SERPAPI_KEY) {
  console.error("SERPAPI_KEY 미설정");
  process.exit(1);
}

// 파일명으로 쓸 수 없는 문자 제거
function safeDirName(name) {
  return name.replace(/[<>:"/\\|?*\s]/g, "_");
}

// 이미지 확장자 추출 (.jpg 기본값)
function getExtension(url) {
  try {
    const pathname = new URL(url).pathname;
    const ext = path.extname(pathname).toLowerCase().split("?")[0];
    return [".jpg", ".jpeg", ".png", ".webp"].includes(ext) ? ext : ".jpg";
  } catch {
    return ".jpg";
  }
}

// 파일 앞 12바이트로 이미지 포맷 확인 (JPEG / PNG / WebP)
function hasImageMagicBytes(filepath) {
  const buf = Buffer.alloc(12);
  const fd = openSync(filepath, "r");
  readSync(fd, buf, 0, 12, 0);
  closeSync(fd);

  // JPEG: FF D8 FF
  if (buf[0] === 0xff && buf[1] === 0xd8 && buf[2] === 0xff) return true;
  // PNG: 89 50 4E 47 0D 0A 1A 0A
  if (buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) return true;
  // WebP: RIFF....WEBP
  if (buf[0] === 0x52 && buf[1] === 0x49 && buf[2] === 0x46 && buf[3] === 0x46 &&
      buf[8] === 0x57 && buf[9] === 0x45 && buf[10] === 0x42 && buf[11] === 0x50) return true;

  return false;
}

// 다운로드된 파일 유효성 검사 → 실패 시 파일 삭제 후 이유 반환
async function validateImage(filepath) {
  // 1) 파일 크기
  const { size } = statSync(filepath);
  if (size < MIN_FILE_SIZE) {
    unlinkSync(filepath);
    return `too small (${(size / 1024).toFixed(1)}KB < 10KB)`;
  }

  // 2) 매직 바이트
  if (!hasImageMagicBytes(filepath)) {
    unlinkSync(filepath);
    return "invalid image header";
  }

  // 3) 해상도 (sharp)
  try {
    const { width, height } = await sharp(filepath).metadata();
    if (!width || !height || width < MIN_WIDTH || height < MIN_HEIGHT) {
      unlinkSync(filepath);
      return `too small resolution (${width}x${height} < ${MIN_WIDTH}x${MIN_HEIGHT})`;
    }
  } catch {
    unlinkSync(filepath);
    return "sharp metadata error";
  }

  return null; // null = 유효
}

// URL에서 파일 다운로드 (리다이렉트 최대 3회)
function downloadFile(url, dest, redirectCount = 0) {
  return new Promise((resolve, reject) => {
    if (redirectCount > 3) return reject(new Error("Too many redirects"));

    const file = createWriteStream(dest);
    const protocol = url.startsWith("https://") ? https : http;

    const req = protocol.get(url, { timeout: 10000 }, (res) => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        file.close();
        downloadFile(res.headers.location, dest, redirectCount + 1)
          .then(resolve)
          .catch(reject);
        return;
      }
      if (res.statusCode !== 200) {
        file.close();
        return reject(new Error(`HTTP ${res.statusCode}`));
      }
      const ct = res.headers["content-type"] || "";
      if (!ct.startsWith("image/")) {
        file.close();
        return reject(new Error(`Not an image: ${ct}`));
      }
      res.pipe(file);
      file.on("finish", () => file.close(resolve));
    });

    req.on("error", (err) => { file.close(); reject(err); });
    req.on("timeout", () => { req.destroy(); file.close(); reject(new Error("Timeout")); });
  });
}

// SerpApi Google Images 검색
async function searchImages(query) {
  const params = new URLSearchParams({
    engine: "google_images",
    q: query,
    api_key: SERPAPI_KEY,
    safe: "active",
    ijn: "0",           // 첫 페이지
  });

  const url = `https://serpapi.com/search?${params}`;
  const res = await fetch(url);
  const data = await res.json();

  if (!res.ok) {
    const msg = data?.error || JSON.stringify(data).slice(0, 200);
    throw new Error(`SerpApi ${res.status}: ${msg}`);
  }

  const items = data.images_results || [];
  return items.slice(0, CANDIDATE_FETCH).map((item) => ({
    url: item.original,
    thumbnail: item.thumbnail,
    source: item.source,
  }));
}

async function collectImagesForCelebrity(celeb, dir) {
  const query = `${celeb.name_en} actress face`;
  const candidates = await searchImages(query);

  if (candidates.length === 0) return { valid: 0, tried: 0, rejected: 0, imagePaths: [] };

  const imagePaths = [];
  let validCount = 0;
  let tried = 0;
  let rejected = 0;

  for (const candidate of candidates) {
    if (validCount >= TARGET_VALID) break;

    const { url } = candidate;
    if (!url) continue;

    tried++;
    const ext = getExtension(url);
    // 저장 파일명은 유효 순번 기준
    const filename = `${String(validCount + 1).padStart(2, "0")}${ext}`;
    const dest = path.join(dir, filename);

    try {
      await downloadFile(url, dest);
    } catch {
      // 다운로드 자체 실패 → 건너뜀
      continue;
    }

    const reason = await validateImage(dest);
    if (reason) {
      rejected++;
      // dest는 validateImage 내부에서 이미 삭제됨
      continue;
    }

    imagePaths.push(`data/images/${safeDirName(celeb.name)}/${filename}`);
    validCount++;
  }

  return { valid: validCount, tried, rejected, imagePaths };
}

async function collectAll() {
  if (!existsSync(IMAGES_DIR)) mkdirSync(IMAGES_DIR, { recursive: true });

  const celebrities = JSON.parse(readFileSync(DB_PATH, "utf-8"));
  const toProcess = celebrities.filter(
    (c) => !c.image_paths || c.image_paths.length === 0
  );

  console.log(`\n총 ${celebrities.length}명 중 ${toProcess.length}명 이미지 수집 시작`);
  console.log(`저장 경로: ${IMAGES_DIR}`);
  console.log(`목표: 유효 ${TARGET_VALID}장 / 후보 최대 ${CANDIDATE_FETCH}장`);
  console.log(`유효 기준: ≥10KB, ≥${MIN_WIDTH}x${MIN_HEIGHT}px, 정상 이미지 헤더`);
  console.log(`API: SerpApi (google_images)\n`);
  console.log("=".repeat(60));

  let totalSuccess = 0;
  let totalFailed = 0;
  let totalImages = 0;

  for (let i = 0; i < toProcess.length; i++) {
    const celeb = toProcess[i];
    const idx = i + 1;
    const dir = path.join(IMAGES_DIR, safeDirName(celeb.name));
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

    process.stdout.write(`[${idx}/${toProcess.length}] ${celeb.name} (${celeb.name_en})... `);

    try {
      const { valid, tried, rejected, imagePaths } = await collectImagesForCelebrity(celeb, dir);

      const record = celebrities.find((c) => c.name === celeb.name);
      record.image_paths = imagePaths;

      console.log(`✓ 유효 ${valid}장  (시도 ${tried} / 거부 ${rejected})`);
      totalSuccess++;
      totalImages += valid;

      if (idx % 5 === 0) {
        writeFileSync(DB_PATH, JSON.stringify(celebrities, null, 2), "utf-8");
        console.log(`  → ${idx}명 완료, DB 중간 저장`);
      }
    } catch (err) {
      console.log(`✗ ${err.message}`);
      totalFailed++;
    }

    if (i < toProcess.length - 1) {
      await new Promise((r) => setTimeout(r, REQUEST_DELAY_MS));
    }
  }

  writeFileSync(DB_PATH, JSON.stringify(celebrities, null, 2), "utf-8");

  console.log("\n" + "=".repeat(60));
  console.log(`완료: 성공 ${totalSuccess}명 / 실패 ${totalFailed}명`);
  console.log(`총 수집 이미지: ${totalImages}장`);
  console.log(`DB 저장: ${DB_PATH}`);
}

collectAll().catch(console.error);
