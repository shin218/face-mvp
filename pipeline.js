'use strict';

// ── .env 로드 ─────────────────────────────────────────────────────────────────
const fs   = require('fs');
const path = require('path');

(function loadEnv() {
  try {
    const lines = fs.readFileSync(path.join(__dirname, '.env'), 'utf8').split(/\r?\n/);
    for (const line of lines) {
      const t = line.trim();
      if (!t || t.startsWith('#')) continue;
      const m = t.match(/^([A-Z0-9_]+)\s*=\s*(.+)$/);
      if (m && !process.env[m[1]]) process.env[m[1]] = m[2].trim();
    }
  } catch (_) {}
})();

const { analyzeWithGPT, analyzeWithGemini, analyzeWithClaude } = require('./llm-analyze');
const { buildHairPrompt, buildMakeupPrompt, buildClothesPrompt } = require('./prompt-builder');

// ── 설정 ─────────────────────────────────────────────────────────────────────

const FAL_API_KEY           = process.env.FAL_API_KEY;
const FAL_SEGMENT_URL       = 'https://fal.run/fal-ai/sam2/image';
const FAL_INPAINT_URL       = 'https://fal.run/fal-ai/flux-pro/v1/fill';
const INPAINT_STRENGTH      = 0.65;
const INPAINT_GUIDANCE      = 7;
const INPAINT_STEPS         = 25;

const SEGMENT_MAP = {
  hair:    { prompt: 'hair',     file: 'hair_mask.png'    },
  face:    { prompt: 'face',     file: 'face_mask.png'    },
  clothes: { prompt: 'clothing', file: 'clothes_mask.png' },
};

const MODELS = ['gpt', 'gemini', 'claude'];

// ── CLI 인자 검증 ─────────────────────────────────────────────────────────────

const BEFORE_PATH = process.argv[2];
if (!BEFORE_PATH) {
  console.error('Usage: node pipeline.js <before_image_path>');
  console.error('  예: node pipeline.js "C:\\Users\\user\\Desktop\\photo.png"');
  process.exit(1);
}
if (!fs.existsSync(BEFORE_PATH)) {
  console.error(`파일을 찾을 수 없습니다: ${BEFORE_PATH}`);
  process.exit(1);
}
if (!FAL_API_KEY) {
  console.error('FAL_API_KEY가 필요합니다 (.env 확인)');
  process.exit(1);
}

// ── 유틸 ─────────────────────────────────────────────────────────────────────

function log(prefix, msg) {
  const now = new Date().toTimeString().slice(0, 8);
  console.log(`[${now}][${prefix.padEnd(8)}] ${msg}`);
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function toDataUri(filePath) {
  const bytes = fs.readFileSync(filePath);
  const ext   = filePath.toLowerCase().endsWith('.png') ? 'png' : 'jpeg';
  return `data:image/${ext};base64,${bytes.toString('base64')}`;
}

function bytesToDataUri(bytes, filePath) {
  const ext = (filePath || '').toLowerCase().endsWith('.png') ? 'png' : 'jpeg';
  return `data:image/${ext};base64,${bytes.toString('base64')}`;
}

async function downloadBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`다운로드 실패 (${res.status}): ${url}`);
  return Buffer.from(await res.arrayBuffer());
}

// ── 마스크 생성 (fal-ai/sam2/image) ──────────────────────────────────────────

async function generateMask(imageDataUri, label, outputPath) {
  const seg = SEGMENT_MAP[label];
  log('MASK', `${label} 마스크 생성 (prompt: "${seg.prompt}")`);

  const res = await fetch(FAL_SEGMENT_URL, {
    method:  'POST',
    headers: { 'Authorization': `Key ${FAL_API_KEY}`, 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      image_url: imageDataUri,
      prompts:   [{ type: 'text', text: seg.prompt }],
    }),
  });

  const text = await res.text();
  if (!res.ok) throw new Error(`SAM2 오류 ${res.status}: ${text.slice(0, 300)}`);

  const json    = JSON.parse(text);
  const maskUrl = json?.image?.url ?? json?.images?.[0]?.url;
  if (!maskUrl) throw new Error(`마스크 URL 없음: ${text.slice(0, 300)}`);

  const buf = await downloadBuffer(maskUrl);
  fs.writeFileSync(outputPath, buf);
  log('MASK', `저장 완료: ${path.basename(outputPath)} (${Math.round(buf.length / 1024)} KB)`);
  return outputPath;
}

// ── 인페인팅 (fal-ai/flux-pro/v1/fill) ───────────────────────────────────────

async function inpaint({ imagePath, maskPath, prompt, outputPath, strength }) {
  const imageDataUri = imagePath.startsWith('data:') ? imagePath : toDataUri(imagePath);
  const maskDataUri  = toDataUri(maskPath);

  log('INPAINT', `prompt: ${prompt.slice(0, 90)}...`);

  const res = await fetch(FAL_INPAINT_URL, {
    method:  'POST',
    headers: { 'Authorization': `Key ${FAL_API_KEY}`, 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      image_url:           imageDataUri,
      mask_url:            maskDataUri,
      prompt,
      strength:            strength ?? INPAINT_STRENGTH,
      guidance_scale:      INPAINT_GUIDANCE,
      num_inference_steps: INPAINT_STEPS,
      output_format:       'png',
      num_images:          1,
    }),
  });

  const text = await res.text();
  if (!res.ok) throw new Error(`Flux 오류 ${res.status}: ${text.slice(0, 300)}`);

  const json     = JSON.parse(text);
  const imageUrl = json?.images?.[0]?.url ?? json?.image?.url;
  if (!imageUrl) throw new Error(`이미지 URL 없음: ${text.slice(0, 300)}`);

  const buf = await downloadBuffer(imageUrl);
  fs.writeFileSync(outputPath, buf);
  log('INPAINT', `저장 완료: ${path.basename(outputPath)} (${Math.round(buf.length / 1024)} KB)`);
  return outputPath;
}

// ── 모델별 파이프라인 ─────────────────────────────────────────────────────────
//
// hair → makeup → clothes 순서로 체이닝
// 각 단계의 output이 다음 단계의 input

async function runModelPipeline({ modelName, beforePath, masks, analysis, outputDir }) {
  const label = modelName.toUpperCase().padEnd(8);

  const hairPrompt    = buildHairPrompt(analysis);
  const makeupPrompt  = buildMakeupPrompt(analysis);
  const clothesPrompt = buildClothesPrompt(analysis);

  log(label, `hair prompt    : ${hairPrompt}`);
  log(label, `makeup prompt  : ${makeupPrompt}`);
  log(label, `clothes prompt : ${clothesPrompt}`);

  // 1단계: hair (before → hair.png)
  const hairOut = path.join(outputDir, 'hair.png');
  log(label, '[1/3] hair inpainting...');
  await inpaint({ imagePath: beforePath, maskPath: masks.hair, prompt: hairPrompt, outputPath: hairOut });

  // 2단계: makeup (hair.png → makeup.png)
  const makeupOut = path.join(outputDir, 'makeup.png');
  log(label, '[2/3] makeup inpainting...');
  await inpaint({ imagePath: hairOut, maskPath: masks.face, prompt: makeupPrompt, outputPath: makeupOut, strength: 0.55 });

  // 3단계: clothes (makeup.png → clothes.png)
  const clothesOut = path.join(outputDir, 'clothes.png');
  log(label, '[3/3] clothes inpainting...');
  await inpaint({ imagePath: makeupOut, maskPath: masks.clothes, prompt: clothesPrompt, outputPath: clothesOut });

  // final: clothes.png 복사
  const finalOut = path.join(outputDir, 'final.png');
  fs.copyFileSync(clothesOut, finalOut);
  log(label, `✓ 완료 → ${finalOut}`);

  return finalOut;
}

// ── 메인 ─────────────────────────────────────────────────────────────────────

async function main() {
  const sep = '='.repeat(60);
  console.log(sep);
  console.log('  Face Style Pipeline');
  console.log(`  Before: ${BEFORE_PATH}`);
  console.log(sep);

  // 출력 디렉토리 준비
  const outputRoot = path.join(__dirname, 'output');
  const masksDir   = path.join(outputRoot, 'masks');
  ensureDir(masksDir);
  for (const m of MODELS) ensureDir(path.join(outputRoot, m));

  // Before 이미지 로드
  const beforeBytes   = fs.readFileSync(BEFORE_PATH);
  const beforeDataUri = bytesToDataUri(beforeBytes, BEFORE_PATH);

  // ───────────────────────────────────────────────────────────────────────────
  // STEP 1: 마스크 생성 (hair / face / clothes)
  // ───────────────────────────────────────────────────────────────────────────
  console.log('\n[STEP 1] 마스크 생성');
  const masks = {};

  for (const label of ['hair', 'face', 'clothes']) {
    const maskPath = path.join(masksDir, SEGMENT_MAP[label].file);
    try {
      masks[label] = await generateMask(beforeDataUri, label, maskPath);
    } catch (e) {
      console.error(`[MASK] ${label} 마스크 생성 실패: ${e.message}`);
      process.exit(1);
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STEP 2: LLM 분석 (GPT / Gemini / Claude 병렬)
  // ───────────────────────────────────────────────────────────────────────────
  console.log('\n[STEP 2] LLM 분석 (3개 모델 병렬)');

  const [gptRes, geminiRes, claudeRes] = await Promise.allSettled([
    analyzeWithGPT(beforeBytes, BEFORE_PATH),
    analyzeWithGemini(beforeBytes, BEFORE_PATH),
    analyzeWithClaude(beforeBytes, BEFORE_PATH),
  ]);

  const analyses = {};
  const resultMap = { gpt: gptRes, gemini: geminiRes, claude: claudeRes };

  for (const model of MODELS) {
    const r = resultMap[model];
    if (r.status === 'fulfilled') {
      analyses[model] = r.value;
      const outPath = path.join(outputRoot, model, 'analysis.json');
      fs.writeFileSync(outPath, JSON.stringify(r.value, null, 2), 'utf8');
      log(model.toUpperCase(), `분석 완료 → analysis.json 저장`);
      log(model.toUpperCase(), `  vibe    : ${(r.value.vibe_keywords || []).join(', ')}`);
      log(model.toUpperCase(), `  hair    : ${(r.value.hair_keywords || []).join(', ')}`);
      log(model.toUpperCase(), `  makeup  : ${(r.value.makeup_keywords || []).join(', ')}`);
      log(model.toUpperCase(), `  clothes : ${(r.value.clothes_keywords || []).join(', ')}`);
      log(model.toUpperCase(), `  summary : ${r.value.summary}`);
    } else {
      log(model.toUpperCase(), `분석 실패: ${r.reason?.message || String(r.reason)}`);
    }
  }

  // ───────────────────────────────────────────────────────────────────────────
  // STEP 3: 모델별 inpainting 파이프라인 (순차 실행)
  // ───────────────────────────────────────────────────────────────────────────
  console.log('\n[STEP 3] 모델별 inpainting (hair → makeup → clothes)');

  for (const model of MODELS) {
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`  [${model.toUpperCase()}]`);
    console.log('─'.repeat(60));

    if (!analyses[model]) {
      log(model.toUpperCase(), 'LLM 분석 실패로 스킵');
      continue;
    }

    const outputDir = path.join(outputRoot, model);
    try {
      await runModelPipeline({
        modelName: model,
        beforePath: BEFORE_PATH,
        masks,
        analysis: analyses[model],
        outputDir,
      });
    } catch (e) {
      log(model.toUpperCase(), `파이프라인 실패: ${e.message}`);
    }
  }

  // ── 완료 요약 ─────────────────────────────────────────────────────────────
  console.log('\n' + sep);
  console.log('  완료 요약');
  console.log(sep);
  for (const m of MODELS) {
    const final = path.join(outputRoot, m, 'final.png');
    const ok    = fs.existsSync(final);
    console.log(`  output/${m}/final.png  ${ok ? '✓ 생성됨' : '✗ 실패'}`);
  }
  console.log(sep);
}

main().catch(e => {
  console.error('\n[치명적 오류]', e.message);
  process.exit(1);
});
