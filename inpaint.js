'use strict';

const fs = require('fs');

// ── 설정 ─────────────────────────────────────────────────────────────────────
const FAL_MODEL    = 'fal-ai/flux-pro/v1/fill';
const FAL_ENDPOINT = `https://fal.run/${FAL_MODEL}`;

const STRENGTH            = 0.5;
const GUIDANCE_SCALE      = 6;
const NUM_INFERENCE_STEPS = 20;

// prompt에 반드시 포함할 접두어
const REQUIRED_PREFIX = 'same person, preserve identity, realistic, natural lighting, ';
// ─────────────────────────────────────────────────────────────────────────────

const FAL_API_KEY  = process.env.FAL_API_KEY;
const IMAGE_PATH   = process.argv[2];
const MASK_PATH    = process.argv[3];
const USER_PROMPT  = process.argv[4];

if (!FAL_API_KEY) {
  console.error('FAL_API_KEY 환경변수가 필요합니다.');
  console.error('  export FAL_API_KEY=<your_key>');
  process.exit(1);
}
if (!IMAGE_PATH || !MASK_PATH || !USER_PROMPT) {
  console.error('Usage: node inpaint.js <image_path> <mask_path> "<prompt>"');
  console.error('  예: node inpaint.js photo.jpg hair_mask.png "glossy black hair"');
  process.exit(1);
}

// ── 헬퍼 ─────────────────────────────────────────────────────────────────────

function toDataUri(filePath) {
  const bytes = fs.readFileSync(filePath);
  const mime  = filePath.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';
  return `data:${mime};base64,${bytes.toString('base64')}`;
}

async function downloadBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Download failed (${res.status}): ${url}`);
  return Buffer.from(await res.arrayBuffer());
}

function extractImageUrl(json) {
  if (json.images?.[0]?.url)          return json.images[0].url;
  if (json.image?.url)                return json.image.url;
  if (typeof json.images?.[0] === 'string') return json.images[0];
  return null;
}

// ── 메인 ─────────────────────────────────────────────────────────────────────

async function main() {
  for (const p of [IMAGE_PATH, MASK_PATH]) {
    if (!fs.existsSync(p)) {
      console.error(`파일을 찾을 수 없습니다: ${p}`);
      process.exit(1);
    }
  }

  const fullPrompt = REQUIRED_PREFIX + USER_PROMPT;

  console.log(`이미지        : ${IMAGE_PATH}`);
  console.log(`마스크        : ${MASK_PATH}`);
  console.log(`프롬프트      : ${fullPrompt}`);
  console.log(`모델          : ${FAL_MODEL}`);
  console.log(`strength      : ${STRENGTH}`);
  console.log(`guidance_scale: ${GUIDANCE_SCALE}`);
  console.log(`steps         : ${NUM_INFERENCE_STEPS}`);

  const imageDataUri = toDataUri(IMAGE_PATH);
  const maskDataUri  = toDataUri(MASK_PATH);

  const body = {
    image_url:           imageDataUri,
    mask_url:            maskDataUri,
    prompt:              fullPrompt,
    strength:            STRENGTH,
    guidance_scale:      GUIDANCE_SCALE,
    num_inference_steps: NUM_INFERENCE_STEPS,
    output_format:       'png',
    num_images:          1,
  };

  console.log('\nfal.ai inpainting 요청 중...');

  const res = await fetch(FAL_ENDPOINT, {
    method:  'POST',
    headers: {
      'Authorization': `Key ${FAL_API_KEY}`,
      'Content-Type':  'application/json',
    },
    body: JSON.stringify(body),
  });

  const text = await res.text();

  if (!res.ok) {
    throw new Error(`fal.ai API 오류 ${res.status}: ${text}`);
  }

  const json = JSON.parse(text);
  console.log('응답:', JSON.stringify(json).slice(0, 400));

  const imageUrl = extractImageUrl(json);
  if (!imageUrl) {
    throw new Error(`이미지 URL 추출 실패. 전체 응답: ${JSON.stringify(json)}`);
  }

  console.log(`\n이미지 URL: ${imageUrl}`);
  const buf = await downloadBuffer(imageUrl);
  fs.writeFileSync('result.png', buf);
  console.log(`저장 완료: result.png (${Math.round(buf.length / 1024)} KB)`);
}

main().catch(e => {
  console.error('\n오류:', e.message);
  process.exit(1);
});
