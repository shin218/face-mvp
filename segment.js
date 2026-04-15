'use strict';

// ── 모델 설정 ─────────────────────────────────────────────────────────────────
//
// fal.ai SAM2 (Segment Anything Model 2) - text prompt 기반 세그멘테이션
// 엔드포인트: https://fal.run/fal-ai/sam2/image
// 입력: { image_url, prompts: [{type: "text", text: "..."}] }
// 출력: { image: { url: "..." } }  → SAM 마스크 이미지 (흰=마스크, 검=배경)
//
const FAL_MODEL    = 'fal-ai/sam2/image';
const FAL_ENDPOINT = `https://fal.run/${FAL_MODEL}`;
// ─────────────────────────────────────────────────────────────────────────────

const fs = require('fs');

// 레이블 → SAM2 텍스트 프롬프트 / 출력 파일명 매핑
const LABEL_MAP = {
  hair:    { prompt: 'hair',     out: 'hair_mask.png'    },
  face:    { prompt: 'face',     out: 'face_mask.png'    },
  clothes: { prompt: 'clothing', out: 'clothes_mask.png' },
};

const FAL_API_KEY = process.env.FAL_API_KEY;
const IMAGE_PATH  = process.argv[2];
const LABEL       = process.argv[3];

if (!FAL_API_KEY) {
  console.error('FAL_API_KEY 환경변수가 필요합니다.');
  console.error('  export FAL_API_KEY=<your_key>');
  process.exit(1);
}
if (!IMAGE_PATH || !LABEL) {
  console.error('Usage: node segment.js <image_path> <hair|face|clothes>');
  process.exit(1);
}
if (!LABEL_MAP[LABEL]) {
  console.error(`지원하지 않는 label: "${LABEL}". hair / face / clothes 중 하나를 입력하세요.`);
  process.exit(1);
}

async function downloadBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`다운로드 실패 (${res.status}): ${url}`);
  return Buffer.from(await res.arrayBuffer());
}

async function main() {
  if (!fs.existsSync(IMAGE_PATH)) {
    console.error(`파일을 찾을 수 없습니다: ${IMAGE_PATH}`);
    process.exit(1);
  }

  const seg       = LABEL_MAP[LABEL];
  const imgBytes  = fs.readFileSync(IMAGE_PATH);
  const ext       = IMAGE_PATH.toLowerCase().endsWith('.png') ? 'png' : 'jpeg';
  const dataUri   = `data:image/${ext};base64,${imgBytes.toString('base64')}`;

  console.log(`이미지  : ${IMAGE_PATH} (${Math.round(imgBytes.length / 1024)} KB)`);
  console.log(`모델    : ${FAL_MODEL}`);
  console.log(`레이블  : ${LABEL} (prompt: "${seg.prompt}")`);
  console.log('요청 중...');

  const res = await fetch(FAL_ENDPOINT, {
    method:  'POST',
    headers: {
      'Authorization': `Key ${FAL_API_KEY}`,
      'Content-Type':  'application/json',
    },
    body: JSON.stringify({
      image_url: dataUri,
      prompts:   [{ type: 'text', text: seg.prompt }],
    }),
  });

  const text = await res.text();

  if (!res.ok) {
    throw new Error(`fal.ai API 오류 ${res.status}: ${text}`);
  }

  const json = JSON.parse(text);
  console.log('응답:', JSON.stringify(json).slice(0, 300));

  const maskUrl = json?.image?.url ?? json?.images?.[0]?.url ?? null;
  if (!maskUrl) {
    throw new Error(`마스크 URL 추출 실패. 전체 응답: ${JSON.stringify(json)}`);
  }

  console.log(`마스크 URL: ${maskUrl}`);
  const maskBuf = await downloadBuffer(maskUrl);
  fs.writeFileSync(seg.out, maskBuf);
  console.log(`저장 완료: ${seg.out} (${Math.round(maskBuf.length / 1024)} KB)`);
}

main().catch(e => {
  console.error('오류:', e.message);
  process.exit(1);
});
