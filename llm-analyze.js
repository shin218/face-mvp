'use strict';

// ── 분석 프롬프트 ─────────────────────────────────────────────────────────────
//
// 목적: 사진 속 인물의 최적 스타일링 방향 추출
// 방식: 외모를 모욕하지 않고, "best vibe / flattering features / styling direction" 중심
// 출력: 반드시 JSON만 반환 (마크다운 코드블록 없이)
//
const ANALYSIS_PROMPT = `You are a professional fashion stylist and beauty consultant. Analyze this photo and identify the best styling direction for this person.

Return ONLY a valid JSON object. No explanation, no markdown, no code blocks. Just the raw JSON.

{
  "vibe_keywords": ["keyword1", "keyword2", "keyword3"],
  "hair_keywords": ["keyword1", "keyword2", "keyword3"],
  "makeup_keywords": ["keyword1", "keyword2", "keyword3"],
  "clothes_keywords": ["keyword1", "keyword2", "keyword3"],
  "summary": "one sentence describing the ideal styling direction"
}

Rules:
- vibe_keywords: 3-5 adjectives capturing the person's best aesthetic vibe (e.g. "soft romantic", "editorial chic", "fresh minimal", "warm natural")
- hair_keywords: 2-4 words describing flattering hair style/color/texture (e.g. "warm brunette waves", "sleek blunt bob", "soft highlights")
- makeup_keywords: 2-4 words for a flattering makeup look (e.g. "dewy glow", "warm earth tones", "fresh natural skin", "subtle definition")
- clothes_keywords: 2-4 words for clothing that would flatter this person (e.g. "soft ivory tones", "minimalist tailored", "flowy feminine", "monochrome chic")
- summary: positive, encouraging one sentence about their best styling direction
- Focus entirely on flattering features and styling improvement, never on flaws
- Return ONLY the JSON object, nothing else`;

// ── JSON 파싱 헬퍼 ────────────────────────────────────────────────────────────

function parseAnalysisJson(raw) {
  let text = (raw || '').trim();
  // 마크다운 코드블록 제거
  text = text.replace(/^```json\s*/i, '').replace(/^```\s*/i, '').replace(/\s*```$/i, '');
  // JSON 객체 추출
  const match = text.match(/\{[\s\S]+\}/);
  if (!match) throw new Error(`JSON 객체를 찾을 수 없습니다. 원본:\n${text.slice(0, 300)}`);
  return JSON.parse(match[0]);
}

function imageToBase64(imageBytes, imagePath) {
  const ext  = (imagePath || '').toLowerCase().endsWith('.png') ? 'png' : 'jpeg';
  const mime = `image/${ext}`;
  return { base64: imageBytes.toString('base64'), mime };
}

// ── GPT-4o ───────────────────────────────────────────────────────────────────

async function analyzeWithGPT(imageBytes, imagePath) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY 미설정');

  const { base64, mime } = imageToBase64(imageBytes, imagePath);

  const body = {
    model: 'gpt-4o',
    max_tokens: 512,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: `data:${mime};base64,${base64}`, detail: 'low' } },
          { type: 'text',      text: ANALYSIS_PROMPT },
        ],
      },
    ],
  };

  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method:  'POST',
    headers: { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  });

  const text = await res.text();
  if (!res.ok) throw new Error(`GPT API 오류 ${res.status}: ${text.slice(0, 300)}`);

  const json    = JSON.parse(text);
  const content = json?.choices?.[0]?.message?.content;
  if (!content) throw new Error(`GPT 응답 없음: ${text.slice(0, 300)}`);

  return parseAnalysisJson(content);
}

// ── Gemini ───────────────────────────────────────────────────────────────────

async function analyzeWithGemini(imageBytes, imagePath) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY 미설정');

  const { base64, mime } = imageToBase64(imageBytes, imagePath);

  const body = {
    contents: [
      {
        parts: [
          { inline_data: { mime_type: mime, data: base64 } },
          { text: ANALYSIS_PROMPT },
        ],
      },
    ],
    generationConfig: { maxOutputTokens: 1024, temperature: 0.3 },
  };

  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=${apiKey}`;

  const res = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  });

  const text = await res.text();
  if (!res.ok) throw new Error(`Gemini API 오류 ${res.status}: ${text.slice(0, 300)}`);

  const json    = JSON.parse(text);
  const content = json?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!content) throw new Error(`Gemini 응답 없음: ${text.slice(0, 300)}`);

  return parseAnalysisJson(content);
}

// ── Claude ───────────────────────────────────────────────────────────────────

/**
 * 닮은 연예인 매칭 결과로 system prompt 컨텍스트 블록 생성
 * @param {Array} matches  - findSimilarCelebrity()가 반환한 matches 배열
 * @returns {string}
 */
function buildCelebrityContext(matches) {
  if (!matches || matches.length === 0) return '';

  const blocks = matches.map((m) =>
    `The customer resembles ${m.name_en} (confidence: ${m.confidence}%).
Reference their signature style as a base:
Makeup: ${m.style.makeup.join(', ')}
Hair: ${m.style.hair.join(', ')}
Outfit: ${m.style.outfit.join(', ')}
Vibe: ${m.style.vibe}
Build upon this foundation to suggest the customer's optimal transformation.`
  );

  return blocks.join('\n\n');
}

/**
 * @param {Buffer}  imageBytes
 * @param {string}  imagePath
 * @param {Array}   [celebrityMatches]  - findSimilarCelebrity().matches (optional)
 */
async function analyzeWithClaude(imageBytes, imagePath, celebrityMatches) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error('ANTHROPIC_API_KEY 미설정');

  const { base64, mime } = imageToBase64(imageBytes, imagePath);

  // 연예인 컨텍스트가 있으면 system prompt로 주입
  const celebrityContext = buildCelebrityContext(celebrityMatches);
  const systemPrompt = celebrityContext
    ? `You are a professional fashion stylist and beauty consultant with deep knowledge of celebrity aesthetics.\n\n${celebrityContext}`
    : 'You are a professional fashion stylist and beauty consultant.';

  const body = {
    model:      'claude-sonnet-4-6',
    max_tokens: 512,
    system:     systemPrompt,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image',  source: { type: 'base64', media_type: mime, data: base64 } },
          { type: 'text',   text: ANALYSIS_PROMPT },
        ],
      },
    ],
  };

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method:  'POST',
    headers: {
      'x-api-key':         apiKey,
      'anthropic-version': '2023-06-01',
      'Content-Type':      'application/json',
    },
    body: JSON.stringify(body),
  });

  const text = await res.text();
  if (!res.ok) throw new Error(`Claude API 오류 ${res.status}: ${text.slice(0, 300)}`);

  const json    = JSON.parse(text);
  const content = json?.content?.[0]?.text;
  if (!content) throw new Error(`Claude 응답 없음: ${text.slice(0, 300)}`);

  return parseAnalysisJson(content);
}

module.exports = { analyzeWithGPT, analyzeWithGemini, analyzeWithClaude };
