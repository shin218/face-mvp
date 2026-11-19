// ── 케이스 분기 ───────────────────────────────────────────────────────────────
//
// Case A : 1위 confidence 70% 이상  → 높은 유사도 매칭
// Case B : 1위 confidence 60~70%   → 분위기 유사 참고
// Case C : 매칭 없음 (60% 미만 전부 필터됨) → 얼굴 데이터만으로 분석

function detectCase(matches) {
  if (!matches || matches.length === 0) return 'C';
  const top = matches[0].confidence;
  if (top >= 70) return 'A';
  if (top >= 60) return 'B';
  return 'C';
}

// ── 시스템 프롬프트 빌더 ──────────────────────────────────────────────────────

function buildVibeSystemPrompt(vibeCase, matches) {
  const base =
    '당신은 전문 이미지 컨설턴트입니다.\n' +
    '아래 데이터를 기반으로 이 사람의 잠재적 best look을 분석해주세요.';

  if (vibeCase === 'C') {
    return (
      base +
      '\n\n분석 케이스: C — 연예인 매칭 없음. 사진의 얼굴형·눈·피부 특징만으로 직접 분석합니다.'
    );
  }

  const caseLabel =
    vibeCase === 'A'
      ? 'A — confidence 70% 이상 (높은 유사도 매칭)'
      : 'B — confidence 60~70% (분위기 유사 참고)';

  const celebData = matches
    .map((m, i) => {
      const header =
        vibeCase === 'A'
          ? `[연예인 ${i + 1}] ${m.name} / ${m.name_en}  (${m.confidence}%)`
          : `[연예인 ${i + 1}] ${m.name} / ${m.name_en}  (비슷한 분위기)`;
      return [
        header,
        `얼굴형: ${m.face_shape}`,
        `헤어: ${m.style.hair.join(', ')}`,
        `메이크업: ${m.style.makeup.join(', ')}`,
        `의상: ${m.style.outfit.join(', ')}`,
        `분위기: ${m.style.vibe}`,
      ].join('\n');
    })
    .join('\n\n');

  return `${base}\n\n분석 케이스: ${caseLabel}\n\n[유사 연예인 스타일 데이터]\n${celebData}`;
}

// ── 유저 프롬프트 (공통) ──────────────────────────────────────────────────────
//
// 모든 케이스·모든 모델에 동일하게 적용.
// display 필드: 케이스에 맞는 한글 리포트 전문
// 나머지 필드: prompt-builder 가 인페인팅 프롬프트 조합에 사용

// 메이크업 상세 형식 블록 — 연예인 개별 분석과 최종 Best Look 모두에 동일하게 적용
const MAKEUP_FORMAT = `  메이크업:
    피부:
    - 베이스 타입: (글로우/매트/세미매트 등)
    - 커버리지: (가벼운/중간/풀커버)
    - 피부 표현: (물광/유리피부/새틴 등)

    눈썹:
    - 모양: (아치형/일자/자연스러운 등)
    - 두께: (가늘게/보통/두껍게)
    - 컬러: (자연갈색/다크브라운/블랙 등)
    - 텍스처: (파우더/펜슬/젤 등)

    눈:
    - 아이섀도: (컬러, 음영 방향, 블렌딩 방식)
    - 아이라인: (형태, 두께, 컬러)
    - 속눈썹: (길이/컬/볼륨 방향)
    - 언더라인: (있음/없음, 있으면 어떻게)
    - 애교살: (강조 여부)

    볼:
    - 블러셔: (컬러, 위치, 그라데이션 방식)
    - 컨투어: (있음/없음, 있으면 위치와 강도)
    - 하이라이터: (위치, 강도)

    입술:
    - 컬러: (구체적 색조)
    - 텍스처: (글로시/매트/세미매트/물들인듯)
    - 형태: (오버라인/자연스럽게/블러드립 등)
    - 립라인: (강조/자연스럽게)`;

const VIBE_USER_PROMPT = `사진 속 인물의 best look을 분석해주세요.

반드시 아래 JSON 형식으로만 응답하세요. 마크다운 코드블록 없이 순수 JSON만 반환하세요.

{
  "display": "전체 한글 리포트 텍스트 (줄바꿈은 \\\\n 사용)",
  "vibe_keywords":   ["영문 키워드1", "영문 키워드2", "영문 키워드3"],
  "hair_keywords":   ["영문 키워드1", "영문 키워드2"],
  "makeup_keywords": ["영문 키워드1", "영문 키워드2"],
  "clothes_keywords":["영문 키워드1", "영문 키워드2"]
}

[display 필드 작성 지침]

▸ Case A (70% 이상 매칭):

📌 닮은 연예인 분석

{연예인 수}명의 유사한 연예인이 발견되었습니다.

1. {name1} ({confidence1}%)
   얼굴형: {face_shape}
   시그니처 스타일:
   - 헤어: {hair_description}
${MAKEUP_FORMAT}
   - 의상: {outfit_description}
   - 전반적 분위기: {vibe}

2. {name2} ({confidence2}%)   ← 해당되는 경우만 출력
   (동일 구조)

---

🔍 공통점 분석
이 연예인들의 공통된 스타일 특징:
- {공통점 1}
- {공통점 2}
- {공통점 3}

---

✨ 최종 Best Look
위 공통점과 고객의 얼굴 특징을 결합한 최적 룩:

전체 분위기: {overall_vibe}
헤어: {hair_recommendation}
${MAKEUP_FORMAT}
의상: {outfit_recommendation}

🔍 Photoshop 작업 참고 검색어:
- "{search_keyword_1}"
- "{search_keyword_2}"
- "{search_keyword_3}"

🎨 Core Vibe: {core_vibe}
📌 Pinterest 키워드:
- Makeup: {makeup_pinterest}
- Hair: {hair_pinterest}
- Outfit: {outfit_pinterest}
- Overall: {overall_pinterest}

▸ Case B (60~70% 매칭):
confidence 수치 언급 없이 "OOO와 비슷한 분위기" 프레이밍으로 동일 구조 출력.
메이크업 상세 형식 동일하게 적용.

▸ Case C (매칭 없음):
연예인 언급 없이 얼굴형·눈·피부 특징 기반으로 직접 best look 추천.
공통점 분석 섹션 제외, 최종 Best Look 섹션부터 출력.
메이크업 상세 형식 동일하게 적용.

공통 조건:
- 전체 한글 출력
- 근거가 명확하게 보이도록
- 포토샵으로 바로 작업 가능할 만큼 구체적으로
- 과장된 표현 없이 담백하게`;

// ── JSON 파싱 헬퍼 ────────────────────────────────────────────────────────────

function parseAnalysisJson(raw) {
  let text = (raw || '').trim();
  text = text.replace(/^```json\s*/i, '').replace(/^```\s*/i, '').replace(/\s*```$/i, '');
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

async function analyzeWithGPT(imageBytes, imagePath, systemPrompt) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY 미설정');

  const { base64, mime } = imageToBase64(imageBytes, imagePath);

  const body = {
    model:      'gpt-4o',
    max_tokens: 2500,
    messages: [
      {
        role:    'system',
        content: systemPrompt,
      },
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: `data:${mime};base64,${base64}`, detail: 'low' } },
          { type: 'text',      text: VIBE_USER_PROMPT },
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

async function analyzeWithGemini(imageBytes, imagePath, systemPrompt) {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY 미설정');

  const { base64, mime } = imageToBase64(imageBytes, imagePath);

  const body = {
    system_instruction: { parts: [{ text: systemPrompt }] },
    contents: [
      {
        parts: [
          { inline_data: { mime_type: mime, data: base64 } },
          { text: VIBE_USER_PROMPT },
        ],
      },
    ],
    generationConfig: { maxOutputTokens: 1500, temperature: 0.3 },
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

async function analyzeWithClaude(imageBytes, imagePath, systemPrompt) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error('ANTHROPIC_API_KEY 미설정');

  const { base64, mime } = imageToBase64(imageBytes, imagePath);

  const body = {
    model:      'claude-sonnet-4-6',
    max_tokens: 2500,
    system:     systemPrompt,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image',  source: { type: 'base64', media_type: mime, data: base64 } },
          { type: 'text',   text: VIBE_USER_PROMPT },
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

export {
  detectCase,
  buildVibeSystemPrompt,
  analyzeWithGPT,
  analyzeWithGemini,
  analyzeWithClaude,
};
