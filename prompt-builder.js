const HAIR_PREFIX    = 'same person, preserve identity, realistic, natural lighting, do not change face, ';
const MAKEUP_PREFIX  = 'same person, preserve identity, subtle makeup, realistic skin texture, natural lighting, ';
const CLOTHES_PREFIX = 'same person, preserve identity, same pose, realistic fabric, natural lighting, ';

function joinKeywords(arr) {
  return (arr || []).filter(Boolean).join(', ');
}

function buildHairPrompt(analysis) {
  const parts = [
    joinKeywords(analysis.hair_keywords),
    joinKeywords(analysis.vibe_keywords),
  ].filter(Boolean).join(', ');
  return HAIR_PREFIX + parts;
}

function buildMakeupPrompt(analysis) {
  const parts = [
    joinKeywords(analysis.makeup_keywords),
    joinKeywords(analysis.vibe_keywords),
  ].filter(Boolean).join(', ');
  return MAKEUP_PREFIX + parts;
}

function buildClothesPrompt(analysis) {
  const parts = [
    joinKeywords(analysis.clothes_keywords),
    joinKeywords(analysis.vibe_keywords),
  ].filter(Boolean).join(', ');
  return CLOTHES_PREFIX + parts;
}

export { buildHairPrompt, buildMakeupPrompt, buildClothesPrompt };
