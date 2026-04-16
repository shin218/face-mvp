import "dotenv/config";
import Anthropic from "@anthropic-ai/sdk";
import { readFileSync, writeFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_PATH = path.join(__dirname, "data", "celebrities.json");

const client = new Anthropic();

const SYSTEM_PROMPT = `You are a professional stylist and fashion expert specializing in Asian and international celebrity aesthetics.
Based on the given celebrity's most iconic and well-known looks, provide their signature style in keywords.
Always respond with valid JSON only, no markdown, no extra text.`;

async function getStyleForCelebrity(nameEn) {
  const userPrompt = `Analyze ${nameEn}'s most iconic looks and provide their signature style.
Return ONLY this JSON (no markdown, no extra text):
{
  "makeup": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "hair": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "outfit": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "vibe": "one sentence describing their overall aesthetic"
}`;

  const response = await client.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 400,
    system: SYSTEM_PROMPT,
    messages: [{ role: "user", content: userPrompt }],
  });

  let text = response.content[0].text.trim();
  // Strip markdown code fences if present
  text = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/, "").trim();
  return JSON.parse(text);
}

async function buildCelebrityDB() {
  const celebrities = JSON.parse(readFileSync(DB_PATH, "utf-8"));

  const toProcess = celebrities.filter(
    (c) =>
      c.style.makeup.length === 0 ||
      c.style.hair.length === 0 ||
      c.style.vibe === ""
  );

  console.log(
    `\n총 ${celebrities.length}명 중 ${toProcess.length}명 스타일 생성 시작\n`
  );
  console.log("=".repeat(60));

  let success = 0;
  let failed = 0;

  for (let i = 0; i < toProcess.length; i++) {
    const celeb = celebrities.find((c) => c.name_en === toProcess[i].name_en);
    const idx = i + 1;

    process.stdout.write(
      `[${idx}/${toProcess.length}] ${celeb.name} (${celeb.name_en})... `
    );

    try {
      const style = await getStyleForCelebrity(celeb.name_en);

      celeb.style.makeup = style.makeup;
      celeb.style.hair = style.hair;
      celeb.style.outfit = style.outfit;
      celeb.style.vibe = style.vibe;

      console.log(`✓`);
      console.log(`  vibe: ${style.vibe}`);

      success++;

      // 5개마다 중간 저장
      if (idx % 5 === 0) {
        writeFileSync(DB_PATH, JSON.stringify(celebrities, null, 2), "utf-8");
        console.log(`  → ${idx}명 완료, 중간 저장`);
      }

      // Rate limit 방지 (300ms 딜레이)
      await new Promise((r) => setTimeout(r, 300));
    } catch (err) {
      console.log(`✗ 실패: ${err.message}`);
      failed++;
    }
  }

  // 최종 저장
  writeFileSync(DB_PATH, JSON.stringify(celebrities, null, 2), "utf-8");

  console.log("\n" + "=".repeat(60));
  console.log(`완료: 성공 ${success}명 / 실패 ${failed}명`);
  console.log(`저장 위치: ${DB_PATH}`);
}

buildCelebrityDB().catch(console.error);
