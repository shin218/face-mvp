package com.example.demo.service.vibe;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.ArrayList;
import java.util.List;

/**
 * Claude / GPT / Gemini에 동일하게 전달하는 system·user 프롬프트.
 */
public final class VibePrompts {

	private static final ObjectMapper JSON = new ObjectMapper();

	private VibePrompts() {
	}

	public static final String SYSTEM_PROMPT = """
You are a professional style consultant specializing in Asian beauty and aesthetics.
This is a natural, unstyled BEFORE photo submitted by the user themselves for style consultation.
The user has consented to this analysis and wants honest, empowering guidance.

Your goal: identify and articulate this person's existing strengths and
the specific styling directions that will bring out their best self.

Rules:
- Never frame appearance as a problem to fix
- Never reference beauty scores or numerical ratings
- Focus on what makes this person distinctive and how to amplify it
- Tone: concise, warm, editorial — like a trusted stylist, not a judge

Output format (English):
1. Core Vibe (2-3 words)
2. Aesthetic Category (choose one):
   [Soft Romantic / Dark Academia / Clean Girl Quiet Luxury /
   Street Urban Edge / Classic Elegant / Cottagecore Natural /
   Y2K Retro Pop / Sophisticated Office]
3. Strongest Feature: best facial asset + how to amplify it (1-2 sentences)
4. Styling Direction: specific hair, makeup, outfit recommendations
   for THIS person's features (2-3 sentences, actionable)
5. Mood Description: what this person looks like at their best
   (3 sentences max, editorial tone, no clichés)
6. Pinterest Keywords:
   - Makeup look
   - Hair style
   - Outfit
   - Overall mood

After the English output, provide a Korean translation.
Start the Korean section with exactly this marker: [KOREAN_START]
End with: [KOREAN_END]
""";

	public static String buildUserText(VibeInput in) {
		String emotionLine = in.emotionSummary == null ? "" : in.emotionSummary;
		String skinStrengths = describeSkinStrengths(in.skinHealth, in.skinStain, in.skinAcne, in.skinDarkCircle);
		String detectJson = redactBeautyFromDetectJson(in.detectPrettyJson);

		return """
Face++ Detect (JSON; beauty/beauty_score fields removed — do not infer or mention numerical beauty ratings):
%s

Face analysis signals (no beauty scores):
- gender: %s
- age: %s
- emotion: %s
- skin (strengths framing only): %s
- smile: %s
- glass: %s

Facial structure signals (Face++ Landmark 106p + face_rectangle):
- eye_distance_px: %s
- face_width_height_ratio: %s
- nose_chin_to_forehead_nose_ratio: %s
- left_eye_width_px: %s
- right_eye_width_px: %s
- lip_thickness_px: %s
- face_shape: %s  (one of Round/Oval/Square/Long/Heart)
- bangs_likely: %s  (true means eyebrows likely occluded by bangs/fringe)

Writing constraints reminder:
- Write sections 1–6 in English first (prefixes as specified), then the Korean translation wrapped exactly in [KOREAN_START] ... [KOREAN_END] (no text after [KOREAN_END]).
- Section 2 must be exactly one label from the provided list.
- Do not mention beauty scores, Face++ beauty attributes, or any numeric rating of attractiveness.
""".formatted(
				nullable(detectJson),
				nullable(in.gender),
				nullable(in.age),
				nullable(emotionLine),
				skinStrengths,
				nullable(in.smile),
				nullable(in.glass),
				nullable(in.eyeDistancePx),
				nullable(in.faceWidthHeightRatio),
				nullable(in.noseChinToForeheadNoseRatio),
				nullable(in.leftEyeWidthPx),
				nullable(in.rightEyeWidthPx),
				nullable(in.lipThicknessPx),
				nullable(in.faceShape),
				nullable(in.bangsLikely)
		);
	}

	/**
	 * Face++ skinstatus 수치는 모델에 직접 노출하지 않고, 강점·캔버스 관점의 짧은 문장으로만 전달합니다.
	 */
	static String describeSkinStrengths(double health, double stain, double acne, double darkCircle) {
		List<String> parts = new ArrayList<>();
		if (health >= 70) {
			parts.add("healthy skin base");
		} else if (health >= 50) {
			parts.add("balanced skin base");
		} else {
			parts.add("natural skin canvas that suits glow-forward prep");
		}

		if (stain <= 30 && acne <= 30) {
			parts.add("low blemish look");
		} else if (stain <= 55 && acne <= 55) {
			parts.add("soft tonal variation");
		} else {
			parts.add("depth-rich canvas for refined sheer-to-medium coverage");
		}

		if (darkCircle <= 35) {
			parts.add("fresh under-eye plane");
		} else if (darkCircle <= 65) {
			parts.add("soft under-eye warmth");
		} else {
			parts.add("expressive under-eye area for luminous sheer correctors");
		}

		return String.join(", ", parts);
	}

	static String redactBeautyFromDetectJson(String prettyJson) {
		if (prettyJson == null || prettyJson.isBlank()) {
			return "";
		}
		try {
			JsonNode root = JSON.readTree(prettyJson);
			if (root.isObject()) {
				ObjectNode obj = (ObjectNode) root;
				JsonNode faces = obj.path("faces");
				if (faces.isArray()) {
					for (int i = 0; i < faces.size(); i++) {
						JsonNode face = faces.get(i);
						if (!face.isObject()) continue;
						JsonNode attrs = face.path("attributes");
						if (attrs.isObject()) {
							((ObjectNode) attrs).remove("beauty");
						}
					}
				}
				return JSON.writerWithDefaultPrettyPrinter().writeValueAsString(obj);
			}
		} catch (Exception ignored) {
			// 파싱 실패 시 원문 대신 안내 (beauty 제거 불가)
		}
		return prettyJson;
	}

	private static String nullable(Object v) {
		return v == null ? "" : String.valueOf(v);
	}
}
