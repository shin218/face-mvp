package com.example.demo.service;

import com.example.demo.service.vibe.VibeInput;
import com.example.demo.service.vibe.VibePrompts;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.netty.http.client.HttpClient;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Base64;
import java.util.concurrent.TimeoutException;

@Service
public class ClaudeVibeService {

	private static final String CLAUDE_MESSAGES_URL = "https://api.anthropic.com/v1/messages";
	private static final Duration RESPONSE_TIMEOUT = Duration.ofSeconds(180);
	private static final int MAX_IMAGE_DIM = 1568;

	private final WebClient webClient;
	private final ObjectMapper objectMapper;
	private final String apiKey;
	private final String model;

	public ClaudeVibeService(
			WebClient.Builder webClientBuilder,
			ObjectMapper objectMapper,
			@Value("${claude.api.key:}") String apiKey,
			@Value("${claude.api.model:claude-sonnet-4-6}") String model
	) {
		this.objectMapper = objectMapper;
		this.apiKey = apiKey == null ? "" : apiKey.trim();
		this.model = model;

		HttpClient httpClient = HttpClient.create()
				.responseTimeout(RESPONSE_TIMEOUT);

		this.webClient = webClientBuilder
				.clientConnector(new ReactorClientHttpConnector(httpClient))
				.build();
	}

	public boolean isConfigured() {
		return apiKey != null && !apiKey.isBlank();
	}

	public String analyzeVibe(VibeInput input) {
		if (!isConfigured()) {
			throw new IllegalStateException("ANTHROPIC_API_KEY(또는 claude.api.key)가 설정되지 않았습니다.");
		}
		try {
			ObjectNode body = buildRequestBody(input);

			String raw = webClient.post()
					.uri(CLAUDE_MESSAGES_URL)
					.header("x-api-key", apiKey)
					.header("anthropic-version", "2023-06-01")
					.header(HttpHeaders.ACCEPT, MediaType.APPLICATION_JSON_VALUE)
					.contentType(MediaType.APPLICATION_JSON)
					.bodyValue(body)
					.retrieve()
					.bodyToMono(String.class)
					.timeout(RESPONSE_TIMEOUT)
					.onErrorMap(TimeoutException.class, ex ->
							new RuntimeException("Claude API 응답이 지연되어 타임아웃이 발생했습니다. (최대 " + RESPONSE_TIMEOUT.toSeconds() + "초)", ex)
					)
					.block();

			if (raw == null || raw.isBlank()) {
				throw new RuntimeException("Claude API 응답이 비어있습니다.");
			}

			return extractTextFromResponse(raw);
		} catch (WebClientResponseException e) {
			String resp = e.getResponseBodyAsString();
			String msg = "Claude API 호출 실패 (" + e.getStatusCode().value() + ")";
			if (resp != null && !resp.isBlank()) {
				msg += ": " + resp;
			}
			throw new RuntimeException(msg, e);
		} catch (Exception e) {
			throw new RuntimeException(e.getMessage() == null ? "Claude API 호출 중 오류가 발생했습니다." : e.getMessage(), e);
		}
	}

	private String loadSystemPrompt() {
		try {
			ClassPathResource res = new ClassPathResource("prompts/analysis-system.md");
			try (InputStream is = res.getInputStream()) {
				return new String(is.readAllBytes(), StandardCharsets.UTF_8);
			}
		} catch (Exception e) {
			return VibePrompts.SYSTEM_PROMPT;
		}
	}

	private ObjectNode buildRequestBody(VibeInput input) throws Exception {
		ObjectNode root = objectMapper.createObjectNode();
		root.put("model", model);
		root.put("max_tokens", 8000);
		root.put("temperature", 0.55);
		root.put("system", loadSystemPrompt());

		ArrayNode messages = root.putArray("messages");
		ObjectNode userMsg = messages.addObject();
		userMsg.put("role", "user");
		ArrayNode content = userMsg.putArray("content");

		// 고객 사진 (필수)
		if (input.customerImageBytes != null && input.customerImageMime != null) {
			addImageBlock(content, input.customerImageBytes, input.customerImageMime);
		}
		// 레퍼런스 1 (선택)
		if (input.ref1Bytes != null && input.ref1Mime != null) {
			addImageBlock(content, input.ref1Bytes, input.ref1Mime);
		}
		// 레퍼런스 2 (선택)
		if (input.ref2Bytes != null && input.ref2Mime != null) {
			addImageBlock(content, input.ref2Bytes, input.ref2Mime);
		}

		// 텍스트 블록
		ObjectNode textBlock = content.addObject();
		textBlock.put("type", "text");
		textBlock.put("text", buildClaudeUserText(input));

		return root;
	}

	private void addImageBlock(ArrayNode content, byte[] imageBytes, String mime) throws Exception {
		byte[] resized = resizeToMaxDim(imageBytes, mime);
		String mediaType = mime.contains("png") ? "image/png" : "image/jpeg";
		String b64 = Base64.getEncoder().encodeToString(resized);

		ObjectNode imgBlock = content.addObject();
		imgBlock.put("type", "image");
		ObjectNode src = imgBlock.putObject("source");
		src.put("type", "base64");
		src.put("media_type", mediaType);
		src.put("data", b64);
	}

	private byte[] resizeToMaxDim(byte[] original, String mime) throws Exception {
		BufferedImage img = ImageIO.read(new ByteArrayInputStream(original));
		if (img == null) return original;

		int w = img.getWidth();
		int h = img.getHeight();
		if (w <= MAX_IMAGE_DIM && h <= MAX_IMAGE_DIM) return original;

		double scale = Math.min((double) MAX_IMAGE_DIM / w, (double) MAX_IMAGE_DIM / h);
		int nw = (int) Math.round(w * scale);
		int nh = (int) Math.round(h * scale);

		BufferedImage out = new BufferedImage(nw, nh, BufferedImage.TYPE_INT_RGB);
		Graphics2D g = out.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
		g.drawImage(img, 0, 0, nw, nh, null);
		g.dispose();

		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		String fmt = mime.contains("png") ? "png" : "jpeg";
		ImageIO.write(out, fmt, baos);
		return baos.toByteArray();
	}

	private String buildClaudeUserText(VibeInput input) {
		StringBuilder sb = new StringBuilder();

		// 이미지 레이블 (첨부 순서와 일치)
		int idx = 1;
		if (input.customerImageBytes != null) {
			sb.append("[Image ").append(idx++).append("] 고객 사진\n");
		}
		if (input.ref1Bytes != null) {
			sb.append("[Image ").append(idx++).append("] 레퍼런스 1\n");
		}
		if (input.ref2Bytes != null) {
			sb.append("[Image ").append(idx).append("] 레퍼런스 2\n");
		}

		// 원하는 룩/바이브 텍스트
		if (input.vibeText != null && !input.vibeText.isBlank()) {
			sb.append("\n[원하는 룩 설명]\n").append(input.vibeText.trim()).append("\n");
		}

		// Face++ 측정 데이터
		sb.append("\n[Face++ 측정 데이터]\n");
		if (input.gender != null) sb.append("gender: ").append(input.gender).append("\n");
		if (input.age != null) sb.append("age: ").append(input.age).append("\n");
		if (input.emotionSummary != null && !input.emotionSummary.isBlank()) {
			sb.append("emotion: ").append(input.emotionSummary).append("\n");
		}
		if (input.eyeDistancePx != null) sb.append("eye_distance_px: ").append(input.eyeDistancePx).append("\n");
		if (input.faceWidthHeightRatio != null) sb.append("face_width_height_ratio: ").append(input.faceWidthHeightRatio).append("\n");
		if (input.noseChinToForeheadNoseRatio != null) sb.append("nose_chin_to_forehead_nose_ratio: ").append(input.noseChinToForeheadNoseRatio).append("\n");
		if (input.leftEyeWidthPx != null) sb.append("left_eye_width_px: ").append(input.leftEyeWidthPx).append("\n");
		if (input.rightEyeWidthPx != null) sb.append("right_eye_width_px: ").append(input.rightEyeWidthPx).append("\n");
		if (input.lipThicknessPx != null) sb.append("lip_thickness_px: ").append(input.lipThicknessPx).append("\n");
		if (input.faceShape != null) sb.append("face_shape: ").append(input.faceShape).append("\n");
		if (input.bangsLikely != null) sb.append("bangs_likely: ").append(input.bangsLikely).append("\n");
		sb.append("skin_health: ").append(input.skinHealth).append(", ")
		  .append("skin_stain: ").append(input.skinStain).append(", ")
		  .append("skin_acne: ").append(input.skinAcne).append(", ")
		  .append("skin_dark_circle: ").append(input.skinDarkCircle).append("\n");

		return sb.toString();
	}

	private String extractTextFromResponse(String rawJson) throws Exception {
		JsonNode root = objectMapper.readTree(rawJson);
		JsonNode content = root.path("content");
		if (content.isArray() && content.size() > 0) {
			StringBuilder sb = new StringBuilder();
			for (JsonNode block : content) {
				if (!block.has("text")) continue;
				String t = block.get("text").asText("");
				if (t.isBlank()) continue;
				if (sb.length() > 0) sb.append('\n');
				sb.append(t);
			}
			String out = sb.toString().trim();
			if (!out.isBlank()) return out;
		}
		throw new RuntimeException("Claude 응답에서 텍스트를 찾지 못했습니다.");
	}
}
