package com.example.demo.service;

import com.example.demo.service.vibe.ModelAfterResult;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.netty.http.client.HttpClient;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Service
public class FalInstantIdService {

	private static final Logger log = LoggerFactory.getLogger(FalInstantIdService.class);
	private static final String FAL_API_URL = "https://fal.run/fal-ai/instantid";
	private static final Duration RESPONSE_TIMEOUT = Duration.ofSeconds(120);
	private static final int MAX_IMAGE_SIZE = 768;

	private static final Pattern AESTHETIC_PATTERN = Pattern.compile(
			"2\\.\\s*Aesthetic Category[:\\s]+([^\\n]+)", Pattern.CASE_INSENSITIVE);
	private static final Pattern STYLING_PATTERN = Pattern.compile(
			"4\\.\\s*Styling Direction[:\\s]+([\\s\\S]+?)(?=\\n\\d+\\.|\\[KOREAN|$)",
			Pattern.CASE_INSENSITIVE);
	private static final Pattern PINTEREST_PATTERN = Pattern.compile(
			"6\\.\\s*Pinterest Keywords[:\\s]+(([\\s\\S]+?)(?=\\[KOREAN|$))",
			Pattern.CASE_INSENSITIVE);
	private static final Pattern PINTEREST_VALUE_PATTERN = Pattern.compile(
			"-\\s*[^:]+:\\s*([^\\n]+)");

	private final WebClient webClient;
	private final ObjectMapper objectMapper;
	private final String apiKey;

	public FalInstantIdService(
			WebClient.Builder webClientBuilder,
			ObjectMapper objectMapper,
			@Value("${fal.api.key:}") String apiKey) {
		this.objectMapper = objectMapper;
		this.apiKey = apiKey == null ? "" : apiKey.trim();

		// [DEBUG] API 키 로딩 확인
		if (!this.apiKey.isBlank()) {
			String prefix = this.apiKey.length() > 8 ? this.apiKey.substring(0, 8) + "..." : this.apiKey;
			log.info("[FAL] API 키 로드 완료 - length={}, prefix={}", this.apiKey.length(), prefix);
		} else {
			log.warn("[FAL] API 키가 비어 있습니다. .env의 FAL_API_KEY를 확인하세요.");
		}

		HttpClient httpClient = HttpClient.create().responseTimeout(RESPONSE_TIMEOUT);
		this.webClient = webClientBuilder
				.clientConnector(new ReactorClientHttpConnector(httpClient))
				.build();
	}

	public boolean isConfigured() {
		return !apiKey.isBlank();
	}

	/**
	 * 모델별 after 이미지 생성 - ModelAfterResult 반환
	 */
	public ModelAfterResult generateForModel(String provider, byte[] imageBytes, String contentType, String vibeText, boolean vibeHadError) {
		if (vibeHadError || vibeText == null) {
			return new ModelAfterResult(provider, null, null, "Vibe 분석 실패로 이미지 생성 불가");
		}
		if (!isConfigured()) {
			return new ModelAfterResult(provider, null, null, "FAL_API_KEY 미설정");
		}

		String prompt = buildPrompt(vibeText);
		log.info("[FAL][{}] 이미지 생성 시작 - prompt: {}", provider, prompt);

		try {
			byte[] compressed = compressImage(imageBytes);
			String base64Image = Base64.getEncoder().encodeToString(compressed);
			String dataUri = "data:image/jpeg;base64," + base64Image;

			ObjectNode body = objectMapper.createObjectNode();
			body.put("face_image_url", dataUri);
			body.put("prompt", prompt);
			body.put("negative_prompt", "blurry, bad quality, distorted, ugly, deformed, multiple faces");
			body.put("num_inference_steps", 30);
			body.put("guidance_scale", 5.0);
			body.put("ip_adapter_scale", 0.8);
			body.put("controlnet_conditioning_scale", 0.8);

			// [DEBUG] 요청 파라미터 출력 (base64 제외)
			ObjectNode logBody = body.deepCopy();
			logBody.put("face_image_url", "[base64 생략, 압축 후 " + (compressed.length / 1024) + "KB]");
			log.info("[FAL][{}] API 요청: {}", provider, logBody);

			String responseBody = webClient.post()
					.uri(FAL_API_URL)
					.header("Authorization", "Key " + apiKey)
					.header("Content-Type", "application/json")
					.bodyValue(objectMapper.writeValueAsString(body))
					.retrieve()
					.bodyToMono(String.class)
					.timeout(RESPONSE_TIMEOUT)
					.block();

			// [DEBUG] 전체 응답 출력
			log.info("[FAL][{}] API 응답: {}", provider, responseBody);

			if (responseBody == null) {
				return new ModelAfterResult(provider, null, prompt, "응답 없음");
			}

			JsonNode root = objectMapper.readTree(responseBody);

			// 응답 구조 우선순위: image.url → images[0].url → images[0] (string)
			String url = null;

			JsonNode imageNode = root.path("image");
			if (imageNode.isObject()) {
				url = imageNode.path("url").asText(null);
			}

			if (url == null) {
				JsonNode images = root.path("images");
				if (images.isArray() && images.size() > 0) {
					JsonNode first = images.get(0);
					if (first.isObject()) {
						url = first.path("url").asText(null);
					} else if (first.isTextual()) {
						url = first.asText(null);
					}
				}
			}

			if (url != null && !url.isBlank()) {
				log.info("[FAL][{}] 생성 완료: {}", provider, url);
				return new ModelAfterResult(provider, url, prompt, null);
			}

			return new ModelAfterResult(provider, null, prompt, "응답에 이미지 없음: " + responseBody);

		} catch (WebClientResponseException e) {
			String errBody = e.getResponseBodyAsString();
			log.error("[FAL][{}] HTTP {} 오류: {}", provider, e.getStatusCode(), errBody);

			// fal.ai 에러 메시지 추출
			String detail = errBody;
			try {
				JsonNode errNode = objectMapper.readTree(errBody);
				if (errNode.has("detail")) detail = errNode.get("detail").asText();
			} catch (Exception ignored) {}

			return new ModelAfterResult(provider, null, prompt, e.getStatusCode() + " - " + detail);
		} catch (Exception e) {
			log.error("[FAL][{}] 호출 실패", provider, e);
			return new ModelAfterResult(provider, null, prompt, e.getMessage());
		}
	}

	public String buildPrompt(String vibeText) {
		if (vibeText == null || vibeText.isBlank()) {
			return "editorial portrait, soft studio lighting, clean aesthetic";
		}

		// 영어 파트만 사용
		String english = vibeText;
		int koreanIdx = vibeText.indexOf("[KOREAN_START]");
		if (koreanIdx > 0) {
			english = vibeText.substring(0, koreanIdx);
		}

		// 2. Aesthetic Category
		String aesthetic = "";
		Matcher m1 = AESTHETIC_PATTERN.matcher(english);
		if (m1.find()) {
			aesthetic = m1.group(1).trim();
		}

		// 4. Styling Direction (최대 2문장)
		String styling = "";
		Matcher m2 = STYLING_PATTERN.matcher(english);
		if (m2.find()) {
			styling = m2.group(1).replaceAll("\\s+", " ").trim();
			String[] sentences = styling.split("(?<=[.!?])\\s+");
			if (sentences.length > 2) {
				styling = sentences[0] + " " + sentences[1];
			}
		}

		// 6. Pinterest Keywords 값들 추출
		List<String> pinterestValues = new ArrayList<>();
		Matcher m3 = PINTEREST_PATTERN.matcher(english);
		if (m3.find()) {
			String pinterestBlock = m3.group(1);
			Matcher mv = PINTEREST_VALUE_PATTERN.matcher(pinterestBlock);
			while (mv.find()) {
				String val = mv.group(1).trim();
				if (!val.isBlank()) pinterestValues.add(val);
			}
		}

		StringBuilder prompt = new StringBuilder();
		if (!aesthetic.isEmpty()) {
			prompt.append(aesthetic).append(" aesthetic");
		}
		if (!styling.isEmpty()) {
			if (!prompt.isEmpty()) prompt.append(", ");
			prompt.append(styling);
		}
		if (!pinterestValues.isEmpty()) {
			if (!prompt.isEmpty()) prompt.append(", ");
			prompt.append(String.join(", ", pinterestValues));
		}
		if (!prompt.isEmpty()) prompt.append(", ");
		prompt.append("soft studio lighting, editorial portrait, high quality");

		return prompt.toString();
	}

	private byte[] compressImage(byte[] original) {
		try {
			BufferedImage src = ImageIO.read(new ByteArrayInputStream(original));
			if (src == null) return original;

			int w = src.getWidth();
			int h = src.getHeight();
			if (w > MAX_IMAGE_SIZE || h > MAX_IMAGE_SIZE) {
				double scale = (double) MAX_IMAGE_SIZE / Math.max(w, h);
				w = (int) (w * scale);
				h = (int) (h * scale);
			}

			BufferedImage resized = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
			Graphics2D g2d = resized.createGraphics();
			g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
			g2d.drawImage(src.getScaledInstance(w, h, Image.SCALE_SMOOTH), 0, 0, null);
			g2d.dispose();

			ImageWriter writer = ImageIO.getImageWritersByFormatName("jpeg").next();
			ImageWriteParam param = writer.getDefaultWriteParam();
			param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
			param.setCompressionQuality(0.85f);

			ByteArrayOutputStream out = new ByteArrayOutputStream();
			writer.setOutput(ImageIO.createImageOutputStream(out));
			writer.write(null, new IIOImage(resized, null, null), param);
			writer.dispose();

			return out.toByteArray();
		} catch (Exception e) {
			log.warn("이미지 압축 실패, 원본 사용", e);
			return original;
		}
	}
}
