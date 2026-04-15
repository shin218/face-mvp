package com.example.demo.service;

import com.example.demo.service.vibe.VibeInput;
import com.example.demo.service.vibe.VibePrompts;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import org.springframework.web.util.UriComponentsBuilder;
import reactor.netty.http.client.HttpClient;

import java.time.Duration;
import java.util.concurrent.TimeoutException;

@Service
public class GeminiVibeService {

	private static final String BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/";
	private static final Duration RESPONSE_TIMEOUT = Duration.ofSeconds(60);
	private static final int RETRIES_503 = 3;
	private static final String MSG_503_EXHAUSTED = "Gemini 일시적 오류, 잠시 후 다시 시도해주세요";

	private final WebClient webClient;
	private final ObjectMapper objectMapper;
	private final String apiKey;
	private final String modelId;
	private final String fallbackModelId;

	public GeminiVibeService(
			WebClient.Builder webClientBuilder,
			ObjectMapper objectMapper,
			@Value("${gemini.api.key:}") String apiKey,
			@Value("${gemini.api.model:gemini-2.5-pro}") String modelId,
			@Value("${gemini.api.fallbackModel:gemini-2.5-flash}") String fallbackModelId
	) {
		this.objectMapper = objectMapper;
		this.apiKey = apiKey == null ? "" : apiKey.trim();
		this.modelId = modelId;
		this.fallbackModelId = fallbackModelId;

		HttpClient httpClient = HttpClient.create().responseTimeout(RESPONSE_TIMEOUT);
		this.webClient = webClientBuilder
				.clientConnector(new ReactorClientHttpConnector(httpClient))
				.build();
	}

	public boolean isConfigured() {
		return apiKey != null && !apiKey.isBlank();
	}

	public String analyzeVibe(VibeInput input) {
		if (!isConfigured()) {
			throw new IllegalStateException("GEMINI_API_KEY(또는 gemini.api.key)가 설정되지 않았습니다.");
		}
		ObjectNode body = buildRequestBody(input);
		try {
			return callModelWith503Policy(modelId, body);
		} catch (RuntimeException e) {
			if (MSG_503_EXHAUSTED.equals(e.getMessage())
					&& fallbackModelId != null
					&& !fallbackModelId.isBlank()
					&& !fallbackModelId.equals(modelId)) {
				return callModelWith503Policy(fallbackModelId, body);
			}
			throw e;
		}
	}

	private String callModelWith503Policy(String model, ObjectNode body) {
		String uri = UriComponentsBuilder
				.fromUriString(BASE_URL + model + ":generateContent")
				.queryParam("key", apiKey)
				.build()
				.toUriString();

		try {
			return postWith503Retries(uri, body);
		} catch (WebClientResponseException e) {
			String resp = e.getResponseBodyAsString();
			String msg = "Gemini API 호출 실패 (" + e.getStatusCode().value() + ")";
			if (resp != null && !resp.isBlank()) {
				msg += ": " + resp;
			}
			throw new RuntimeException(msg, e);
		} catch (Exception e) {
			throw new RuntimeException(e.getMessage() == null ? "Gemini API 호출 중 오류가 발생했습니다." : e.getMessage(), e);
		}
	}

	/**
	 * HTTP 503일 때만 최대 3회 재시도. 대기: 1초, 2초, 3초 (시도 간격).
	 * 초기 1회 + 재시도 3회 = 총 4번 호출까지.
	 */
	private String postWith503Retries(String uri, ObjectNode body) throws Exception {
		for (int attempt = 0; attempt <= RETRIES_503; attempt++) {
			try {
				String raw = webClient.post()
						.uri(uri)
						.contentType(MediaType.APPLICATION_JSON)
						.bodyValue(body)
						.retrieve()
						.bodyToMono(String.class)
						.timeout(RESPONSE_TIMEOUT)
						.onErrorMap(TimeoutException.class, ex ->
								new RuntimeException("Gemini API 응답이 지연되어 타임아웃이 발생했습니다. (최대 " + RESPONSE_TIMEOUT.toSeconds() + "초)", ex)
						)
						.block();

				if (raw == null || raw.isBlank()) {
					throw new RuntimeException("Gemini API 응답이 비어있습니다.");
				}
				return extractText(raw);
			} catch (WebClientResponseException e) {
				int code = e.getStatusCode().value();
				if (code == 503 && attempt < RETRIES_503) {
					long waitMs = (attempt + 1) * 1000L;
					try {
						Thread.sleep(waitMs);
					} catch (InterruptedException ie) {
						Thread.currentThread().interrupt();
					}
					continue;
				}
				if (code == 503) {
					throw new RuntimeException(MSG_503_EXHAUSTED, e);
				}
				throw e;
			}
		}
		throw new IllegalStateException("Gemini postWith503Retries: expected return or throw inside loop");
	}

	private ObjectNode buildRequestBody(VibeInput input) {
		ObjectNode root = objectMapper.createObjectNode();

		ObjectNode systemInstruction = root.putObject("systemInstruction");
		ArrayNode sysParts = systemInstruction.putArray("parts");
		sysParts.addObject().put("text", VibePrompts.SYSTEM_PROMPT);

		ArrayNode contents = root.putArray("contents");
		ObjectNode turn = contents.addObject();
		turn.put("role", "user");
		ArrayNode parts = turn.putArray("parts");
		parts.addObject().put("text", VibePrompts.buildUserText(input));

		ObjectNode gen = root.putObject("generationConfig");
		gen.put("temperature", 0.55);
		gen.put("maxOutputTokens", 4096);

		return root;
	}

	private String extractText(String rawJson) throws Exception {
		JsonNode root = objectMapper.readTree(rawJson);

		JsonNode block = root.path("promptFeedback").path("blockReason");
		if (!block.isMissingNode() && !block.asText("").isBlank()) {
			throw new RuntimeException("Gemini가 프롬프트를 차단했습니다: " + block.asText());
		}

		JsonNode candidates = root.path("candidates");
		if (!candidates.isArray() || candidates.isEmpty()) {
			throw noTextException(root, "candidates 비어 있음");
		}

		StringBuilder nonThought = new StringBuilder();
		StringBuilder fallback = new StringBuilder();
		for (JsonNode cand : candidates) {
			JsonNode parts = cand.path("content").path("parts");
			if (!parts.isArray()) {
				continue;
			}
			for (JsonNode p : parts) {
				if (!p.has("text")) {
					continue;
				}
				String t = p.get("text").asText("");
				if (t.isBlank()) {
					continue;
				}
				if (fallback.length() > 0) {
					fallback.append("\n");
				}
				fallback.append(t);
				boolean thought = p.path("thought").asBoolean(false);
				if (!thought) {
					if (nonThought.length() > 0) {
						nonThought.append("\n");
					}
					nonThought.append(t);
				}
			}
		}

		String out = nonThought.toString().trim();
		if (!out.isBlank()) {
			return out;
		}
		out = fallback.toString().trim();
		if (!out.isBlank()) {
			return out;
		}

		throw noTextException(root, "유효한 text part 없음");
	}

	private static RuntimeException noTextException(JsonNode root, String hint) {
		JsonNode c0 = root.path("candidates").path(0);
		String finish = c0.path("finishReason").asText("");
		String refusal = c0.path("refusal").asText("");
		String safety = c0.path("safetyRatings").toString();
		String msg = "Gemini 응답에서 텍스트를 찾지 못했습니다. (" + hint + ")";
		if (!finish.isBlank()) {
			msg += " finishReason=" + finish;
		}
		if (!refusal.isBlank()) {
			msg += " refusal=" + refusal;
		}
		if (!c0.path("safetyRatings").isMissingNode() && c0.path("safetyRatings").isArray() && c0.path("safetyRatings").size() > 0) {
			msg += " safetyRatings=" + safety;
		}
		return new RuntimeException(msg);
	}
}
